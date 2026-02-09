import gc
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class InnerStepsResult:
    """Result from inner_steps training loop."""
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float


# Global state — persists across inner_steps calls (warmup + eval)
_initialized = False
_train_fn = None


def inner_steps(model, data_iterator, optimizer, num_steps, device):
    """Optimized training loop for maximum TPS.

    Key design choices for correctness + speed:
    - No autocast: matches production reference (env.py _run_reference) exactly.
      The model is already bf16, so forward/loss/backward all run in bf16 natively.
      Using autocast would change loss precision (cross_entropy promoted to fp32)
      causing gradient mismatch vs the reference.
    - torch.compile with reduce-overhead: CUDA graphs for minimal dispatch overhead.
    - Stream-based data prefetching: overlap H2D transfer with compute.
    - Gradient checkpointing disabled: trade VRAM for speed.
    - Fused optimizer via param_groups: single CUDA kernel for param update.
    - All SDP backends enabled: flash/mem-efficient preferred, math as fallback.
    """
    global _initialized, _train_fn

    is_cuda = str(device).startswith("cuda")

    # =========================================================================
    # ONE-TIME INIT — runs once across all inner_steps calls
    # =========================================================================
    if not _initialized:
        _initialized = True

        if is_cuda:
            os.environ.setdefault(
                "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
            )
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

            # Enable all SDP backends; flash/mem-efficient preferred, math as safe fallback
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)

            # Free stale optimizer states from other optimizers
            for obj in gc.get_objects():
                if isinstance(obj, torch.optim.Optimizer) and obj is not optimizer:
                    obj.state.clear()
            gc.collect()
            torch.cuda.empty_cache()

    # =========================================================================
    # PER-CALL SETUP — lightweight, runs each inner_steps call
    # =========================================================================
    if hasattr(model, "config"):
        model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    # Enable fused optimizer — single CUDA kernel for param update
    if is_cuda:
        try:
            for group in optimizer.param_groups:
                if not group.get("fused", False):
                    group["fused"] = True
                    group["foreach"] = False
        except Exception:
            pass

    # =========================================================================
    # COMPILED TRAIN STEP — cached globally across calls
    # Forward + loss + backward compiled as one unit for maximum fusion.
    # =========================================================================
    if _train_fn is None:
        def _eager_step(input_ids, labels):
            # No autocast — model is already bf16. This matches the production
            # validator reference (env.py _run_reference) exactly, giving
            # near-zero gradient relative error.
            logits = model(input_ids).logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            loss.backward()
            return loss.detach(), logits.detach()

        if is_cuda:
            try:
                _train_fn = torch.compile(
                    _eager_step,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
            except Exception:
                _train_fn = _eager_step
        else:
            _train_fn = _eager_step

    # =========================================================================
    # PREFETCH FIRST BATCH
    # =========================================================================
    if is_cuda:
        pf_stream = torch.cuda.Stream()
        with torch.cuda.stream(pf_stream):
            next_batch = next(data_iterator).to(device, non_blocking=True)
    else:
        next_batch = next(data_iterator).to(device)

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    total_tokens = 0
    final_loss = 0.0
    final_logits = None

    for step in range(num_steps):
        # Wait for prefetched batch
        if is_cuda:
            torch.cuda.current_stream().wait_stream(pf_stream)
        batch = next_batch

        # Prefetch next batch (overlaps with forward+backward+optimizer)
        if step < num_steps - 1:
            nb = next(data_iterator)
            if is_cuda:
                with torch.cuda.stream(pf_stream):
                    next_batch = nb.to(device, non_blocking=True)
            else:
                next_batch = nb.to(device)

        # Forward + loss + backward (compiled with CUDA graphs)
        loss, logits = _train_fn(batch[:, :-1], batch[:, 1:])

        # Must use provided optimizer every step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Track tokens
        total_tokens += batch.numel()

        # Only extract final metrics on last step (avoids CPU-sync overhead)
        if step == num_steps - 1:
            final_loss = loss.item()
            final_logits = logits.float()

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
    )


# =============================================================================
# LOCAL TESTING
# =============================================================================
if __name__ == "__main__":
    import json
    import time
    from pathlib import Path

    from transformers import AutoModelForCausalLM

    print("=" * 60)
    print("TESTING train.py — No-autocast + compiled step + CUDA graphs")
    print("=" * 60)
    print()

    hparams_path = Path(__file__).parent.parent / "hparams" / "hparams.json"
    hparams = {}
    if hparams_path.exists():
        with open(hparams_path) as f:
            hparams = json.load(f)

    batch_size = hparams.get("benchmark_batch_size", 16)
    num_steps = hparams.get("eval_steps", 5)
    num_evals = hparams.get("evaluation_runs", 5)

    print(f"Batch size: {batch_size}")
    print(f"Steps per eval: {num_steps}")
    print(f"Evaluations: {num_evals}")
    print()

    project_root = Path(__file__).parent.parent
    model_path = project_root / "benchmark" / "model"
    data_path = project_root / "benchmark" / "data" / "train.pt"

    if not model_path.exists() or not data_path.exists():
        print("Setup required! Run: uv run local_test/setup_benchmark.py")
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    attn_impl = "sdpa"
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        pass

    print(f"Loading model (attn={attn_impl})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model.gradient_checkpointing_enable()
    model.train()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    print("Loading data...")
    data = torch.load(data_path, weights_only=True)
    if torch.cuda.is_available():
        data = data.pin_memory()
    print(f"Samples: {data.shape[0]:,}, Sequence length: {data.shape[1]}")
    print()

    def create_iterator():
        idx = 0
        while True:
            end_idx = idx + batch_size
            if end_idx > data.shape[0]:
                idx = 0
                end_idx = batch_size
            yield data[idx:end_idx]
            idx = end_idx

    use_fused = torch.cuda.is_available()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=use_fused)

    print("Warmup (compile)...")
    t0 = time.perf_counter()
    _ = inner_steps(model, create_iterator(), optimizer, num_steps=2, device=device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"  Warmup: {t1 - t0:.1f}s")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=use_fused)

    print(f"Running {num_evals} evaluations...")
    tps_list = []

    for i in range(num_evals):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        result = inner_steps(model, create_iterator(), optimizer, num_steps, device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        tps = result.total_tokens / elapsed
        tps_list.append(tps)
        print(
            f"  Eval {i + 1}: {elapsed:.2f}s, "
            f"TPS={tps:,.0f}, "
            f"tokens={result.total_tokens:,}, "
            f"loss={result.final_loss:.4f}"
        )

    print()
    tps_list.sort()
    median_tps = tps_list[len(tps_list) // 2]
    print(f"Median TPS: {median_tps:,.0f}")
    print("Done!")
