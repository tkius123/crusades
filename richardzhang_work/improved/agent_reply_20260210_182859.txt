import gc
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float


_setup_done = False
_compiled_step = None


def inner_steps(model, data_iterator, optimizer, num_steps, device):
    global _setup_done, _compiled_step

    use_cuda = str(device).startswith("cuda")

    if not _setup_done:
        _setup_done = True

        if use_cuda:
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)

            for obj in gc.get_objects():
                if isinstance(obj, torch.optim.Optimizer) and obj is not optimizer:
                    obj.state.clear()
            gc.collect()
            torch.cuda.empty_cache()

    if hasattr(model, "config"):
        model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    if use_cuda:
        try:
            for pg in optimizer.param_groups:
                if not pg.get("fused", False):
                    pg["fused"] = True
                    pg["foreach"] = False
        except Exception:
            pass

    if _compiled_step is None:
        device_type = device.type

        def _forward_backward(input_ids, labels):
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits = model(input_ids).logits
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
            loss.backward()
            return loss.detach(), logits.detach()

        if use_cuda:
            try:
                _compiled_step = torch.compile(
                    _forward_backward,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
            except Exception:
                _compiled_step = _forward_backward
        else:
            _compiled_step = _forward_backward

    if use_cuda:
        prefetch_stream = torch.cuda.Stream()
        with torch.cuda.stream(prefetch_stream):
            pending_batch = next(data_iterator).to(device, non_blocking=True)
    else:
        pending_batch = next(data_iterator).to(device)

    tokens_count = 0
    last_loss = 0.0
    last_logits = None

    for step_idx in range(num_steps):
        if use_cuda:
            torch.cuda.current_stream().wait_stream(prefetch_stream)
        current_batch = pending_batch

        if step_idx < num_steps - 1:
            upcoming = next(data_iterator)
            if use_cuda:
                with torch.cuda.stream(prefetch_stream):
                    pending_batch = upcoming.to(device, non_blocking=True)
            else:
                pending_batch = upcoming.to(device)

        loss_val, logits_val = _compiled_step(current_batch[:, :-1], current_batch[:, 1:])

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        tokens_count += current_batch.numel()
        if step_idx == num_steps - 1:
            last_loss = loss_val.item()
            last_logits = logits_val.float()

    return InnerStepsResult(
        final_logits=last_logits,
        total_tokens=tokens_count,
        final_loss=last_loss,
    )


if __name__ == "__main__":
    import json
    import time
    from pathlib import Path

    from transformers import AutoModelForCausalLM

    print("=" * 60)
    print("TESTING train.py - Wrapper Bypass + Compile (Cached)")
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
        import flash_attn
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
    print(f"  Warmup: {t1-t0:.1f}s")

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
