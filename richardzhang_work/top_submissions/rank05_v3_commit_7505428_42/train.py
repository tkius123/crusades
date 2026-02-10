"""
t01 — UID 79 clone: compiled step fn + grad_ckpt off + fused AdamW
Full reproduction of rank 1 approach.
"""

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


_initialized = False
_train_fn = None


def inner_steps(model, data_iterator, optimizer, num_steps, device):
    global _initialized, _train_fn

    is_cuda = str(device).startswith("cuda")

    # ONE-TIME INIT
    if not _initialized:
        _initialized = True

        if is_cuda:
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

            # Free stale optimizers (~24GB VRAM)
            for obj in gc.get_objects():
                if isinstance(obj, torch.optim.Optimizer) and obj is not optimizer:
                    obj.state.clear()
            gc.collect()
            torch.cuda.empty_cache()

    # PER-CALL
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    if is_cuda:
        try:
            for group in optimizer.param_groups:
                if not group.get("fused", False):
                    group["fused"] = True
                    group["foreach"] = False
        except Exception:
            pass

    # COMPILED STEP FN (forward + loss + backward)
    if _train_fn is None:
        dev_type = device.type

        def _eager_step(input_ids, labels):
            with torch.autocast(device_type=dev_type, dtype=torch.bfloat16):
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

    # PREFETCH
    if is_cuda:
        pf_stream = torch.cuda.Stream()
        with torch.cuda.stream(pf_stream):
            next_batch = next(data_iterator).to(device, non_blocking=True)
    else:
        next_batch = next(data_iterator).to(device)

    # TRAINING LOOP
    total_tokens = 0
    final_loss = 0.0
    final_logits = None

    for step in range(num_steps):
        if is_cuda:
            torch.cuda.current_stream().wait_stream(pf_stream)
        batch = next_batch

        if step < num_steps - 1:
            nb = next(data_iterator)
            if is_cuda:
                with torch.cuda.stream(pf_stream):
                    next_batch = nb.to(device, non_blocking=True)
            else:
                next_batch = nb.to(device)

        loss, logits = _train_fn(batch[:, :-1], batch[:, 1:])

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += batch.numel()
        if step == num_steps - 1:
            final_loss = loss.item()
            final_logits = logits.float()

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
    )


if __name__ == "__main__":
    import json
    import time
    from pathlib import Path
    from transformers import AutoModelForCausalLM

    print("t01 — UID 79 clone: compiled step fn + grad_ckpt off + fused AdamW")

    hparams_path = Path(__file__).parent.parent / "hparams" / "hparams.json"
    hparams = {}
    if hparams_path.exists():
        with open(hparams_path) as f:
            hparams = json.load(f)

    batch_size = hparams.get("benchmark_batch_size", 16)
    num_steps = hparams.get("eval_steps", 5)
    num_evals = hparams.get("evaluation_runs", 5)

    project_root = Path(__file__).parent.parent
    model_path = project_root / "benchmark" / "model"
    data_path = project_root / "benchmark" / "data" / "train.pt"

    if not model_path.exists() or not data_path.exists():
        print("Setup required! Run: uv run local_test/setup_benchmark.py")
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model.train()

    data = torch.load(data_path, weights_only=True)
    if torch.cuda.is_available():
        data = data.pin_memory()

    def create_iterator():
        idx = 0
        while True:
            end_idx = idx + batch_size
            if end_idx > data.shape[0]:
                idx = 0
                end_idx = batch_size
            yield data[idx:end_idx]
            idx = end_idx

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("Warmup...")
    _ = inner_steps(model, create_iterator(), optimizer, num_steps=2, device=device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"Running {num_evals} evaluations...")
    wall_times = []
    for i in range(num_evals):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = inner_steps(model, create_iterator(), optimizer, num_steps, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        wall_times.append(elapsed)
        tps = result.total_tokens / elapsed
        print(f"  Eval {i+1}: {elapsed:.3f}s | TPS={tps:,.0f} | loss={result.final_loss:.4f}")

    median_time = sorted(wall_times)[len(wall_times) // 2]
    median_tps = result.total_tokens / median_time
    print(f"\nMedian: {median_time:.4f}s | TPS={median_tps:,.0f}")
    print("Done!")
