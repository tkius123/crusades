"""
Basic training implementation - Miners can optimize this!

Usage:
    1. Run setup: uv run local_test/setup_benchmark.py
    2. Test locally: uv run local_test/train.py
    3. Submit when ready!
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


@dataclass
class InnerStepsResult:
    """Required return type from inner_steps function."""

    final_logits: torch.Tensor  # Output logits from last forward pass
    total_tokens: int  # Total tokens processed across all steps
    final_loss: float  # Loss value from last training step


_COMPILED_MODELS: dict[int, torch.nn.Module] = {}
_CUDA_STREAMS: dict[int, torch.cuda.Stream] = {}


def _get_execution_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return a cached compiled wrapper when available.

    The validator calls `inner_steps` twice per run (warmup + timed pass) with
    the same model object. Caching the compiled wrapper by model id avoids
    paying compile overhead during the timed pass.
    """
    if not torch.cuda.is_available() or not hasattr(torch, "compile"):
        return model

    model_id = id(model)
    compiled = _COMPILED_MODELS.get(model_id)
    if compiled is not None:
        return compiled

    try:
        compiled = torch.compile(
            model,
            mode="max-autotune",
            fullgraph=False,
            dynamic=False,
        )
        _COMPILED_MODELS[model_id] = compiled
        return compiled
    except Exception:
        return model


def _get_cuda_stream(device: torch.device) -> torch.cuda.Stream | None:
    """Return one reusable CUDA stream per device index for async host->device copies."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None

    index = device.index if device.index is not None else torch.cuda.current_device()
    stream = _CUDA_STREAMS.get(index)
    if stream is None:
        stream = torch.cuda.Stream(device=index)
        _CUDA_STREAMS[index] = stream
    return stream


def inner_steps(model, data_iterator, optimizer, num_steps, device):
    """
    Run training steps and return results.

    Args:
        model: Pre-loaded model (already on device, in train mode)
        data_iterator: Iterator yielding batches of shape (batch_size, seq_len)
        optimizer: Pre-configured optimizer
        num_steps: Number of training steps to run
        device: Target device (cuda or cpu)

    Returns:
        InnerStepsResult with outputs for verification
    """
    # Keep behavior deterministic for validator verification.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Keep math path aligned with validator reference, but enable fast kernels.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    run_model = _get_execution_model(model)
    stream = _get_cuda_stream(device)
    total_tokens = 0
    final_logits = None
    final_loss = 0.0

    # Preload first batch.
    batch = next(data_iterator)
    if stream is not None:
        with torch.cuda.stream(stream):
            batch = batch.to(device, dtype=torch.long, non_blocking=True)
        torch.cuda.current_stream().wait_stream(stream)
    else:
        batch = batch.to(device, dtype=torch.long, non_blocking=True)

    for step in range(num_steps):
        # Prepare inputs and labels (process ALL tokens - no truncation)
        input_ids = batch[:, :-1].contiguous()
        labels = batch[:, 1:].contiguous()

        # Prefetch next batch while current step computes.
        if step + 1 < num_steps:
            next_batch = next(data_iterator)
            if stream is not None:
                with torch.cuda.stream(stream):
                    next_batch = next_batch.to(device, dtype=torch.long, non_blocking=True)
            else:
                next_batch = next_batch.to(device, dtype=torch.long, non_blocking=True)
        else:
            next_batch = None

        # Forward pass
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = run_model(input_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Track metrics
        total_tokens += batch.numel()
        final_logits = logits.detach().float()
        final_loss = loss.item()

        if next_batch is not None:
            if stream is not None:
                torch.cuda.current_stream().wait_stream(stream)
            batch = next_batch

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
    )


# =============================================================================
# LOCAL TESTING - Run this file to test your implementation
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING train.py - Basic Implementation")
    print("=" * 60)
    print()

    # Load configuration
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

    # Check paths
    project_root = Path(__file__).parent.parent
    model_path = project_root / "benchmark" / "model"
    data_path = project_root / "benchmark" / "data" / "train.pt"

    if not model_path.exists() or not data_path.exists():
        print("Setup required! Run: uv run local_test/setup_benchmark.py")
        exit(1)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()  # Required to fit in GPU memory
    model.train()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Load data
    print("Loading data...")
    data = torch.load(data_path, weights_only=True)
    print(f"Samples: {data.shape[0]:,}, Sequence length: {data.shape[1]}")
    print()

    # Create data iterator
    def create_iterator():
        idx = 0
        while True:
            end_idx = idx + batch_size
            if end_idx > data.shape[0]:
                idx = 0
                end_idx = batch_size
            yield data[idx:end_idx]
            idx = end_idx

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    print("Warmup...")
    _ = inner_steps(model, create_iterator(), optimizer, num_steps=2, device=device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print()

    # Reset optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Run evaluations
    print(f"Running {num_evals} evaluations...")

    for i in range(num_evals):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        result = inner_steps(model, create_iterator(), optimizer, num_steps, device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        print(
            f"  Eval {i + 1}: {elapsed:.2f}s, tokens={result.total_tokens:,}, loss={result.final_loss:.4f}"
        )

    print()
    print("Done!")