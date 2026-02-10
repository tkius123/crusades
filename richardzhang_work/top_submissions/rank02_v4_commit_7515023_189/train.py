"""
I'm Am1l dont pls copy
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM


@dataclass
class InnerStepsResult:
    """Required return type from inner_steps function."""

    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float

_compiled_cache = {}


def inner_steps(model, data_iterator, optimizer, num_steps, device):
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    if hasattr(model, "config"):
        model.config.use_cache = False
        model.config.output_hidden_states = False
        model.config.output_attentions = False

    model_id = id(model)
    if model_id not in _compiled_cache:
        torch._dynamo.config.suppress_errors = True
        _compiled_cache[model_id] = torch.compile(model, dynamic=False)
    compiled_model = _compiled_cache[model_id]

    # === Training loop ===
    total_tokens = 0
    final_logits = None
    final_loss = 0.0
    last_step = num_steps - 1

    for step in range(num_steps):
        batch = next(data_iterator)
        batch = batch.to(device, dtype=torch.long)

        input_ids = batch[:, :-1]
        labels = batch[:, 1:]

        # Forward pass through compiled model
        outputs = compiled_model(input_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        # Backward + optimizer step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Token count (cheap: just tensor shape, no GPU sync)
        total_tokens += batch.numel()
        
        if step == last_step:
            final_logits = logits.detach()
            final_loss = loss.item()

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
    print("TESTING train.py - Optimized Implementation")
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

    # Load model with RANDOM initialization (same as validator)
    print("Loading model (random init, same as validator)...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model = model.to(device)
    model.gradient_checkpointing_enable()  # Validator enables this on load
    model.train()
    for param in model.parameters():
        param.requires_grad = True
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

    # Create optimizer (same config as validator)
    use_fused = torch.cuda.is_available()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=0.1, betas=(0.9, 0.95), fused=use_fused
    )

    # Warmup (same as validator: 2 steps to prime torch.compile cache)
    print("Warmup (compiling Triton kernels)...")
    _ = inner_steps(model, create_iterator(), optimizer, num_steps=2, device=device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print("Warmup done.")
    print()

    # Reset optimizer (same config as validator)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=0.1, betas=(0.9, 0.95), fused=use_fused
    )

    # Run evaluations
    model_params = sum(p.numel() for p in model.parameters())
    gpu_peak_tflops = 312.0
    print(f"Running {num_evals} evaluations...")

    for i in range(num_evals):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        result = inner_steps(model, create_iterator(), optimizer, num_steps, device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        # Calculate MFU
        flops = 6 * model_params * result.total_tokens
        actual_tflops = flops / elapsed / 1e12
        mfu = (actual_tflops / gpu_peak_tflops) * 100

        print(
            f"  Eval {i + 1}: {elapsed:.2f}s, tokens={result.total_tokens:,}, "
            f"loss={result.final_loss:.4f}, MFU={mfu:.1f}%"
        )

    print()
    print("Done!")
