"""
Reference training implementation for Templar Crusades.

This is the baseline implementation. Miners should optimize it for maximum MFU
(Model FLOPs Utilization) while passing all verification checks.

Usage:
    1. Run setup: uv run local_test/setup_benchmark.py
    2. Test locally: uv run local_test/train.py
    3. Verify locally: uv run local_test/verify.py
    4. Submit this file (or your optimized version) as a GitHub Gist!

=== SUBMISSION ===

You can submit this entire file as-is. The validator only calls the inner_steps
function — the `if __name__ == "__main__":` block is for local testing and is
ignored during evaluation.

=== VERIFICATION RULES ===

Your inner_steps function MUST:
  - Use the provided optimizer (call optimizer.step() and optimizer.zero_grad())
  - Process ALL tokens in each batch (no truncation)
  - Return actual final_logits tensor (not None)
  - Return logits with correct shape: (batch_size, seq_len - 1, vocab_size)
  - Produce gradients that closely match the reference implementation
  - Train all model parameters (don't freeze layers)
  - Call optimizer.step() for each training step

Your inner_steps function MUST NOT:
  - Access optimizer internals (e.g., optimizer.optimizer)
  - Truncate or skip parts of input sequences
  - Return None for final_logits
  - Report inflated token counts
  - Modify the model's requires_grad settings
  - Modify torch backend settings (deterministic, benchmark, SDP toggles, etc.)
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
    """Required return type from inner_steps function.

    All fields are verified by the validator:
    - final_logits: Must be a 3D tensor (batch, seq_len-1, vocab), NOT None
    - total_tokens: Should equal batch_size * seq_len * num_steps
    - final_loss: Must be a positive float, close to reference loss
    """

    final_logits: torch.Tensor  # Output logits from last forward pass
    total_tokens: int  # Total tokens processed across all steps
    final_loss: float  # Loss value from last training step


def inner_steps(model, data_iterator, optimizer, num_steps, device):
    """
    Run training steps and return results.

    This is the function the validator calls. It receives:
    - model: Pre-loaded model (already on device, in train mode, with gradient checkpointing)
    - data_iterator: Infinite iterator yielding batches of shape (batch_size, seq_len)
    - optimizer: Pre-configured AdamW optimizer (wrapped by validator for gradient capture)
    - num_steps: Number of training steps to run (must complete all of them)
    - device: Target device (cuda or cpu)

    The validator measures wall_time of this function and calculates:
        MFU = (6 * model_params * batch_size * seq_len * num_steps) / (wall_time * gpu_peak_tflops)

    Higher MFU = you completed the same training faster = better score.

    Returns:
        InnerStepsResult with outputs for verification
    """
    total_tokens = 0
    final_logits = None
    final_loss = 0.0

    for step in range(num_steps):
        # Get batch - shape: (batch_size, seq_len)
        batch = next(data_iterator)
        batch = batch.to(device, dtype=torch.long)

        # Prepare inputs and labels (causal LM: predict next token)
        # input_ids: all tokens except last, labels: all tokens except first
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]

        # Forward pass
        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        # Backward pass
        loss.backward()

        # Update weights - MUST use the provided optimizer
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Track metrics
        total_tokens += batch.numel()
        # Keep logits from the last step for verification
        final_logits = logits.detach().float()
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
    model.gradient_checkpointing_enable()  # Required to fit in GPU memory
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

    # Warmup
    print("Warmup...")
    _ = inner_steps(model, create_iterator(), optimizer, num_steps=2, device=device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print()

    # Reset optimizer (same config as validator)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=0.1, betas=(0.9, 0.95), fused=use_fused
    )

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
