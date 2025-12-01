# Miner Guide

This guide explains how to submit optimized training code to the Templar Tournament.

## Overview

Miners compete by submitting training code that is evaluated on **tokens per second (TPS)**. The miner with the highest TPS receives 100% of subnet emissions.

**Important**: Your code must produce outputs that match the reference implementation. We verify that you're actually training by comparing:
- Output logits from the last step
- Total token count
- Final loss value

## Requirements

- Python 3.12+
- uv package manager
- Bittensor wallet with registered hotkey on subnet 3

## Installation

```bash
git clone https://github.com/your-org/templar-tournament.git
cd templar-tournament
uv sync
```

## Submission Format

Create a `train.py` file with a single **required function**: `inner_steps`

```python
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterator


@dataclass
class InnerStepsResult:
    """Result from inner_steps function."""
    final_logits: torch.Tensor  # Output logits from last forward pass
    total_tokens: int           # Total tokens processed
    final_loss: float           # Loss from last step


def inner_steps(
    model: torch.nn.Module,
    data_iterator: Iterator[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    device: torch.device,
) -> InnerStepsResult:
    """
    Execute N training steps and return results for verification.

    Your implementation MUST produce the same outputs as the reference
    implementation, within tolerance:
    - Logits: atol=1e-3, rtol=1e-3
    - Token count: exact match
    - Loss: tolerance=1e-3

    Args:
        model: Pre-loaded model (already on device, in train mode)
        data_iterator: Iterator yielding input tensors (batch_size, seq_len)
        optimizer: Pre-configured AdamW optimizer
        num_steps: Number of training steps to run
        device: Target device (cuda/cpu)

    Returns:
        InnerStepsResult with final_logits, total_tokens, final_loss
    """
    # Your optimized implementation here
    pass
```

## Reference Implementation

Here is the reference implementation your code must match:

```python
def inner_steps(
    model: torch.nn.Module,
    data_iterator: Iterator[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    device: torch.device,
) -> InnerStepsResult:
    """Reference implementation - your outputs must match this."""

    # Ensure deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    total_tokens = 0
    final_logits = None
    final_loss = 0.0

    for step in range(num_steps):
        # Get next batch
        batch = next(data_iterator)
        batch = batch.to(device, dtype=torch.long)

        # Prepare inputs and labels for causal LM
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]

        # Forward pass with bf16 autocast
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        # Backward and optimizer step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Track metrics
        total_tokens += batch.numel()
        final_logits = logits.detach().float()
        final_loss = loss.item()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
    )
```

## What You Can Optimize

You have complete freedom to optimize **how** training happens, as long as the outputs match:

### Forward Pass
- Use `torch.compile()` for kernel fusion
- Implement custom CUDA kernels
- Use flash attention
- Apply gradient checkpointing

### Data Loading
- Use pinned memory for faster H2D transfers
- Prefetch next batch while current is processing
- Use non-blocking transfers

### Optimizer
- Use fused optimizers (`torch.optim.AdamW(fused=True)`)
- Combine backward and optimizer step

### Memory
- Use gradient accumulation efficiently
- Implement activation checkpointing
- Use memory-efficient attention

## What You Cannot Change

- Model architecture
- Loss function (cross_entropy)
- Input/output format
- Label preparation (next-token prediction)

## Verification Errors

If your submission fails verification, you'll get detailed feedback:

### LogitsMismatchError
```
VERIFICATION FAILED: Output logits don't match expected values.

  Statistics:
    Max difference:  0.015234
    Mean difference: 0.003421
    Tolerance:       0.001000

  Possible causes:
    - Using different precision (fp16 vs bf16 vs fp32)
    - Non-deterministic operations
    - Incorrect forward pass implementation
```

### TokenCountMismatchError
```
VERIFICATION FAILED: Token count mismatch.

  Token counts:
    Expected: 819,200
    Actual:   802,816
    Difference: -16,384 (-2.00%)

  Possible causes:
    - Skipping batches in the training loop
    - Incorrect batch size calculation
    - Early termination
```

### LossMismatchError
```
VERIFICATION FAILED: Loss value mismatch.

  Loss values:
    Expected: 2.345678
    Actual:   2.456789
    Difference: 0.111111 (4.74%)

  Possible causes:
    - Modified loss function
    - Incorrect gradient accumulation
    - Precision issues
```

## Submitting Your Code

### Validate Locally First

```bash
uv run python -c "
from tournament.verification.verifier import SandboxVerifier
# Local validation code here
"
```

### Submit

```bash
uv run python -m neurons.miner train.py \
    --wallet.name <your_wallet_name> \
    --wallet.hotkey <your_hotkey_name>
```

### Check Status

After submission, you'll receive a submission ID:

```
Submission created: abc123-def456-...
Status: pending
```

Check status via the API:

```bash
curl http://localhost:8000/submissions/abc123-def456-...
```

## Example Optimized Submission

```python
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterator


@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float


def inner_steps(
    model: torch.nn.Module,
    data_iterator: Iterator[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    device: torch.device,
) -> InnerStepsResult:
    """Optimized inner_steps with torch.compile and prefetching."""

    # Ensure deterministic for verification
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Compile model for faster execution
    compiled_model = torch.compile(model, mode="reduce-overhead")

    total_tokens = 0
    final_logits = None
    final_loss = 0.0

    # Prefetch first batch
    next_batch = next(data_iterator)
    next_batch = next_batch.to(device, dtype=torch.long, non_blocking=True)

    for step in range(num_steps):
        # Current batch
        batch = next_batch

        # Prefetch next batch (non-blocking)
        if step < num_steps - 1:
            next_batch = next(data_iterator)
            next_batch = next_batch.to(device, dtype=torch.long, non_blocking=True)

        # Prepare inputs/labels
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]

        # Forward with bf16
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = compiled_model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        # Backward + step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)  # More efficient

        total_tokens += batch.numel()
        final_logits = logits.detach().float()
        final_loss = loss.item()

    torch.cuda.synchronize()

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
    )
```

## Forbidden Imports

For security, these imports are **not allowed**:

- `os`, `subprocess` (system access)
- `socket`, `http`, `urllib`, `requests` (network)
- `pickle`, `marshal` (serialization exploits)
- `ctypes`, `multiprocessing` (low-level access)

Your code will fail validation if it uses these.

## Scoring

Your submission is evaluated by multiple validators:

1. Code runs in isolated Docker sandbox
2. Outputs are verified against reference
3. TPS is measured externally (host-side timing)
4. Final score = average TPS across validators
5. Top scorer receives 100% of emissions

## Leaderboard

View current rankings:

```bash
curl http://localhost:8000/leaderboard
```
