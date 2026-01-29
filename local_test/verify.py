"""
Verify your train.py passes validator checks before submitting.

Usage:
    uv run local_test/verify.py

This script:
1. Loads your train.py implementation
2. Runs a reference baseline
3. Runs your inner_steps function
4. Compares outputs using the same checks as the validator

Fix any failures before submitting to avoid failed evaluations!
"""

import copy
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


@dataclass
class InnerStepsResult:
    """Result type for verification (mirrors train.py's InnerStepsResult)."""

    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float


def load_train_module(train_path: Path):
    """Load train.py as a module."""
    spec = importlib.util.spec_from_file_location("train", train_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_reference(model, data_iterator, optimizer, num_steps, device):
    """Run reference training (same as validator does)."""
    # Set deterministic mode like validator
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    total_tokens = 0
    final_logits = None
    final_loss = 0.0

    for _ in range(num_steps):
        batch = next(data_iterator)
        batch = batch.to(device, dtype=torch.long)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            input_ids = batch[:, :-1]
            labels = batch[:, 1:]
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )

        # Training step (same as validator reference)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += batch.numel()
        final_logits = logits.detach().float()
        final_loss = float(loss.item())

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
    )


def verify_outputs(
    reference,
    candidate,
    expected_tokens: int,
    output_tolerance: float = 0.05,
    loss_ratio_min: float = 0.8,
    loss_ratio_max: float = 1.2,
) -> bool:
    """Verify candidate outputs match reference within tolerance.

    These are the same checks the validator runs during evaluation.
    """
    print()
    print("=" * 60)
    print("VERIFICATION: Running validator checks")
    print("=" * 60)

    all_passed = True

    # 1. Token count check
    print(f"\n[CHECK 1/4] Token count: expected={expected_tokens}, got={candidate.total_tokens}")
    if candidate.total_tokens != expected_tokens:
        print("  [FAILED] Token count mismatch!")
        all_passed = False
    else:
        print("  [PASSED] Token count matches")

    # 2. Loss validity check
    print(f"\n[CHECK 2/4] Loss validity: candidate_loss={candidate.final_loss:.6f}")
    if candidate.final_loss != candidate.final_loss:  # NaN check
        print("  [FAILED] Loss is NaN!")
        all_passed = False
    elif abs(candidate.final_loss) > 100:
        print(f"  [FAILED] Loss unreasonable: {candidate.final_loss:.4f} (expected 1-10)")
        all_passed = False
    else:
        print("  [PASSED] Loss is valid (not NaN/Inf, in reasonable range)")

    # 3. Logits validity check
    print("\n[CHECK 3/4] Logits validity")
    cand_logits = candidate.final_logits
    if cand_logits is None:
        print("  [FAILED] No logits returned!")
        all_passed = False
    else:
        print(f"  Candidate logits shape: {cand_logits.shape}")
        if len(cand_logits.shape) != 3:
            print(
                f"  [FAILED] Logits should be 3D (batch, seq, vocab), got {len(cand_logits.shape)}D"
            )
            all_passed = False
        elif torch.isnan(cand_logits).any():
            nan_count = torch.isnan(cand_logits).sum().item()
            print(f"  [FAILED] Logits contain {nan_count} NaN values!")
            all_passed = False
        elif torch.isinf(cand_logits).any():
            inf_count = torch.isinf(cand_logits).sum().item()
            print(f"  [FAILED] Logits contain {inf_count} Inf values!")
            all_passed = False
        else:
            print("  [PASSED] Logits are valid (3D shape, no NaN/Inf)")

    # 4. Reference comparison
    print("\n[CHECK 4/4] Reference comparison")
    ref_logits = reference.final_logits

    # 4a. Loss comparison
    if reference.final_loss > 0:
        loss_ratio = candidate.final_loss / reference.final_loss
        print(f"  Loss: candidate={candidate.final_loss:.6f}, reference={reference.final_loss:.6f}")
        print(f"  Loss ratio: {loss_ratio:.4f} (allowed: {loss_ratio_min}-{loss_ratio_max})")

        if loss_ratio < loss_ratio_min or loss_ratio > loss_ratio_max:
            print("  [FAILED] Loss ratio outside allowed range!")
            all_passed = False
        else:
            print("  [PASSED] Loss matches reference within tolerance")

    # 4b. Logits comparison
    if cand_logits is not None and ref_logits is not None:
        if cand_logits.shape != ref_logits.shape:
            print(f"  [FAILED] Logits shape mismatch: {cand_logits.shape} vs {ref_logits.shape}")
            all_passed = False
        else:
            ref_scale = ref_logits.abs().mean()
            abs_diff = (cand_logits - ref_logits).abs()
            relative_diff = (abs_diff.mean() / ref_scale).item() if ref_scale > 0 else 0

            print(
                f"  Logits relative difference: {relative_diff:.6f} (tolerance: {output_tolerance})"
            )

            if relative_diff > output_tolerance:
                print("  [FAILED] Logits difference exceeds tolerance!")
                all_passed = False
            else:
                print(
                    f"  [PASSED] Logits match reference (diff={relative_diff:.4f} <= {output_tolerance})"
                )

    # Summary
    print()
    print("=" * 60)
    if all_passed:
        print("VERIFICATION: ALL CHECKS PASSED")
        print("Your submission should pass validator evaluation!")
    else:
        print("VERIFICATION: SOME CHECKS FAILED")
        print("Fix the issues above before submitting.")
    print("=" * 60)

    return all_passed


def main():
    print("=" * 60)
    print("VERIFYING train.py - Validator Checks")
    print("=" * 60)
    print()

    # Load configuration
    project_root = Path(__file__).parent.parent
    hparams_path = project_root / "hparams" / "hparams.json"
    hparams = {}
    if hparams_path.exists():
        with open(hparams_path) as f:
            hparams = json.load(f)

    batch_size = hparams.get("benchmark_batch_size", 4)
    seq_len = hparams.get("benchmark_sequence_length", 1024)
    num_steps = hparams.get("eval_steps", 5)
    output_tolerance = hparams.get("verification", {}).get("output_vector_tolerance", 0.05)
    loss_ratio_min = hparams.get("verification", {}).get("loss_ratio_min", 0.8)
    loss_ratio_max = hparams.get("verification", {}).get("loss_ratio_max", 1.2)

    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Steps per eval: {num_steps}")
    print(f"Output tolerance: {output_tolerance}")
    print(f"Loss ratio range: {loss_ratio_min}-{loss_ratio_max}")
    print()

    # Check paths
    model_path = project_root / "benchmark" / "model"
    data_path = project_root / "benchmark" / "data" / "train.pt"
    train_path = project_root / "local_test" / "train.py"

    if not model_path.exists() or not data_path.exists():
        print("Setup required! Run: uv run local_test/setup_benchmark.py")
        sys.exit(1)

    if not train_path.exists():
        print(f"train.py not found at {train_path}")
        sys.exit(1)

    # Load miner's train.py module
    print("Loading train.py...")
    train_module = load_train_module(train_path)

    if not hasattr(train_module, "inner_steps"):
        print("ERROR: train.py must have an 'inner_steps' function!")
        sys.exit(1)

    if not hasattr(train_module, "InnerStepsResult"):
        print("ERROR: train.py must have an 'InnerStepsResult' dataclass!")
        sys.exit(1)

    print("  Found inner_steps function")
    print("  Found InnerStepsResult dataclass")
    print()

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
    model.gradient_checkpointing_enable()
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

    # Warmup
    print("Warmup...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    _ = train_module.inner_steps(model, create_iterator(), optimizer, num_steps=2, device=device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print()

    # Save initial model state
    initial_state = copy.deepcopy(model.state_dict())

    # Expected tokens = batch_size * seq_len * num_steps
    expected_tokens = batch_size * seq_len * num_steps

    # Run reference (with training, same as validator)
    print("Running reference baseline...")
    model.train()
    optimizer_ref = torch.optim.AdamW(model.parameters(), lr=1e-4)
    reference = run_reference(model, create_iterator(), optimizer_ref, num_steps, device)
    print(f"  Reference loss: {reference.final_loss:.6f}")
    print(f"  Reference tokens: {reference.total_tokens:,}")

    # Reset model state (same initial weights for candidate)
    model.load_state_dict(initial_state)

    # Run candidate (miner's inner_steps)
    print()
    print("Running your inner_steps...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    candidate = train_module.inner_steps(model, create_iterator(), optimizer, num_steps, device)
    print(f"  Candidate loss: {candidate.final_loss:.6f}")
    print(f"  Candidate tokens: {candidate.total_tokens:,}")

    # Verify
    passed = verify_outputs(
        reference=reference,
        candidate=candidate,
        expected_tokens=expected_tokens,
        output_tolerance=output_tolerance,
        loss_ratio_min=loss_ratio_min,
        loss_ratio_max=loss_ratio_max,
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
