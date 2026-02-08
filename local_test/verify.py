"""
Verify your train.py passes validator checks before submitting.

Usage:
    uv run local_test/verify.py

This script runs the SAME checks as the production validator:
1. final_logits must not be None
2. Logits must be 3D (batch, seq_len-1, vocab)
3. Sequence length must match expected
4. Token count matches expected
5. Loss must be positive, valid, and close to reference
6. 100% of parameters must be trainable (no frozen layers)
7. 80% of parameter elements must change during training
8. Gradient relative error |g - g_truth| / |g_truth| must be small

Fix any failures before submitting to avoid failed evaluations!
"""

import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM


@dataclass
class InnerStepsResult:
    """Result type for verification (mirrors train.py's InnerStepsResult)."""

    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float


@dataclass
class GradientInfo:
    """Gradient information for verification."""

    norm: float
    vectors: list  # List of per-layer gradient tensors on CPU
    layers_with_grad: int
    total_layers: int


class GradientCapturingOptimizer:
    """Same wrapper as the production validator uses.

    Captures gradients from the model on every optimizer.step() call.
    This is how the validator verifies your gradients match the reference.
    """

    _PROTECTED_ATTRS = frozenset(
        {
            "optimizer",
            "model",
            "captured_gradients",
            "step_count",
            "gradient_capture_time",
            "_initialized",
        }
    )

    def __init__(self, optimizer, model):
        object.__setattr__(self, "optimizer", optimizer)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "captured_gradients", None)
        object.__setattr__(self, "step_count", 0)
        object.__setattr__(self, "gradient_capture_time", 0.0)
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name, value):
        # Use CLASS reference (not self) to prevent bypass via
        # optimizer._PROTECTED_ATTRS = frozenset()
        if (
            getattr(self, "_initialized", False)
            and name in GradientCapturingOptimizer._PROTECTED_ATTRS
        ):
            raise AttributeError(f"Cannot modify protected attribute '{name}' on optimizer wrapper")
        object.__setattr__(self, name, value)

    def step(self, *args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        capture_start = time.perf_counter()
        object.__setattr__(self, "captured_gradients", capture_gradients(self.model))
        object.__setattr__(
            self,
            "gradient_capture_time",
            self.gradient_capture_time + (time.perf_counter() - capture_start),
        )
        object.__setattr__(self, "step_count", self.step_count + 1)
        return self.optimizer.step(*args, **kwargs)

    def zero_grad(self, set_to_none=False):
        return self.optimizer.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optimizer.param_groups = value

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self.optimizer.load_state_dict(state_dict)

    def add_param_group(self, param_group):
        return self.optimizer.add_param_group(param_group)

    @property
    def state(self):
        return self.optimizer.state

    def __getattr__(self, name):
        return getattr(self.optimizer, name)


def load_train_module(train_path: Path):
    """Load train.py as a module."""
    spec = importlib.util.spec_from_file_location("train", train_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture_gradients(model: torch.nn.Module) -> GradientInfo:
    """Capture gradient information from model after backward pass."""
    grad_vectors = []
    layers_with_grad = 0
    total_norm_sq = 0.0
    total_layers = 0

    for param in model.parameters():
        total_layers += 1
        if param.grad is not None:
            grad_flat = param.grad.detach().cpu().float().view(-1)
            total_norm_sq += grad_flat.pow(2).sum().item()
            if grad_flat.abs().sum().item() > 1e-10:
                layers_with_grad += 1
            grad_vectors.append(grad_flat)
        else:
            grad_vectors.append(None)

    return GradientInfo(
        norm=total_norm_sq**0.5,
        vectors=grad_vectors,
        layers_with_grad=layers_with_grad,
        total_layers=total_layers,
    )


def run_reference(model, data_iterator, optimizer, num_steps, device):
    """Run reference training and capture gradients on final step."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    total_tokens = 0
    final_logits = None
    final_loss = 0.0
    grad_info = None

    for step in range(num_steps):
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

        loss.backward()

        if step == num_steps - 1:
            grad_info = capture_gradients(model)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += batch.numel()
        final_logits = logits.detach().float()
        final_loss = float(loss.item())

    result = InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
    )
    return result, grad_info


def verify_all(
    reference,
    candidate,
    expected_tokens: int,
    expected_seq_len: int,
    batch_size: int,
    initial_state: dict,
    model,
    reference_grad: GradientInfo,
    candidate_grad: GradientInfo,
    max_loss_difference: float = 0.3,
    min_changed_ratio: float = 0.8,
    gradient_norm_ratio_max: float = 1.02,
) -> bool:
    """Run all validator checks - SAME as production validator."""
    print()
    print("=" * 70)
    print("VERIFICATION: Running validator checks (same as production)")
    print("=" * 70)

    all_passed = True
    check_num = 0

    # CHECK: final_logits not None
    check_num += 1
    print(f"\n[CHECK {check_num}] final_logits must not be None")
    if candidate.final_logits is None:
        print("  [FAILED] final_logits is None! Must return actual logits tensor.")
        all_passed = False
    else:
        print("  [PASSED]")

    # CHECK: Logits shape is 3D
    check_num += 1
    print(f"\n[CHECK {check_num}] Logits shape must be 3D (batch, seq, vocab)")
    if candidate.final_logits is not None:
        shape = candidate.final_logits.shape
        print(f"  Shape: {tuple(shape)}")
        if len(shape) != 3:
            print(f"  [FAILED] Expected 3D tensor, got {len(shape)}D")
            all_passed = False
        else:
            print("  [PASSED]")
    else:
        print("  [SKIPPED] (final_logits is None)")

    # CHECK: Sequence length
    check_num += 1
    print(f"\n[CHECK {check_num}] Sequence length (detects truncation)")
    if candidate.final_logits is not None and len(candidate.final_logits.shape) >= 2:
        logits_seq_len = candidate.final_logits.shape[1]
        print(f"  Expected: {expected_seq_len}, Got: {logits_seq_len}")
        if logits_seq_len != expected_seq_len:
            print("  [FAILED] Sequence length mismatch - possible truncation!")
            all_passed = False
        else:
            print("  [PASSED]")
    else:
        print("  [SKIPPED] (no valid logits)")

    # CHECK: Token count
    check_num += 1
    print(f"\n[CHECK {check_num}] Token count")
    print(f"  Expected: {expected_tokens}, Got: {candidate.total_tokens}")
    if candidate.total_tokens != expected_tokens:
        print("  [FAILED] Token count mismatch!")
        all_passed = False
    else:
        print("  [PASSED]")

    # CHECK: Loss validity
    check_num += 1
    print(f"\n[CHECK {check_num}] Loss validity and comparison")
    print(f"  Reference loss: {reference.final_loss:.6f}")
    print(f"  Candidate loss: {candidate.final_loss:.6f}")

    if candidate.final_loss != candidate.final_loss:
        print("  [FAILED] Loss is NaN!")
        all_passed = False
    elif candidate.final_loss <= 0:
        print(f"  [FAILED] Loss must be positive, got {candidate.final_loss:.4f}")
        all_passed = False
    elif candidate.final_loss > 100:
        print(f"  [FAILED] Loss unreasonable: {candidate.final_loss:.4f}")
        all_passed = False
    else:
        loss_diff = abs(candidate.final_loss - reference.final_loss)
        print(f"  Loss difference: {loss_diff:.4f} (max allowed: {max_loss_difference})")
        if loss_diff > max_loss_difference:
            print("  [FAILED] Loss difference too large!")
            all_passed = False
        else:
            print("  [PASSED]")

    # CHECK: Trainable parameters
    check_num += 1
    print(f"\n[CHECK {check_num}] Trainable parameters (need 100%)")
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    ratio = trainable_params / total_params if total_params > 0 else 0.0
    print(f"  Trainable: {trainable_params:,} / {total_params:,} ({ratio:.1%})")
    if ratio < 1.0:
        print("  [FAILED] All parameters must be trainable!")
        all_passed = False
    else:
        print("  [PASSED]")

    # CHECK: Parameters changed
    check_num += 1
    print(f"\n[CHECK {check_num}] Parameters changed (need {min_changed_ratio:.0%})")
    total_elements = 0
    changed_elements = 0
    for name, param in model.named_parameters():
        if name in initial_state:
            initial = initial_state[name].to(param.device)
            diffs = (param.data - initial).abs()
            total_elements += param.numel()
            changed_elements += (diffs > 1e-6).sum().item()
    changed_ratio = changed_elements / total_elements if total_elements > 0 else 0.0
    print(f"  Changed: {int(changed_elements):,} / {total_elements:,} ({changed_ratio:.1%})")
    if changed_ratio < min_changed_ratio:
        print(f"  [FAILED] Only {changed_ratio:.1%} changed, need {min_changed_ratio:.0%}")
        all_passed = False
    else:
        print("  [PASSED]")

    # CHECK: Gradient relative error
    check_num += 1
    relative_error_threshold = gradient_norm_ratio_max - 1.0
    print(f"\n[CHECK {check_num}] Gradient relative error |g - g_truth| / |g_truth|")
    print(f"  Max allowed: {relative_error_threshold:.4f} ({relative_error_threshold * 100:.1f}%)")

    # Gradient coverage
    if candidate_grad.total_layers > 0:
        coverage = candidate_grad.layers_with_grad / candidate_grad.total_layers
        print(
            f"  Gradient coverage: {coverage:.1%} ({candidate_grad.layers_with_grad}/{candidate_grad.total_layers})"
        )
        if coverage < 1.0:
            print("  [FAILED] Not all layers have gradients!")
            all_passed = False

    ref_vecs = reference_grad.vectors
    cand_vecs = candidate_grad.vectors
    if ref_vecs and cand_vecs and len(ref_vecs) == len(cand_vecs):
        diff_norm_sq = 0.0
        ref_norm_sq = 0.0
        for ref_layer, cand_layer in zip(ref_vecs, cand_vecs):
            if ref_layer is None or cand_layer is None:
                continue
            if ref_layer.shape != cand_layer.shape:
                print(
                    f"  [FAILED] Gradient shape mismatch: {ref_layer.shape} vs {cand_layer.shape}"
                )
                all_passed = False
                break
            diff = cand_layer - ref_layer
            diff_norm_sq += (diff * diff).sum().item()
            ref_norm_sq += (ref_layer * ref_layer).sum().item()

        ref_norm = ref_norm_sq**0.5
        diff_norm = diff_norm_sq**0.5
        relative_error = (
            diff_norm / ref_norm if ref_norm > 0 else (0.0 if diff_norm == 0 else float("inf"))
        )

        print(f"  |g - g_truth|: {diff_norm:.6f}")
        print(f"  |g_truth|: {ref_norm:.6f}")
        print(f"  Relative error: {relative_error:.6f}")

        if relative_error > relative_error_threshold:
            print(
                f"  [FAILED] Relative error {relative_error:.6f} > {relative_error_threshold:.6f}"
            )
            all_passed = False
        else:
            print("  [PASSED]")
    else:
        print("  [FAILED] Gradient vectors unavailable or shape mismatch")
        all_passed = False

    # Summary
    print()
    print("=" * 70)
    if all_passed:
        print("VERIFICATION: ALL CHECKS PASSED")
        print("Your submission should pass validator evaluation!")
    else:
        print("VERIFICATION: SOME CHECKS FAILED")
        print("Fix the issues above before submitting.")
    print("=" * 70)

    return all_passed


def main():
    print("=" * 70)
    print("VERIFYING train.py - Same checks as production validator")
    print("=" * 70)
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

    verification = hparams.get("verification", {})
    max_loss_difference = verification.get("max_loss_difference", 0.3)
    min_changed_ratio = verification.get("min_params_changed_ratio", 0.8)
    gradient_norm_ratio_max = verification.get("gradient_norm_ratio_max", 1.02)

    expected_tokens = batch_size * seq_len * num_steps
    expected_seq_len = seq_len - 1  # Causal LM: input_ids = batch[:, :-1]

    print("Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Steps per eval: {num_steps}")
    print(f"  Expected tokens: {expected_tokens:,}")
    print(f"  Expected logits seq_len: {expected_seq_len}")
    print(f"  Max loss difference: {max_loss_difference}")
    print(f"  Min params changed: {min_changed_ratio:.0%}")
    relative_err = gradient_norm_ratio_max - 1.0
    print(f"  Max gradient relative error: {relative_err:.4f} ({relative_err * 100:.1f}%)")
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

    # Load miner's module
    print("Loading train.py...")
    train_module = load_train_module(train_path)

    if not hasattr(train_module, "inner_steps"):
        print("ERROR: train.py must have an 'inner_steps' function!")
        sys.exit(1)
    print("  Found inner_steps function")
    print()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load model with random init (same as validator)
    print("Loading model with RANDOM INITIALIZATION (same as validator)...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = model.to(device)
    model.gradient_checkpointing_enable()
    model.train()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Load data
    print("Loading data...")
    data = torch.load(data_path, weights_only=True)
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

    # Save initial state
    initial_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    # === Run reference ===
    print("Running reference baseline...")
    optimizer_ref = torch.optim.AdamW(model.parameters(), lr=1e-4)
    reference, reference_grad = run_reference(
        model, create_iterator(), optimizer_ref, num_steps, device
    )
    print(f"  Reference loss: {reference.final_loss:.6f}")
    print(f"  Reference gradient norm: {reference_grad.norm:.4f}")
    print()

    # === Reset model ===
    model.load_state_dict({k: v.to(device) for k, v in initial_state.items()})
    model.train()

    # === Warmup (same as validator) ===
    print("Running warmup (2 steps, not verified)...")
    warmup_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    try:
        warmup_result = train_module.inner_steps(
            model=model,
            data_iterator=create_iterator(),
            optimizer=warmup_optimizer,
            num_steps=2,
            device=device,
        )
        if warmup_result is None:
            print("  [FAILED] inner_steps returned None during warmup!")
            sys.exit(1)
        if warmup_result.final_logits is None:
            print("  [WARNING] final_logits is None during warmup - will FAIL during timed eval!")
        print("  Warmup passed")
    except Exception as e:
        print(f"  [FAILED] Warmup crashed: {e}")
        sys.exit(1)
    print()

    # === Reset model again ===
    model.load_state_dict({k: v.to(device) for k, v in initial_state.items()})
    model.train()

    # === Run miner's code with GradientCapturingOptimizer (same as validator) ===
    print("Running your inner_steps with GradientCapturingOptimizer...")
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    capturing_optimizer = GradientCapturingOptimizer(base_optimizer, model)

    try:
        miner_result = train_module.inner_steps(
            model=model,
            data_iterator=create_iterator(),
            optimizer=capturing_optimizer,
            num_steps=num_steps,
            device=device,
        )
    except AttributeError as e:
        if "protected attribute" in str(e):
            print(f"  [FAILED] Your code tried to modify optimizer internals: {e}")
            sys.exit(1)
        raise
    except Exception as e:
        print(f"  [FAILED] inner_steps crashed: {e}")
        sys.exit(1)

    candidate_grad = capturing_optimizer.captured_gradients
    if candidate_grad is None:
        print("  [FAILED] No gradients captured! Make sure you call optimizer.step()")
        sys.exit(1)

    # Build candidate result
    ok = (
        hasattr(miner_result, "final_logits")
        and hasattr(miner_result, "total_tokens")
        and hasattr(miner_result, "final_loss")
    )
    if not ok:
        print(
            "  [FAILED] inner_steps must return InnerStepsResult with final_logits, total_tokens, final_loss"
        )
        sys.exit(1)

    candidate = InnerStepsResult(
        final_logits=miner_result.final_logits,
        total_tokens=miner_result.total_tokens,
        final_loss=miner_result.final_loss,
    )

    print(f"  Candidate loss: {candidate.final_loss:.6f}")
    print(f"  Candidate gradient norm: {candidate_grad.norm:.4f}")
    print(f"  Optimizer step count: {capturing_optimizer.step_count}")

    # === Verify ===
    passed = verify_all(
        reference=reference,
        candidate=candidate,
        expected_tokens=expected_tokens,
        expected_seq_len=expected_seq_len,
        batch_size=batch_size,
        initial_state=initial_state,
        model=model,
        reference_grad=reference_grad,
        candidate_grad=candidate_grad,
        max_loss_difference=max_loss_difference,
        min_changed_ratio=min_changed_ratio,
        gradient_norm_ratio_max=gradient_norm_ratio_max,
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
