"""
Verify your train.py passes validator checks before submitting.

Usage:
    uv run local_test/verify.py

This script runs the SAME checks as the production validator:
0. Security scan (AST) — forbidden imports, patterns, backend modifications
1. final_logits must not be None
2. Logits must be 3D (batch, seq_len-1, vocab)
3. Sequence length must match expected
4. Token count matches expected
5. Loss must be positive, valid, and close to reference
6. 100% of parameters must be trainable (no frozen layers)
7. 80% of parameter elements must change during training
8. Gradient relative error |g - g_truth| / |g_truth| must be small
9. Final weight relative error |w - w_ref| / |w_ref| must be small

Fix any failures before submitting to avoid failed evaluations!
"""

import ast
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

# =========================================================================
# Security checks — same as production env.py _validate_code_structure
# =========================================================================

_FORBIDDEN_STRINGS = [
    "__setattr__",
    "__delattr__",
    "__class__",
    "perf_counter",
    "get_objects",
    "get_referrers",
    "get_referents",
    "enable_flash_sdp",
    "enable_mem_efficient_sdp",
    "enable_math_sdp",
    "set_float32_matmul_precision",
    "captured_gradients",
    "_opt_impl",
    "_grad_snapshot_gpu",
    "step_count",
    "GradientCapturingOptimizer",
    # Dynamic code execution / import bypasses
    "__import__",
    "importlib",
    "import_module",
]


def _is_main_guard(node: ast.AST) -> bool:
    """Check if an AST node is an `if __name__ == "__main__":` block."""
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "__main__"
    )


def _scan_for_dangerous_patterns(tree: ast.AST) -> list[str]:
    """AST scan to find all forbidden code patterns. Returns list of violations."""
    violations = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ("__setattr__", "__delattr__"):
            if isinstance(node.value, ast.Name) and node.value.id == "object":
                line = getattr(node, "lineno", "?")
                violations.append(f"Line {line}: object.{node.attr} is forbidden")

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "__class__":
                    line = getattr(node, "lineno", "?")
                    violations.append(f"Line {line}: __class__ assignment is forbidden")

        if isinstance(node, ast.Attribute) and node.attr == "__class__":
            if not isinstance(getattr(node, "_parent", None), ast.AnnAssign):
                line = getattr(node, "lineno", "?")
                violations.append(f"Line {line}: __class__ access is forbidden")

        if isinstance(node, ast.Attribute) and node.attr == "perf_counter":
            if isinstance(node.value, ast.Name) and node.value.id == "time":
                line = getattr(node, "lineno", "?")
                violations.append(f"Line {line}: time.perf_counter is forbidden")

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "synchronize":
                    line = getattr(node, "lineno", "?")
                    violations.append(f"Line {line}: overriding synchronize is forbidden")

        if isinstance(node, ast.Attribute) and node.attr == "__slots__":
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                line = getattr(node, "lineno", "?")
                violations.append(f"Line {line}: __slots__ modification is forbidden")

        # Block: gc introspection (gc.get_objects, gc.get_referrers, gc.get_referents)
        if isinstance(node, ast.Attribute) and node.attr in (
            "get_objects",
            "get_referrers",
            "get_referents",
        ):
            if isinstance(node.value, ast.Name) and node.value.id == "gc":
                line = getattr(node, "lineno", "?")
                violations.append(f"Line {line}: gc.{node.attr}() is forbidden")

        # Block: torch backend setting modifications (deterministic, benchmark)
        if isinstance(node, ast.Attribute) and node.attr in (
            "deterministic",
            "benchmark",
        ):
            if isinstance(node.ctx, ast.Store):
                line = getattr(node, "lineno", "?")
                violations.append(f"Line {line}: setting torch.backends.{node.attr} is forbidden")

        # Block: torch SDP toggle calls and float32_matmul_precision
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in (
                "enable_flash_sdp",
                "enable_mem_efficient_sdp",
                "enable_math_sdp",
                "set_float32_matmul_precision",
            ):
                line = getattr(node, "lineno", "?")
                violations.append(f"Line {line}: {func.attr}() is forbidden")

        # Block: dynamic code execution (exec, eval, compile, __import__)
        # These bypass AST-level checks by running arbitrary code at runtime.
        # torch.compile is safe (ast.Attribute, not ast.Name).
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ("exec", "eval", "compile", "__import__"):
                line = getattr(node, "lineno", "?")
                violations.append(f"Line {line}: {node.func.id}() is forbidden")

        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("ctypes", "_ctypes", "gc", "subprocess") or alias.name.startswith(
                    "importlib"
                ):
                    line = getattr(node, "lineno", "?")
                    violations.append(f"Line {line}: import {alias.name} is forbidden")

        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module in ("ctypes", "_ctypes", "gc", "subprocess") or node.module.startswith(
                "importlib"
            ):
                line = getattr(node, "lineno", "?")
                violations.append(f"Line {line}: from {node.module} import is forbidden")

        # Block: accessing .optimizer attribute (wrapper bypass attempt)
        if isinstance(node, ast.Attribute) and node.attr == "optimizer":
            # Allow 'self.optimizer' inside class definitions, but block
            # attempts to unwrap the GradientCapturingOptimizer via optimizer.optimizer
            if not (isinstance(node.value, ast.Name) and node.value.id == "self"):
                line = getattr(node, "lineno", "?")
                violations.append(f"Line {line}: accessing .optimizer attribute is forbidden")

    return violations


def validate_code_structure(code: str) -> list[str]:
    """Validate that train.py passes all security checks (same as production).

    Returns:
        List of violation messages (empty = all passed).
    """
    violations = []

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"Syntax error at line {exc.lineno}: {exc.msg}"]

    # Strip `if __name__ == "__main__":` blocks before scanning
    scan_tree = ast.Module(
        body=[node for node in tree.body if not _is_main_guard(node)],
        type_ignores=tree.type_ignores,
    )

    violations.extend(_scan_for_dangerous_patterns(scan_tree))

    # Scan string literals for forbidden patterns
    for node in ast.walk(scan_tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for pattern in _FORBIDDEN_STRINGS:
                if pattern in node.value:
                    line = getattr(node, "lineno", "?")
                    violations.append(f"Line {line}: forbidden string pattern '{pattern}' detected")

    inner_steps_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "inner_steps":
            inner_steps_found = True
            args = node.args
            if len(args.args) < 5:
                violations.append(f"inner_steps has {len(args.args)} args, expected at least 5")
            break

    if not inner_steps_found:
        violations.append("Missing required function: inner_steps")

    return violations


# =========================================================================
# Data classes
# =========================================================================


@dataclass
class InnerStepsResult:
    """Result type for verification (mirrors train.py's InnerStepsResult)."""

    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float


@dataclass
class GradientInfo:
    """Gradient information for verification. Matches production env.py."""

    grad_norm: float
    grad_vector: list  # List of per-layer gradient tensors on CPU
    layers_with_grad: int
    total_layers: int


class GradientCapturingOptimizer:
    """Same wrapper as the production validator uses.

    On the final step, clones gradient tensors on GPU (fast) before
    optimizer.step() clears them. Call finalize_gradients() after
    timing to do the slow CPU transfer.

    Security: __getattribute__ blocks access to private internals (_opt_impl,
    _grad_snapshot_gpu) so miners cannot introspect the wrapper via getattr()
    or dir(). Internal methods use object.__getattribute__() to bypass the guard.
    """

    __slots__ = (
        "_opt_impl",
        "model",
        "captured_gradients",
        "step_count",
        "num_steps",
        "_grad_snapshot_gpu",
        "_initialized",
    )

    # Attributes accessible to miner code. Everything else starting with "_"
    # (except dunder methods) is blocked by __getattribute__.
    _PUBLIC_ATTRS = frozenset(
        {
            "step",
            "zero_grad",
            "param_groups",
            "state",
            "state_dict",
            "load_state_dict",
            "add_param_group",
            "finalize_gradients",
            "captured_gradients",
            "num_steps",
            "model",
            "step_count",
        }
    )

    def __init__(self, optimizer, model, num_steps):
        object.__setattr__(self, "_opt_impl", optimizer)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "captured_gradients", None)
        object.__setattr__(self, "step_count", 0)
        object.__setattr__(self, "num_steps", num_steps)
        object.__setattr__(self, "_grad_snapshot_gpu", None)
        object.__setattr__(self, "_initialized", True)

    def __getattribute__(self, name):
        """Guard access to private internals.

        Allows: public attrs, dunder methods (__class__, __repr__, etc.),
        and non-underscore names (forwarded via __getattr__).
        Blocks: _opt_impl, _grad_snapshot_gpu, _initialized, and any
        other single-underscore private attr.
        """
        # Always allow dunder attrs (Python internals need them)
        if name.startswith("__") and name.endswith("__"):
            return object.__getattribute__(self, name)
        # Allow explicitly public attrs
        if name in GradientCapturingOptimizer._PUBLIC_ATTRS:
            return object.__getattribute__(self, name)
        # Block single-underscore private attrs (_opt_impl, _grad_snapshot_gpu, etc.)
        if name.startswith("_"):
            raise AttributeError(f"Access to '{name}' is not allowed on optimizer wrapper")
        # Non-underscore names not in _PUBLIC_ATTRS: forward via __getattr__
        return object.__getattribute__(self, name)

    def __dir__(self):
        """Only expose public API to dir(), hiding internal slots."""
        return sorted(GradientCapturingOptimizer._PUBLIC_ATTRS)

    def __setattr__(self, name, value):
        try:
            initialized = object.__getattribute__(self, "_initialized")
        except AttributeError:
            initialized = False
        if initialized:
            raise AttributeError(
                f"Cannot modify attribute '{name}' on optimizer wrapper (read-only after init)"
            )
        object.__setattr__(self, name, value)

    def step(self, *args, **kwargs):
        current_step = object.__getattribute__(self, "step_count")
        object.__setattr__(self, "step_count", current_step + 1)

        # On final step, snapshot gradients on GPU (fast clone, stays on device)
        if current_step == object.__getattribute__(self, "num_steps") - 1:
            snapshot = []
            model = object.__getattribute__(self, "model")
            for param in model.parameters():
                if param.grad is not None:
                    snapshot.append(param.grad.detach().clone())
                else:
                    snapshot.append(None)
            object.__setattr__(self, "_grad_snapshot_gpu", snapshot)

        opt = object.__getattribute__(self, "_opt_impl")
        return opt.step(*args, **kwargs)

    def finalize_gradients(self) -> None:
        """Convert GPU gradient snapshot to GradientInfo (CPU)."""
        snapshot = object.__getattribute__(self, "_grad_snapshot_gpu")
        if snapshot is None:
            return

        grad_vectors_cpu = []
        total_norm_sq = 0.0
        layers_with_grad = 0
        layers_without_grad = 0

        for grad_gpu in snapshot:
            if grad_gpu is not None:
                grad_flat = grad_gpu.cpu().float().view(-1)
                total_norm_sq += grad_flat.pow(2).sum().item()
                if grad_flat.abs().sum().item() > 1e-10:
                    layers_with_grad += 1
                    grad_vectors_cpu.append(grad_flat)
                else:
                    layers_without_grad += 1
                    grad_vectors_cpu.append(grad_flat)
            else:
                layers_without_grad += 1
                grad_vectors_cpu.append(None)

        total_layers = layers_with_grad + layers_without_grad
        object.__setattr__(
            self,
            "captured_gradients",
            GradientInfo(
                grad_norm=total_norm_sq**0.5,
                grad_vector=grad_vectors_cpu,
                layers_with_grad=layers_with_grad,
                total_layers=total_layers,
            ),
        )
        object.__setattr__(self, "_grad_snapshot_gpu", None)

    def zero_grad(self, set_to_none=False):
        opt = object.__getattribute__(self, "_opt_impl")
        return opt.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        opt = object.__getattribute__(self, "_opt_impl")
        return opt.param_groups

    @param_groups.setter
    def param_groups(self, value):
        opt = object.__getattribute__(self, "_opt_impl")
        opt.param_groups = value

    def state_dict(self):
        opt = object.__getattribute__(self, "_opt_impl")
        return opt.state_dict()

    def load_state_dict(self, state_dict):
        opt = object.__getattribute__(self, "_opt_impl")
        return opt.load_state_dict(state_dict)

    def add_param_group(self, param_group):
        opt = object.__getattribute__(self, "_opt_impl")
        return opt.add_param_group(param_group)

    @property
    def state(self):
        opt = object.__getattribute__(self, "_opt_impl")
        return opt.state

    def __getattr__(self, name):
        """Forward non-private attribute access to underlying optimizer."""
        # Block private attrs that fell through from __getattribute__ raising
        if name.startswith("_") and not (name.startswith("__") and name.endswith("__")):
            raise AttributeError(f"Access to '{name}' is not allowed on optimizer wrapper")
        opt = object.__getattribute__(self, "_opt_impl")
        return getattr(opt, name)


def load_train_module(train_path: Path):
    """Load train.py as a module."""
    spec = importlib.util.spec_from_file_location("train", train_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture_gradients(model: torch.nn.Module) -> GradientInfo:
    """Capture gradient information from model after backward pass.

    Matches production env.py's _capture_gradients().
    """
    grad_vectors_cpu = []
    total_norm_sq = 0.0
    layers_with_grad = 0
    layers_without_grad = 0

    for param in model.parameters():
        if param.grad is not None:
            grad_flat = param.grad.detach().cpu().float().view(-1)
            total_norm_sq += grad_flat.pow(2).sum().item()
            if grad_flat.abs().sum().item() > 1e-10:
                layers_with_grad += 1
                grad_vectors_cpu.append(grad_flat)
            else:
                layers_without_grad += 1
                grad_vectors_cpu.append(grad_flat)
        else:
            layers_without_grad += 1
            grad_vectors_cpu.append(None)

    return GradientInfo(
        grad_norm=total_norm_sq**0.5,
        grad_vector=grad_vectors_cpu,
        layers_with_grad=layers_with_grad,
        total_layers=layers_with_grad + layers_without_grad,
    )


def run_reference(model, data_iterator, optimizer, num_steps, device):
    """Run reference training and capture gradients on final step."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    # set_float32_matmul_precision("highest") controls matmul TF32 precision.
    # Do NOT set cuda.matmul.allow_tf32 separately -- it would be overridden.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    total_tokens = 0
    final_logits = None
    final_loss = 0.0
    grad_info = None

    for step in range(num_steps):
        batch = next(data_iterator)
        batch = batch.to(device, dtype=torch.long)

        # No autocast - model is already in bfloat16.
        # Using autocast here would change loss computation precision
        # and create gradient mismatches with miner code that doesn't use autocast.
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


def main():
    print("=" * 70)
    print("VERIFYING train.py - Same checks as production validator")
    print("=" * 70)
    print()

    # Track all results: list of (check_name, passed, detail)
    results: list[tuple[str, bool, str]] = []

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
    weight_relative_error_max = verification.get("weight_relative_error_max", 0.04)

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
    gradient_err = gradient_norm_ratio_max - 1.0
    print(f"  Max gradient relative error: {gradient_err:.4f} ({gradient_err * 100:.1f}%)")
    print(
        f"  Max weight relative error: {weight_relative_error_max:.4f} ({weight_relative_error_max * 100:.1f}%)"
    )
    print()

    # Check paths (these are fatal — can't continue without files)
    model_path = project_root / "benchmark" / "model"
    data_path = project_root / "benchmark" / "data" / "train.pt"
    train_path = project_root / "local_test" / "train.py"

    if not model_path.exists() or not data_path.exists():
        print("Setup required! Run: uv run local_test/setup_benchmark.py")
        sys.exit(1)

    if not train_path.exists():
        print(f"train.py not found at {train_path}")
        sys.exit(1)

    # =====================================================================
    # CHECK: Security scan (same as production validator)
    # =====================================================================
    print("Security scan (same as production validator)...")
    code = train_path.read_text()
    security_violations = validate_code_structure(code)
    if security_violations:
        for v in security_violations:
            print(f"  [FAILED] {v}")
            results.append(("Security: " + v, False, v))
        print()
        print("  WARNING: Production validator would REJECT this code before execution.")
        print("  Continuing with remaining checks for debugging purposes...")
    else:
        print("  [PASSED] No forbidden patterns detected")
        results.append(("Security scan", True, "No forbidden patterns"))
    print()

    # Load miner's module
    print("Loading train.py...")
    train_module = load_train_module(train_path)

    if not hasattr(train_module, "inner_steps"):
        print("ERROR: train.py must have an 'inner_steps' function!")
        results.append(("inner_steps function", False, "Function not found"))
        _print_summary(results)
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
        config, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa"
    )
    model = model.to(device)
    model.gradient_checkpointing_enable()
    model.train()
    # Ensure all parameters are trainable (same as production validator)
    for param in model.parameters():
        param.requires_grad = True
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
    use_fused = torch.cuda.is_available()
    optimizer_ref = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=0.1, betas=(0.9, 0.95), fused=use_fused
    )
    reference, reference_grad = run_reference(
        model, create_iterator(), optimizer_ref, num_steps, device
    )
    print(f"  Reference loss: {reference.final_loss:.6f}")
    print(f"  Reference gradient norm: {reference_grad.grad_norm:.4f}")

    # Capture reference final weights for weight verification (same as production validator)
    reference_final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print("  Captured reference final model state for weight verification")
    print()

    # === Reset model ===
    model.load_state_dict({k: v.to(device) for k, v in initial_state.items()})
    model.train()

    # === Warmup (same as validator) ===
    print("Running warmup (2 steps, not verified)...")
    warmup_optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=0.1, betas=(0.9, 0.95), fused=use_fused
    )
    warmup_ok = True
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
            results.append(("Warmup", False, "inner_steps returned None"))
            warmup_ok = False
        elif warmup_result.final_logits is None:
            print("  [WARNING] final_logits is None during warmup")
            print("  Warmup passed (with warning)")
        else:
            print("  Warmup passed")
    except Exception as e:
        print(f"  [FAILED] Warmup crashed: {e}")
        results.append(("Warmup", False, str(e)))
        warmup_ok = False
    print()

    if not warmup_ok:
        print("Cannot continue — warmup failed (code doesn't run).")
        _print_summary(results)
        sys.exit(1)

    # === Reset model again ===
    model.load_state_dict({k: v.to(device) for k, v in initial_state.items()})
    model.train()

    # === Enforce backend state (same as production validator) ===
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    # === Run miner's code with GradientCapturingOptimizer (same as validator) ===
    print("Running your inner_steps with GradientCapturingOptimizer...")
    base_optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=0.1, betas=(0.9, 0.95), fused=use_fused
    )
    capturing_optimizer = GradientCapturingOptimizer(base_optimizer, model, num_steps=num_steps)

    miner_result = None
    exec_error = None
    try:
        miner_result = train_module.inner_steps(
            model=model,
            data_iterator=create_iterator(),
            optimizer=capturing_optimizer,
            num_steps=num_steps,
            device=device,
        )
    except AttributeError as e:
        if "read-only" in str(e) or "protected" in str(e):
            exec_error = f"Tried to modify optimizer internals: {e}"
        else:
            exec_error = str(e)
    except Exception as e:
        exec_error = str(e)

    if exec_error:
        print(f"  [FAILED] inner_steps crashed: {exec_error}")
        results.append(("Execution", False, exec_error))
        print()
        print("Cannot continue — inner_steps crashed during timed eval.")
        _print_summary(results)
        sys.exit(1)

    # Finalize gradients (move GPU snapshot to CPU — same as production validator)
    capturing_optimizer.finalize_gradients()

    # Verify backend settings were not tampered with during eval (same as production validator)
    backend_violations = []
    if not torch.backends.cudnn.deterministic:
        backend_violations.append("cudnn.deterministic changed to False")
    if torch.backends.cudnn.benchmark:
        backend_violations.append("cudnn.benchmark changed to True")
    if backend_violations:
        print("  [WARNING] Backend settings changed during eval:")
        for v in backend_violations:
            print(f"    - {v}")
            results.append((f"Backend: {v}", False, v))
        print("  Production validator would REJECT this submission.")
    print()

    # Verify optimizer wrapper integrity (same as production validator)
    if type(capturing_optimizer) is not GradientCapturingOptimizer:
        print("  [FAILED] Optimizer wrapper type was replaced!")
        results.append(("Optimizer integrity", False, "Wrapper type changed"))

    candidate_grad = capturing_optimizer.captured_gradients

    # Check return type
    ok = miner_result is not None and all(
        hasattr(miner_result, attr) for attr in ("final_logits", "total_tokens", "final_loss")
    )
    if not ok:
        print(
            "  [FAILED] inner_steps must return object with final_logits, total_tokens, final_loss"
        )
        results.append(("Return type", False, "Invalid return type"))
        _print_summary(results)
        sys.exit(1)

    candidate = InnerStepsResult(
        final_logits=miner_result.final_logits,
        total_tokens=miner_result.total_tokens,
        final_loss=miner_result.final_loss,
    )

    if candidate_grad is not None:
        print(f"  Candidate loss: {candidate.final_loss:.6f}")
        print(f"  Candidate gradient norm: {candidate_grad.grad_norm:.4f}")
    else:
        print(f"  Candidate loss: {candidate.final_loss:.6f}")
        print("  Candidate gradients: NOT CAPTURED")
    print(f"  Optimizer step count: {capturing_optimizer.step_count}")

    # =====================================================================
    # Run all verification checks (never exit early)
    # =====================================================================
    print()
    print("=" * 70)
    print("VERIFICATION: Running validator checks (same as production)")
    print("=" * 70)

    # CHECK: final_logits not None
    check = "final_logits not None"
    print(f"\n[CHECK] {check}")
    if candidate.final_logits is None:
        print("  [FAILED] final_logits is None! Must return actual logits tensor.")
        results.append((check, False, "final_logits is None"))
    else:
        print("  [PASSED]")
        results.append((check, True, ""))

    # CHECK: Logits shape is 3D
    check = "Logits shape 3D"
    print(f"\n[CHECK] {check}")
    if candidate.final_logits is not None:
        shape = candidate.final_logits.shape
        print(f"  Shape: {tuple(shape)}")
        if len(shape) != 3:
            print(f"  [FAILED] Expected 3D tensor, got {len(shape)}D")
            results.append((check, False, f"Got {len(shape)}D"))
        else:
            print("  [PASSED]")
            results.append((check, True, ""))
    else:
        print("  [SKIPPED] (final_logits is None)")
        results.append((check, False, "Skipped — no logits"))

    # CHECK: Sequence length
    check = "Sequence length"
    print(f"\n[CHECK] {check}")
    if candidate.final_logits is not None and len(candidate.final_logits.shape) >= 2:
        logits_seq_len = candidate.final_logits.shape[1]
        print(f"  Expected: {expected_seq_len}, Got: {logits_seq_len}")
        if logits_seq_len != expected_seq_len:
            print("  [FAILED] Sequence length mismatch — possible truncation!")
            results.append((check, False, f"Got {logits_seq_len}, expected {expected_seq_len}"))
        else:
            print("  [PASSED]")
            results.append((check, True, ""))
    else:
        print("  [SKIPPED] (no valid logits)")
        results.append((check, False, "Skipped — no logits"))

    # CHECK: Token count
    check = "Token count"
    print(f"\n[CHECK] {check}")
    print(f"  Expected: {expected_tokens}, Got: {candidate.total_tokens}")
    if candidate.total_tokens != expected_tokens:
        print("  [FAILED] Token count mismatch!")
        results.append((check, False, f"Got {candidate.total_tokens}, expected {expected_tokens}"))
    else:
        print("  [PASSED]")
        results.append((check, True, ""))

    # CHECK: Loss validity
    check = "Loss validity"
    print(f"\n[CHECK] {check}")
    print(f"  Reference loss: {reference.final_loss:.6f}")
    print(f"  Candidate loss: {candidate.final_loss:.6f}")

    if candidate.final_loss != candidate.final_loss:
        print("  [FAILED] Loss is NaN!")
        results.append((check, False, "Loss is NaN"))
    elif candidate.final_loss <= 0:
        print(f"  [FAILED] Loss must be positive, got {candidate.final_loss:.4f}")
        results.append((check, False, f"Non-positive loss: {candidate.final_loss:.4f}"))
    elif candidate.final_loss > 100:
        print(f"  [FAILED] Loss unreasonable: {candidate.final_loss:.4f}")
        results.append((check, False, f"Unreasonable loss: {candidate.final_loss:.4f}"))
    else:
        loss_diff = abs(candidate.final_loss - reference.final_loss)
        print(f"  Loss difference: {loss_diff:.4f} (max allowed: {max_loss_difference})")
        if loss_diff > max_loss_difference:
            print("  [FAILED] Loss difference too large!")
            results.append((check, False, f"Diff {loss_diff:.4f} > {max_loss_difference}"))
        else:
            print("  [PASSED]")
            results.append((check, True, ""))

    # CHECK: Trainable parameters
    check = "Trainable parameters (100%)"
    print(f"\n[CHECK] {check}")
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
        results.append((check, False, f"Only {ratio:.1%} trainable"))
    else:
        print("  [PASSED]")
        results.append((check, True, ""))

    # CHECK: Parameters changed
    check = f"Parameters changed (>={min_changed_ratio:.0%})"
    print(f"\n[CHECK] {check}")
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
        results.append((check, False, f"{changed_ratio:.1%} < {min_changed_ratio:.0%}"))
    else:
        print("  [PASSED]")
        results.append((check, True, ""))

    # CHECK: Gradient capture
    check = "Gradients captured"
    print(f"\n[CHECK] {check}")
    if candidate_grad is None:
        print("  [FAILED] No gradients captured! optimizer.step() must be called via wrapper.")
        print(f"  Wrapper step_count = {capturing_optimizer.step_count} (expected {num_steps})")
        results.append((check, False, "No gradients captured"))
    else:
        print(f"  [PASSED] step_count={capturing_optimizer.step_count}")
        results.append((check, True, ""))

    # CHECK: Gradient relative error
    check = "Gradient relative error"
    relative_error_threshold = gradient_norm_ratio_max - 1.0
    print(f"\n[CHECK] {check}")
    print(f"  Max allowed: {relative_error_threshold:.4f} ({relative_error_threshold * 100:.1f}%)")

    if candidate_grad is None:
        print("  [SKIPPED] No candidate gradients")
        results.append((check, False, "No gradients to compare"))
    else:
        # Coverage
        if candidate_grad.total_layers > 0:
            coverage = candidate_grad.layers_with_grad / candidate_grad.total_layers
            print(
                f"  Gradient coverage: {coverage:.1%}"
                f" ({candidate_grad.layers_with_grad}/{candidate_grad.total_layers})"
            )
            if coverage < 1.0:
                print("  [FAILED] Not all layers have gradients!")
                results.append(("Gradient coverage", False, f"{coverage:.1%} < 100%"))

        ref_vecs = reference_grad.grad_vector
        cand_vecs = candidate_grad.grad_vector
        if ref_vecs and cand_vecs and len(ref_vecs) == len(cand_vecs):
            diff_norm_sq = 0.0
            ref_norm_sq = 0.0
            shape_ok = True
            for ref_layer, cand_layer in zip(ref_vecs, cand_vecs):
                if ref_layer is None or cand_layer is None:
                    continue
                if ref_layer.shape != cand_layer.shape:
                    print(
                        f"  [FAILED] Gradient shape mismatch: {ref_layer.shape} vs {cand_layer.shape}"
                    )
                    shape_ok = False
                    break
                diff = cand_layer - ref_layer
                diff_norm_sq += (diff * diff).sum().item()
                ref_norm_sq += (ref_layer * ref_layer).sum().item()

            if not shape_ok:
                results.append((check, False, "Shape mismatch"))
            else:
                ref_norm = ref_norm_sq**0.5
                diff_norm = diff_norm_sq**0.5
                relative_error = (
                    diff_norm / ref_norm
                    if ref_norm > 0
                    else (0.0 if diff_norm == 0 else float("inf"))
                )

                print(f"  |g - g_truth|: {diff_norm:.6f}")
                print(f"  |g_truth|: {ref_norm:.6f}")
                print(f"  Relative error: {relative_error:.6f}")

                if relative_error > relative_error_threshold:
                    print(f"  [FAILED] {relative_error:.6f} > {relative_error_threshold:.6f}")
                    results.append(
                        (check, False, f"{relative_error:.6f} > {relative_error_threshold:.6f}")
                    )
                else:
                    print("  [PASSED]")
                    results.append((check, True, ""))
        else:
            print("  [FAILED] Gradient vectors unavailable or layer count mismatch")
            results.append((check, False, "Vectors unavailable"))

    # CHECK: Final weight verification
    check = "Weight relative error"
    print(f"\n[CHECK] {check}")
    print(
        f"  Max allowed: {weight_relative_error_max:.4f} ({weight_relative_error_max * 100:.1f}%)"
    )

    if reference_final_state is not None:
        w_diff_norm_sq = 0.0
        w_ref_norm_sq = 0.0
        w_total_elements = 0
        w_mismatched_layers = 0

        for name, param in model.named_parameters():
            if name not in reference_final_state:
                continue
            ref_param = reference_final_state[name].to(param.device)
            diff = param.data.float() - ref_param.float()

            layer_diff_sq = (diff * diff).sum().item()
            layer_ref_sq = (ref_param.float() * ref_param.float()).sum().item()

            w_diff_norm_sq += layer_diff_sq
            w_ref_norm_sq += layer_ref_sq
            w_total_elements += param.numel()

            if layer_ref_sq > 0:
                layer_rel_error = (layer_diff_sq**0.5) / (layer_ref_sq**0.5)
                if layer_rel_error > weight_relative_error_max:
                    w_mismatched_layers += 1

        w_ref_norm = w_ref_norm_sq**0.5
        w_diff_norm = w_diff_norm_sq**0.5
        w_relative_error = (
            w_diff_norm / w_ref_norm
            if w_ref_norm > 0
            else (0.0 if w_diff_norm == 0 else float("inf"))
        )

        print(f"  |w_miner - w_ref|: {w_diff_norm:.6f}")
        print(f"  |w_ref|: {w_ref_norm:.6f}")
        print(f"  Relative error: {w_relative_error:.6f}")
        print(f"  Total elements: {w_total_elements:,}")
        print(f"  Mismatched layers: {w_mismatched_layers}")

        if w_relative_error > weight_relative_error_max:
            print(f"  [FAILED] {w_relative_error:.6f} > {weight_relative_error_max:.6f}")
            results.append(
                (check, False, f"{w_relative_error:.6f} > {weight_relative_error_max:.6f}")
            )
        else:
            print("  [PASSED]")
            results.append((check, True, ""))
    else:
        print("  [SKIPPED] (no reference final state available)")
        results.append((check, False, "Skipped — no reference state"))

    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    _print_summary(results)

    all_passed = all(passed for _, passed, _ in results)
    sys.exit(0 if all_passed else 1)


def _print_summary(results: list[tuple[str, bool, str]]) -> None:
    """Print final summary of all check results."""
    passed = [(name, detail) for name, ok, detail in results if ok]
    failed = [(name, detail) for name, ok, detail in results if not ok]

    print()
    print("=" * 70)
    print(f"SUMMARY: {len(passed)} passed, {len(failed)} failed")
    print("=" * 70)

    if passed:
        print()
        print("PASSED:")
        for name, _ in passed:
            print(f"  [PASS] {name}")

    if failed:
        print()
        print("FAILED:")
        for name, detail in failed:
            print(f"  [FAIL] {name}: {detail}")

    print()
    if not failed:
        print("Your submission should pass validator evaluation!")
    else:
        print("Fix the issues above before submitting.")
    print("=" * 70)


if __name__ == "__main__":
    main()
