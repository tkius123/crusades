"""
Templar TPS Evaluation Environment (Validator-Owned)

URL-Based Architecture:
- This env.py is owned by the VALIDATOR, not the miner
- Miner commits a URL pointing to their train.py code
- Validator downloads code from URL and passes it to this env
- This ensures miners can't tamper with evaluation logic

Flow:
1. Validator reads miner commitment (contains code URL)
2. Validator downloads train.py from URL
3. Validator calls env.evaluate(code="...", ...)
4. This Actor runs benchmark and returns MFU
"""

import ast
import gc
import hashlib
import importlib.util
import logging
import os
import random
import sys
import time
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel

# Setup logging - only for this module, don't configure root logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)

# Suppress noisy loggers from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

# Saved references for use in evaluation timing
_perf_counter = time.perf_counter
_cuda_synchronize = torch.cuda.synchronize if torch.cuda.is_available() else lambda: None

# Configuration from environment variables
DETERMINISTIC_MODE = os.getenv("DETERMINISTIC_MODE", "1") == "1"
EVAL_SEQUENCE_LENGTH = int(os.getenv("EVAL_SEQUENCE_LENGTH", "1024"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", "/tmp/templar_eval"))


@dataclass
class InnerStepsResult:
    """Expected return type from miner's inner_steps function."""

    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float


@dataclass
class GradientInfo:
    """Captured gradient information for verification.

    MEMORY OPTIMIZATION: grad_vector is a list of per-layer tensors stored on CPU,
    rather than a single concatenated tensor. This avoids allocating ~6-12GB for
    large models. Cosine similarity is computed incrementally layer-by-layer.
    """

    grad_norm: float  # L2 norm of all gradients
    grad_vector: list | None = None  # List of per-layer gradient tensors (on CPU)
    layers_with_grad: int = 0  # Layers with non-zero gradients
    total_layers: int = 0  # Total trainable layers


def _log_vram(tag: str):
    """Log current VRAM usage for debugging memory leaks."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    logger.info(f"[VRAM {tag}] allocated={allocated:.2f}GB reserved={reserved:.2f}GB")


# Global cache for model (data is NOT cached for validators)
_CACHE = {
    "model": None,
    "model_path": None,
    "initial_state": None,
    "use_random_init": None,
}


def _load_miner_module(train_path: Path):
    """Dynamically load miner's train.py as a module.

    Args:
        train_path: Path to train.py

    Returns:
        Loaded module with inner_steps function
    """
    spec = importlib.util.spec_from_file_location("miner_train", train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {train_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["miner_train"] = module
    spec.loader.exec_module(module)
    return module


def _reset_torch_state():
    """Reset torch global state between evaluations."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    _enforce_backend_state()
    torch.set_grad_enabled(True)
    torch.set_default_dtype(torch.float32)

    # Clear torch.compile caches (miner code may have used torch.compile,
    # which holds references to model tensors and compiled graphs)
    try:
        torch._dynamo.reset()
    except Exception:
        pass

    # Remove miner module from sys.modules to release all references
    # (closures, compiled functions, global state in the miner's code)
    if "miner_train" in sys.modules:
        del sys.modules["miner_train"]

    # Restore saved references
    time.perf_counter = _perf_counter
    if torch.cuda.is_available():
        torch.cuda.synchronize = _cuda_synchronize

    logger.debug("Torch state reset complete")


# Canonical backend settings used during reference and timed evaluation.
# Both runs MUST use identical settings so gradients/weights can be compared.
# Changing e.g. cudnn.benchmark or deterministic causes non-deterministic
# algorithm selection, making the reference vs miner comparison unreliable.
_REFERENCE_BACKEND_STATE = {
    "cudnn.deterministic": True,
    "cudnn.benchmark": False,
    "cudnn.allow_tf32": True,
    "float32_matmul_precision": "highest",
    "flash_sdp_enabled": True,
    "mem_efficient_sdp_enabled": True,
    "math_sdp_enabled": True,
}
# NOTE: cuda.matmul.allow_tf32 is intentionally omitted.
# torch.set_float32_matmul_precision("highest") overrides it to False,
# so setting it to True and then checking it would always fail.
# The matmul precision setting is the authoritative control for TF32 in matmuls.


def _enforce_backend_state():
    """Set all torch backend settings to the canonical reference values."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    # set_float32_matmul_precision("highest") is the authoritative control
    # for matmul TF32 precision. It internally sets cuda.matmul.allow_tf32,
    # so we do NOT set allow_tf32 separately (they would conflict).
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)


def _check_backend_state() -> list[str]:
    """Check all torch backend settings against canonical reference values.

    Returns a list of setting names that were tampered with (empty if all OK).
    """
    violations = []
    if not torch.backends.cudnn.deterministic:
        violations.append("cudnn.deterministic")
    if torch.backends.cudnn.benchmark:
        violations.append("cudnn.benchmark")
    if not torch.backends.cudnn.allow_tf32:
        violations.append("cudnn.allow_tf32")
    if torch.get_float32_matmul_precision() != "highest":
        violations.append("float32_matmul_precision")
    if not torch.backends.cuda.flash_sdp_enabled():
        violations.append("flash_sdp_enabled")
    if not torch.backends.cuda.mem_efficient_sdp_enabled():
        violations.append("mem_efficient_sdp_enabled")
    if not torch.backends.cuda.math_sdp_enabled():
        violations.append("math_sdp_enabled")
    return violations


def _load_hf_dataset(
    dataset_name: str,
    model_name: str,
    num_samples: int = 10000,
    sequence_length: int = 1024,
    split: str = "train",
    validator_seed: str | None = None,
) -> torch.Tensor:
    """Load and tokenize dataset from HuggingFace or local cache.

    When running with --network none, uses pre-cached dataset from Docker image.
    The validator seed is used to shuffle the cached data.

    Args:
        dataset_name: HuggingFace dataset name
        model_name: Model name for tokenizer
        num_samples: Number of samples to load
        sequence_length: Sequence length
        split: Dataset split
        validator_seed: Seed string from validator (for unpredictable sampling)

    Returns:
        Tensor of shape [num_samples, sequence_length]
    """
    import json

    from transformers import AutoTokenizer

    # Determine seed: validators use unpredictable seed
    if validator_seed:
        seed_hash = hashlib.sha256(validator_seed.encode()).hexdigest()
        actual_seed = int(seed_hash[:8], 16)
        logger.info(f"Validator mode: seed={actual_seed} (from {validator_seed})")
    else:
        actual_seed = 42
        logger.info(f"Test mode: fixed seed={actual_seed}")

    logger.info(f"Loading dataset: {dataset_name} (samples={num_samples})")

    # Cache tokenizer to avoid reloading every eval (HF tokenizers leak via Rust FFI)
    if _CACHE.get("tokenizer") is not None and _CACHE.get("tokenizer_model") == model_name:
        tokenizer = _CACHE["tokenizer"]
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _CACHE["tokenizer"] = tokenizer
        _CACHE["tokenizer_model"] = model_name

    # Check for cached dataset (enables --network none operation)
    cached_path = os.getenv("CACHED_DATASET_PATH", "/home/appuser/.cache/templar/dataset.json")

    if Path(cached_path).exists():
        # Tokenize ALL samples once and cache the full tensor.
        # Subsequent evals just shuffle indices (fast) instead of
        # re-tokenizing 10k samples (slow + leaks memory).
        cache_key = f"tokenized_{cached_path}_{sequence_length}"
        all_tokenized = _CACHE.get(cache_key)

        if all_tokenized is None:
            logger.info(f"First load — tokenizing cached dataset: {cached_path}")
            with open(cached_path) as f:
                all_samples = json.load(f)

            tokens_list = []
            for text in all_samples:
                encoded = tokenizer(
                    text,
                    max_length=sequence_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                tokens_list.append(encoded["input_ids"].squeeze(0))

            if not tokens_list:
                raise ValueError("No samples in cached dataset")

            all_tokenized = torch.stack(tokens_list)
            _CACHE[cache_key] = all_tokenized
            logger.info(f"Tokenized and cached: {all_tokenized.shape}")
        else:
            logger.info(f"Using pre-tokenized cache: {all_tokenized.shape}")

        # Shuffle indices with validator seed for unpredictability
        total = all_tokenized.size(0)
        indices = list(range(total))
        rng = random.Random(actual_seed)
        rng.shuffle(indices)
        selected = indices[:num_samples]

        data = all_tokenized[selected]
        logger.info(f"Sampled data: shape={data.shape}, seed={actual_seed}")
        return data

    # Fallback: Load from HuggingFace (requires network)
    logger.info("No cache found, loading from HuggingFace...")
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    dataset = dataset.shuffle(seed=actual_seed, buffer_size=10000)

    tokens_list = []
    dataset_iter = iter(dataset)

    for _ in range(num_samples):
        try:
            sample = next(dataset_iter)
            text = sample.get("text", sample.get("content", ""))

            encoded = tokenizer(
                text,
                max_length=sequence_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            tokens_list.append(encoded["input_ids"].squeeze(0))

        except StopIteration:
            break

    if not tokens_list:
        raise ValueError(f"No samples loaded from {dataset_name}")

    data = torch.stack(tokens_list)
    logger.info(f"Loaded data: shape={data.shape}, seed={actual_seed}")
    return data


def _set_deterministic(seed: int) -> None:
    """Set deterministic mode for reproducibility."""
    if not DETERMINISTIC_MODE:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _scan_for_dangerous_patterns(tree: ast.AST) -> tuple[bool, str | None]:
    """AST scan to reject forbidden code patterns."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ("__setattr__", "__delattr__"):
            if isinstance(node.value, ast.Name) and node.value.id == "object":
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: forbidden pattern detected"

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "__class__":
                    line = getattr(node, "lineno", "?")
                    return False, f"Line {line}: forbidden pattern detected"

        if isinstance(node, ast.Attribute) and node.attr == "__class__":
            if not isinstance(getattr(node, "_parent", None), ast.AnnAssign):
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: forbidden pattern detected"

        if isinstance(node, ast.Attribute) and node.attr == "perf_counter":
            if isinstance(node.value, ast.Name) and node.value.id == "time":
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: forbidden pattern detected"

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "synchronize":
                    line = getattr(node, "lineno", "?")
                    return False, f"Line {line}: forbidden pattern detected"

        if isinstance(node, ast.Attribute) and node.attr == "__slots__":
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: forbidden pattern detected"

        # Block: gc introspection (gc.get_objects, gc.get_referrers, gc.get_referents)
        if isinstance(node, ast.Attribute) and node.attr in (
            "get_objects",
            "get_referrers",
            "get_referents",
        ):
            if isinstance(node.value, ast.Name) and node.value.id == "gc":
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: forbidden pattern detected"

        # Block: torch backend setting modifications (deterministic, benchmark)
        if isinstance(node, ast.Attribute) and node.attr in (
            "deterministic",
            "benchmark",
        ):
            if isinstance(node.ctx, ast.Store):
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: forbidden pattern detected"

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
                return False, f"Line {line}: forbidden pattern detected"

        # Block: dynamic code execution (exec, eval, compile, __import__)
        # These bypass AST-level checks by running arbitrary code at runtime.
        # torch.compile is safe (ast.Attribute, not ast.Name).
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ("exec", "eval", "compile", "__import__"):
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: {node.func.id}() is forbidden"

        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("ctypes", "_ctypes", "gc", "subprocess") or alias.name.startswith(
                    "importlib"
                ):
                    line = getattr(node, "lineno", "?")
                    return False, f"Line {line}: forbidden import"

        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module in ("ctypes", "_ctypes", "gc", "subprocess") or node.module.startswith(
                "importlib"
            ):
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: forbidden import"

        # Block: accessing .optimizer attribute (wrapper bypass attempt)
        if isinstance(node, ast.Attribute) and node.attr == "optimizer":
            # Allow 'self.optimizer' inside class definitions, but block
            # attempts to unwrap the GradientCapturingOptimizer via optimizer.optimizer
            if not (isinstance(node.value, ast.Name) and node.value.id == "self"):
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: accessing .optimizer attribute is forbidden"

    return True, None


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


def _validate_code_structure(code: str) -> tuple[bool, str | None]:
    """Validate that train.py has the required structure."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"Syntax error at line {exc.lineno}: {exc.msg}"

    # Strip `if __name__ == "__main__":` blocks before security scanning.
    # These blocks are for local testing only and never execute during
    # evaluation (the validator imports inner_steps, it doesn't run __main__).
    scan_tree = ast.Module(
        body=[node for node in tree.body if not _is_main_guard(node)],
        type_ignores=tree.type_ignores,
    )

    safe, danger_error = _scan_for_dangerous_patterns(scan_tree)
    if not safe:
        return False, f"Security violation: {danger_error}"

    # Scan string literals for forbidden patterns (also skip __main__ blocks)
    for node in ast.walk(scan_tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for pattern in _FORBIDDEN_STRINGS:
                if pattern in node.value:
                    line = getattr(node, "lineno", "?")
                    return False, (
                        f"Security violation: Line {line}: forbidden string pattern detected"
                    )

    inner_steps_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "inner_steps":
            inner_steps_found = True
            args = node.args
            if len(args.args) < 5:
                return False, f"inner_steps has {len(args.args)} args, expected at least 5"
            break

    if not inner_steps_found:
        return False, "Missing required function: inner_steps"

    return True, None


def _validate_return_type(result) -> tuple[bool, str | None, InnerStepsResult | None]:
    """Validate that inner_steps returned correct type."""
    if isinstance(result, InnerStepsResult):
        return True, None, result

    if all(hasattr(result, attr) for attr in ("final_logits", "total_tokens", "final_loss")):
        return (
            True,
            None,
            InnerStepsResult(
                final_logits=result.final_logits,
                total_tokens=result.total_tokens,
                final_loss=result.final_loss,
            ),
        )

    return False, f"Invalid return type from inner_steps: {type(result)}", None


def _load_model(model_path: str, use_random_init: bool = False):
    """Load model from HuggingFace.

    Args:
        model_path: HuggingFace model name/path
        use_random_init: If True, initialize with random weights

    Note:
        Docker evaluation runs with --network=none. Models must be
        pre-cached in the Docker image. If cache is missing, rebuild:
        docker build --network=host -f environments/templar/Dockerfile -t templar-eval:latest .
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    if use_random_init:
        # Random initialization
        logger.info(f"Loading model config from {model_path} with RANDOM initialization")

        # Use local cache only (Docker runs with --network=none)
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        # Simple model loading - Qwen2.5-3B (~6GB) fits easily on A100 80GB
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        model = model.to(device)
        logger.info(f"Model loaded on {device}")
    else:
        logger.info(f"Loading pretrained model from {model_path}")
        # Use local cache only (Docker runs with --network=none)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
            local_files_only=True,
        )

    model.train()

    # Ensure ALL parameters have requires_grad=True for training
    for param in model.parameters():
        param.requires_grad = True

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


def _count_model_params(model: torch.nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())


def _calculate_mfu(
    total_tokens: int,
    wall_time: float,
    model_params: int,
    gpu_peak_tflops: float = 312.0,
) -> float:
    """Calculate Model FLOPs Utilization (MFU).

    MFU = actual_flops / theoretical_peak_flops

    For transformer training (forward + backward):
    - Forward: 2 * params * tokens (multiply-accumulate)
    - Backward: 4 * params * tokens (gradients are 2x forward)
    - Total: 6 * params * tokens

    Args:
        total_tokens: Total tokens processed
        wall_time: Wall clock time in seconds
        model_params: Number of model parameters
        gpu_peak_tflops: GPU theoretical peak TFLOPS (A100 80GB = 312 TFLOPS bfloat16)

    Returns:
        MFU as a percentage (0-100)
    """
    # Guard against division by zero / invalid inputs
    if wall_time <= 0 or gpu_peak_tflops <= 0:
        return 0.0

    # FLOPs for forward + backward pass
    flops_per_token = 6 * model_params
    total_flops = flops_per_token * total_tokens

    # Actual TFLOPS achieved
    actual_tflops = total_flops / wall_time / 1e12

    # MFU as percentage
    mfu = (actual_tflops / gpu_peak_tflops) * 100

    return min(mfu, 100.0)  # Cap at 100%


def _capture_gradients(model: torch.nn.Module) -> GradientInfo:
    """Capture gradient information from model after backward pass.

    MEMORY OPTIMIZATION: Instead of concatenating all gradients into a single
    large tensor (which can be ~6-12GB for 3B models), we store per-layer
    gradient vectors on CPU. Cosine similarity is computed incrementally
    in _verify_gradients() to avoid memory pressure.

    Returns:
        GradientInfo with norm and per-layer gradient vectors (on CPU)
    """
    grad_vectors_cpu = []  # Store per-layer on CPU to save GPU memory
    total_norm_sq = 0.0
    layers_with_grad = 0
    layers_without_grad = 0

    for param in model.parameters():
        if param.grad is not None:
            # Move to CPU immediately to free GPU memory
            grad_flat = param.grad.detach().cpu().float().view(-1)
            total_norm_sq += grad_flat.pow(2).sum().item()
            # Check if gradient is actually non-zero (not just allocated)
            if grad_flat.abs().sum().item() > 1e-10:
                layers_with_grad += 1
                grad_vectors_cpu.append(grad_flat)
            else:
                layers_without_grad += 1
                grad_vectors_cpu.append(grad_flat)  # Still store for shape matching
        else:
            layers_without_grad += 1
            grad_vectors_cpu.append(None)

    grad_norm = total_norm_sq**0.5

    # Log layer gradient coverage
    total_layers = layers_with_grad + layers_without_grad
    if total_layers > 0:
        coverage = layers_with_grad / total_layers
        logger.info(f"Gradient coverage: {layers_with_grad}/{total_layers} layers ({coverage:.1%})")
        if layers_without_grad > 0:
            logger.warning(f"WARNING: {layers_without_grad} layers have zero/no gradients!")

    return GradientInfo(
        grad_norm=grad_norm,
        grad_vector=grad_vectors_cpu,  # Now a list of per-layer tensors on CPU
        layers_with_grad=layers_with_grad,
        total_layers=total_layers,
    )


def _verify_trainable_params(
    model: torch.nn.Module,
    min_trainable_ratio: float = 1.0,
) -> tuple[bool, str | None, dict]:
    """Check that minimum % of params are trainable (prevents layer freezing)."""
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    trainable_ratio = trainable_params / total_params if total_params > 0 else 0.0

    details = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_ratio,
        "min_required": min_trainable_ratio,
    }

    if trainable_ratio < min_trainable_ratio:
        error = (
            f"Insufficient trainable params: {trainable_ratio:.1%} "
            f"({trainable_params:,}/{total_params:,}) - minimum {min_trainable_ratio:.0%} required"
        )
        logger.error(f"[FAILED] {error}")
        return False, error, details

    logger.info(
        f"[PASSED] Trainable params check ({trainable_ratio:.1%} >= {min_trainable_ratio:.0%})"
    )
    return True, None, details


def _verify_params_changed(
    model: torch.nn.Module,
    initial_state: dict,
    min_changed_ratio: float = 0.5,
    element_threshold: float = 1e-6,
) -> tuple[bool, str | None, dict]:
    """Verify that minimum % of individual parameter elements changed during training.

    Args:
        model: Model after training
        initial_state: Model state before training
        min_changed_ratio: Minimum fraction of elements that must change
        element_threshold: Minimum absolute change for an element to count as "changed"
    """
    total_elements = 0
    changed_elements = 0

    for name, param in model.named_parameters():
        if name in initial_state:
            initial = initial_state[name].to(param.device)
            # Count individual elements that changed (not whole tensors)
            element_diffs = (param.data - initial).abs()
            changed_mask = element_diffs > element_threshold

            total_elements += param.numel()
            changed_elements += changed_mask.sum().item()

    changed_ratio = changed_elements / total_elements if total_elements > 0 else 0.0

    details = {
        "total_elements": total_elements,
        "changed_elements": int(changed_elements),
        "changed_ratio": changed_ratio,
        "min_required": min_changed_ratio,
        "element_threshold": element_threshold,
    }

    if changed_ratio < min_changed_ratio:
        error = (
            f"Insufficient parameter updates: {changed_ratio:.1%} "
            f"({int(changed_elements):,}/{total_elements:,} elements) - minimum {min_changed_ratio:.0%} required"
        )
        logger.error(f"[FAILED] {error}")
        return False, error, details

    logger.info(
        f"[PASSED] Parameter changes check ({changed_ratio:.1%} >= {min_changed_ratio:.0%})"
    )
    return True, None, details


def _get_cached_model(model_path: str, use_random_init: bool = False):
    """Get model from cache or load it."""
    cached = _CACHE.get("model")
    cached_path = _CACHE.get("model_path")
    cached_random_init = _CACHE.get("use_random_init")

    # Cache hit only if path AND init mode match
    if cached is not None and cached_path == model_path and cached_random_init == use_random_init:
        if _CACHE.get("initial_state"):
            cached.load_state_dict(_CACHE["initial_state"])
        return cached

    model = _load_model(model_path, use_random_init=use_random_init)
    _CACHE["model"] = model
    _CACHE["model_path"] = model_path
    _CACHE["use_random_init"] = use_random_init
    _CACHE["initial_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return model


def _create_data_iterator(
    data: torch.Tensor, batch_size: int, sequence_length: int
) -> Iterator[torch.Tensor]:
    """Create infinite data iterator."""
    if data.size(1) < sequence_length:
        raise ValueError(f"Data sequence length {data.size(1)} < required {sequence_length}")

    data = data[:, :sequence_length]
    num_samples = data.size(0)

    def _iter():
        idx = 0
        while True:
            end_idx = idx + batch_size
            if end_idx > num_samples:
                idx = 0
                end_idx = batch_size
            yield data[idx:end_idx]
            idx = end_idx

    return _iter()


def _create_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    """Create standard AdamW optimizer (fused on CUDA for performance)."""
    use_fused = torch.cuda.is_available()
    return torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        fused=use_fused,
    )


class GradientCapturingOptimizer:
    """Optimizer wrapper that snapshots gradients on the final step.

    On the last step, clones gradient tensors on GPU (fast, ~3ms) before
    optimizer.step() clears them. The slow CPU transfer + norm computation
    happens AFTER the timer stops via finalize_gradients().

    This keeps the timed section free of validator overhead.
    Read-only after initialization.

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

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        num_steps: int,
    ):
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
        """On the final step, snapshot gradients on GPU before optimizer.step().

        Only clones on the last step (step_count == num_steps - 1).
        GPU-to-GPU clone is ~3ms for 3B params vs ~7s for GPU→CPU transfer.
        The slow CPU conversion happens in finalize_gradients() after the timer.
        """
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
        """Convert GPU gradient snapshot to GradientInfo (CPU).

        Call AFTER the timer stops. This does the slow GPU→CPU transfer
        and norm computation outside the timed section.
        """
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

        grad_norm = total_norm_sq**0.5
        total_layers = layers_with_grad + layers_without_grad
        if total_layers > 0:
            coverage = layers_with_grad / total_layers
            logger.info(
                f"Gradient coverage: {layers_with_grad}/{total_layers} layers ({coverage:.1%})"
            )
            if layers_without_grad > 0:
                logger.warning(f"WARNING: {layers_without_grad} layers have zero/no gradients!")

        object.__setattr__(
            self,
            "captured_gradients",
            GradientInfo(
                grad_norm=grad_norm,
                grad_vector=grad_vectors_cpu,
                layers_with_grad=layers_with_grad,
                total_layers=total_layers,
            ),
        )

        # Free GPU snapshot memory
        object.__setattr__(self, "_grad_snapshot_gpu", None)

    def zero_grad(self, set_to_none: bool = False):
        """Forward to underlying optimizer."""
        opt = object.__getattribute__(self, "_opt_impl")
        return opt.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        """Forward param_groups access to underlying optimizer."""
        opt = object.__getattribute__(self, "_opt_impl")
        return opt.param_groups

    @param_groups.setter
    def param_groups(self, value):
        """Forward param_groups setter to underlying optimizer."""
        opt = object.__getattribute__(self, "_opt_impl")
        opt.param_groups = value

    def state_dict(self):
        """Forward to underlying optimizer."""
        opt = object.__getattribute__(self, "_opt_impl")
        return opt.state_dict()

    def load_state_dict(self, state_dict):
        """Forward to underlying optimizer."""
        opt = object.__getattribute__(self, "_opt_impl")
        return opt.load_state_dict(state_dict)

    def add_param_group(self, param_group):
        """Forward to underlying optimizer."""
        opt = object.__getattribute__(self, "_opt_impl")
        return opt.add_param_group(param_group)

    @property
    def state(self):
        """Forward state access to underlying optimizer."""
        opt = object.__getattribute__(self, "_opt_impl")
        return opt.state

    def __getattr__(self, name):
        """Forward non-private attribute access to underlying optimizer."""
        # Block private attrs that fell through from __getattribute__ raising
        if name.startswith("_") and not (name.startswith("__") and name.endswith("__")):
            raise AttributeError(f"Access to '{name}' is not allowed on optimizer wrapper")
        opt = object.__getattribute__(self, "_opt_impl")
        return getattr(opt, name)


def _run_reference(
    model: torch.nn.Module,
    data_iterator: Iterator[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    device: torch.device,
    capture_final_gradients: bool = True,
) -> tuple[InnerStepsResult, GradientInfo | None]:
    """Run reference implementation for comparison.

    Returns:
        Tuple of (InnerStepsResult, GradientInfo) where GradientInfo
        contains the gradients from the final step (before optimizer.step)
    """
    _enforce_backend_state()

    total_tokens = 0
    final_logits = None
    final_loss = 0.0
    final_gradients = None

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

        # Capture gradients on final step (before optimizer.step clears them)
        if capture_final_gradients and step == num_steps - 1:
            final_gradients = _capture_gradients(model)

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
    return result, final_gradients


def _verify_gradients(
    reference_grad: GradientInfo | None,
    candidate_grad: GradientInfo | None,
    norm_ratio_max: float = 1.02,
) -> tuple[bool, str | None, dict]:
    """Verify candidate gradients match reference using relative error |g - g_truth| / |g_truth|.

    This is a direct comparison that catches any deviation including truncation,
    layer freezing, and step skipping. The threshold should be near numerical
    precision (bfloat16 has ~0.8% relative error for accumulated ops).

    Args:
        reference_grad: Reference implementation gradients
        candidate_grad: Miner's implementation gradients
        norm_ratio_max: Encoded as 1 + max_relative_error. e.g., 1.04 means 4% max error.
            The relative error threshold is derived as norm_ratio_max - 1.0.

    Returns:
        Tuple of (success, error_message, details)
    """
    # Derive relative error threshold from norm_ratio_max
    # e.g., 1.04 -> 0.04 (4% relative error allowed)
    relative_error_threshold = norm_ratio_max - 1.0

    details = {
        "relative_error_threshold": relative_error_threshold,
        "checks_passed": [],
        "checks_failed": [],
    }

    logger.info("=" * 60)
    logger.info("VERIFICATION: Gradient-based verification")
    logger.info("=" * 60)

    if reference_grad is None or candidate_grad is None:
        error = "Missing gradient information for verification"
        details["checks_failed"].append({"check": "gradient_availability", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details

    details["reference_grad_norm"] = reference_grad.grad_norm
    details["candidate_grad_norm"] = candidate_grad.grad_norm
    details["candidate_layers_with_grad"] = candidate_grad.layers_with_grad
    details["candidate_total_layers"] = candidate_grad.total_layers

    # Check 0: All layers must have gradients
    if candidate_grad.total_layers > 0:
        grad_coverage = candidate_grad.layers_with_grad / candidate_grad.total_layers
        details["gradient_coverage"] = grad_coverage
        logger.info(f"[CHECK 0/2] Gradient coverage: {grad_coverage:.1%}")
        logger.info(
            f"   Layers with gradients: {candidate_grad.layers_with_grad}/{candidate_grad.total_layers}"
        )

        if grad_coverage < 1.0:
            error = (
                f"Not all layers have gradients: {candidate_grad.layers_with_grad}/{candidate_grad.total_layers} "
                f"({grad_coverage:.1%}) - possible layer freezing detected"
            )
            details["checks_failed"].append({"check": "gradient_coverage", "error": error})
            details["error_code"] = "gradient_coverage_failed"
            logger.error(f"[FAILED] {error}")
            return False, error, details
        details["checks_passed"].append("gradient_coverage")
        logger.info("[PASSED] All layers have gradients")

    # Check 1: Relative gradient error |g - g_truth| / |g_truth|
    # Catches ALL deviations: truncation, layer freezing, step skipping, etc.
    ref_vecs = reference_grad.grad_vector
    cand_vecs = candidate_grad.grad_vector

    if ref_vecs is not None and cand_vecs is not None and len(ref_vecs) > 0:
        if len(ref_vecs) != len(cand_vecs):
            error = f"Gradient layer count mismatch: ref={len(ref_vecs)}, cand={len(cand_vecs)}"
            details["checks_failed"].append({"check": "gradient_shape", "error": error})
            logger.error(f"[FAILED] {error}")
            return False, error, details

        # Compute |g - g_truth|^2 and |g_truth|^2 incrementally (layer-by-layer)
        diff_norm_sq = 0.0
        ref_norm_sq = 0.0
        total_elements = 0

        for ref_layer, cand_layer in zip(ref_vecs, cand_vecs):
            if ref_layer is None or cand_layer is None:
                continue

            if ref_layer.shape != cand_layer.shape:
                error = f"Gradient shape mismatch at layer: ref={ref_layer.shape}, cand={cand_layer.shape}"
                details["checks_failed"].append({"check": "gradient_shape", "error": error})
                logger.error(f"[FAILED] {error}")
                return False, error, details

            diff = cand_layer - ref_layer
            diff_norm_sq += (diff * diff).sum().item()
            ref_norm_sq += (ref_layer * ref_layer).sum().item()
            total_elements += ref_layer.numel()

        ref_norm = ref_norm_sq**0.5
        diff_norm = diff_norm_sq**0.5

        if ref_norm > 0:
            relative_error = diff_norm / ref_norm
        else:
            relative_error = 0.0 if diff_norm == 0 else float("inf")

        details["relative_error"] = relative_error
        details["diff_norm"] = diff_norm
        details["ref_norm"] = ref_norm
        details["gradient_elements"] = total_elements

        logger.info(f"[CHECK 1/2] Gradient relative error: {relative_error:.6f}")
        logger.info(f"   |g - g_truth|: {diff_norm:.6f}")
        logger.info(f"   |g_truth|: {ref_norm:.6f}")
        logger.info(f"   Max allowed: {relative_error_threshold:.6f}")
        logger.info(f"   Total gradient elements: {total_elements:,}")

        if relative_error > relative_error_threshold:
            error = (
                f"Gradient relative error {relative_error:.6f} exceeds threshold "
                f"{relative_error_threshold:.6f} (|g - g_truth| / |g_truth|)"
            )
            details["checks_failed"].append({"check": "gradient_relative_error", "error": error})
            details["error_code"] = "gradient_relative_error_failed"
            logger.error(f"[FAILED] {error}")
            return False, error, details
        details["checks_passed"].append("gradient_relative_error")
        logger.info("[PASSED] Gradient relative error within threshold")
    else:
        logger.warning("Gradient vectors unavailable, skipping relative error check")

    logger.info("=" * 60)
    logger.info("VERIFICATION: GRADIENT CHECKS PASSED")
    logger.info(f"   Checks passed: {details['checks_passed']}")
    logger.info("=" * 60)

    return True, None, details


def _verify_final_weights(
    model: torch.nn.Module,
    reference_final_state: dict,
    max_relative_error: float = 0.05,
) -> tuple[bool, str | None, dict]:
    """Verify miner's final model weights match reference after full training.

    Compares miner final weights against reference to verify the entire
    training pipeline end-to-end. Computed layer-by-layer on CPU to avoid
    GPU memory pressure.

    Args:
        model: Model after miner's training (post optimizer steps)
        reference_final_state: Model state dict after reference training
        max_relative_error: Maximum allowed |w_miner - w_ref| / |w_ref|

    Returns:
        Tuple of (success, error_message, details)
    """
    details: dict = {
        "relative_error_threshold": max_relative_error,
        "checks_passed": [],
        "checks_failed": [],
    }

    logger.info("=" * 60)
    logger.info("VERIFICATION: Final model weight comparison")
    logger.info("=" * 60)

    # Compute relative error layer-by-layer (memory efficient)
    diff_norm_sq = 0.0
    ref_norm_sq = 0.0
    total_elements = 0
    mismatched_layers = 0

    for name, param in model.named_parameters():
        if name not in reference_final_state:
            continue

        # Move reference to same device for comparison, then back to CPU
        ref_param = reference_final_state[name].to(param.device)
        diff = param.data.float() - ref_param.float()

        layer_diff_sq = (diff * diff).sum().item()
        layer_ref_sq = (ref_param.float() * ref_param.float()).sum().item()

        diff_norm_sq += layer_diff_sq
        ref_norm_sq += layer_ref_sq
        total_elements += param.numel()

        # Track layers with significant differences
        if layer_ref_sq > 0:
            layer_rel_error = (layer_diff_sq**0.5) / (layer_ref_sq**0.5)
            if layer_rel_error > max_relative_error:
                mismatched_layers += 1

    ref_norm = ref_norm_sq**0.5
    diff_norm = diff_norm_sq**0.5

    if ref_norm > 0:
        relative_error = diff_norm / ref_norm
    else:
        relative_error = 0.0 if diff_norm == 0 else float("inf")

    details["relative_error"] = relative_error
    details["diff_norm"] = diff_norm
    details["ref_norm"] = ref_norm
    details["total_elements"] = total_elements
    details["mismatched_layers"] = mismatched_layers

    logger.info(f"[CHECK] Weight relative error: {relative_error:.6f}")
    logger.info(f"   |w_miner - w_ref|: {diff_norm:.6f}")
    logger.info(f"   |w_ref|: {ref_norm:.6f}")
    logger.info(f"   Max allowed: {max_relative_error:.6f}")
    logger.info(f"   Total elements: {total_elements:,}")
    logger.info(f"   Mismatched layers: {mismatched_layers}")

    if relative_error > max_relative_error:
        error = (
            f"Final weight relative error {relative_error:.6f} exceeds threshold "
            f"{max_relative_error:.6f} (|w_miner - w_ref| / |w_ref|)"
        )
        details["checks_failed"].append({"check": "weight_relative_error", "error": error})
        details["error_code"] = "weight_mismatch"
        logger.error(f"[FAILED] {error}")
        return False, error, details

    details["checks_passed"].append("weight_relative_error")
    logger.info("[PASSED] Final model weights match reference")

    logger.info("=" * 60)
    logger.info("VERIFICATION: WEIGHT CHECK PASSED")
    logger.info("=" * 60)

    return True, None, details


def _verify_outputs(
    reference: InnerStepsResult,
    candidate: InnerStepsResult,
    expected_tokens: int,
    reference_grad: GradientInfo | None = None,
    candidate_grad: GradientInfo | None = None,
    reference_final_state: dict | None = None,
    model: torch.nn.Module | None = None,
    max_loss_difference: float = 0.5,
    gradient_norm_ratio_max: float = 1.02,
    weight_relative_error_max: float = 0.05,
) -> tuple[bool, str | None, dict]:
    """Verify candidate outputs match reference.

    Verification checks:
    1. Token count matches expected
    2. Loss is valid and similar to reference (small difference)
    3. Gradient-based verification (captures backward pass correctness)
    4. Final weight verification (captures optimizer step correctness)

    Args:
        reference: Reference implementation results
        candidate: Miner's implementation results
        expected_tokens: Expected token count
        reference_grad: Reference gradients for comparison
        candidate_grad: Candidate gradients for comparison
        reference_final_state: Reference model state after full training
        model: Model after miner's training (for weight comparison)
        max_loss_difference: Maximum allowed |candidate_loss - reference_loss|
        gradient_norm_ratio_max: Encoded as 1 + max_relative_error for gradient check
        weight_relative_error_max: Max relative error for final weight comparison

    Returns:
        Tuple of (success, error_message, verification_details)
    """
    details = {
        "expected_tokens": expected_tokens,
        "candidate_tokens": candidate.total_tokens,
        "candidate_loss": candidate.final_loss,
        "reference_loss": reference.final_loss if reference else None,
        "checks_passed": [],
        "checks_failed": [],
    }

    logger.info("=" * 60)
    logger.info("VERIFICATION: Starting output verification")
    logger.info("=" * 60)

    # 1. Verify token count matches expected
    logger.info(
        f"[CHECK 1/4] Token count: expected={expected_tokens}, got={candidate.total_tokens}"
    )
    if candidate.total_tokens != expected_tokens:
        error = f"Token count mismatch: expected {expected_tokens}, got {candidate.total_tokens}"
        details["checks_failed"].append({"check": "token_count", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details
    details["checks_passed"].append("token_count")
    logger.info("[PASSED] Token count matches")

    # 2. Verify loss is reasonable and similar to reference
    logger.info(f"[CHECK 2/4] Loss validity: candidate_loss={candidate.final_loss:.6f}")
    if candidate.final_loss != candidate.final_loss:  # NaN check
        error = "Loss is NaN"
        details["checks_failed"].append({"check": "loss_validity", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details
    if candidate.final_loss <= 0:
        error = f"Loss must be positive (cross-entropy > 0): got {candidate.final_loss:.4f}"
        details["checks_failed"].append({"check": "loss_validity", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details
    if candidate.final_loss > 100:
        error = f"Loss unreasonable: {candidate.final_loss:.4f} (expected 1-10)"
        details["checks_failed"].append({"check": "loss_validity", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details

    # Compare losses - check absolute difference
    if reference is None:
        error = "No reference result available for loss comparison"
        details["checks_failed"].append({"check": "loss_comparison", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details

    loss_difference = abs(candidate.final_loss - reference.final_loss)
    details["loss_difference"] = loss_difference
    details["reference_loss"] = reference.final_loss
    logger.info(
        f"   Reference loss: {reference.final_loss:.4f}, Candidate loss: {candidate.final_loss:.4f}"
    )
    logger.info(f"   Loss difference: {loss_difference:.4f} (max allowed: {max_loss_difference})")

    if loss_difference > max_loss_difference:
        error = (
            f"Loss difference too large: candidate={candidate.final_loss:.4f}, "
            f"reference={reference.final_loss:.4f}, diff={loss_difference:.4f} > {max_loss_difference}"
        )
        details["checks_failed"].append({"check": "loss_comparison", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details
    details["checks_passed"].append("loss_validity")
    logger.info("[PASSED] Loss is valid and similar to reference")

    # 3. Gradient-based verification (captures backward pass correctness)
    logger.info("[CHECK 3/4] Gradient verification")
    grad_ok, grad_error, grad_details = _verify_gradients(
        reference_grad,
        candidate_grad,
        norm_ratio_max=gradient_norm_ratio_max,
    )
    details["gradient_verification"] = grad_details

    if not grad_ok:
        details["checks_failed"].append({"check": "gradient_verification", "error": grad_error})
        return False, grad_error, details
    details["checks_passed"].append("gradient_verification")

    # 4. Final weight verification (verifies optimizer step correctness)
    if reference_final_state is not None and model is not None:
        logger.info("[CHECK 4/4] Final weight verification")
        weight_ok, weight_error, weight_details = _verify_final_weights(
            model,
            reference_final_state,
            max_relative_error=weight_relative_error_max,
        )
        details["weight_verification"] = weight_details

        if not weight_ok:
            details["checks_failed"].append({"check": "weight_verification", "error": weight_error})
            return False, weight_error, details
        details["checks_passed"].append("weight_verification")
    else:
        logger.warning("Skipping weight verification (no reference state available)")

    logger.info("=" * 60)
    logger.info("VERIFICATION: ALL CHECKS PASSED")
    logger.info(f"   Checks passed: {details['checks_passed']}")
    logger.info("=" * 60)

    return True, None, details


class Actor:
    """Templar MFU Evaluation Actor for Affinetes (Validator-Owned).

    This Actor is owned by the validator, not the miner.
    Code is passed directly from the validator (downloaded from URL).
    """

    async def evaluate(
        self,
        task_id: int = 0,
        seed: str = "0:0:0",
        model_url: str = "",
        data_url: str = "",
        steps: int = 5,
        batch_size: int = 8,
        timeout: int = 600,
        sequence_length: int | None = None,
        data_samples: int = 10000,
        code: str = "",  # Miner's code passed directly
        # Verification settings
        max_loss_difference: float = 0.5,
        use_random_init: bool = True,
        min_trainable_params_ratio: float = 1.0,
        min_params_changed_ratio: float = 0.5,
        # Gradient verification
        gradient_norm_ratio_max: float = 1.02,
        # Weight verification
        weight_relative_error_max: float = 0.04,
        # MFU calculation
        gpu_peak_tflops: float = 312.0,
        model_params_override: int | None = None,
    ) -> dict:
        """
        Run MFU evaluation on miner's code.

        Args:
            task_id: Evaluation run identifier
            seed: Deterministic seed string (format: "block:uid:run_idx")
            model_url: HuggingFace model name
            data_url: HuggingFace dataset name
            steps: Number of training steps
            batch_size: Batch size
            timeout: Maximum seconds
            sequence_length: Sequence length
            data_samples: Number of data samples
            code: Miner's train.py code (passed directly from validator)
            max_loss_difference: Max allowed |candidate_loss - reference_loss|
            use_random_init: Use random weights
            min_trainable_params_ratio: Min % params that must be trainable
            min_params_changed_ratio: Min % params that must change
            gradient_norm_ratio_max: Encoded as 1 + max_relative_error (e.g., 1.04 = 4%)
            weight_relative_error_max: Max relative error for final weight check (e.g., 0.04 = 4%)
            gpu_peak_tflops: GPU peak TFLOPS for MFU calculation
            model_params_override: Override model param count (None = auto-detect)

        Returns:
            Dict with: mfu, tps, total_tokens, wall_time_seconds, success, error, code
        """
        # Validate code
        if not code:
            return {
                "task_id": task_id,
                "mfu": 0.0,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": "No code provided",
                "seed": seed,
            }

        # Validate model/data
        if not model_url or not data_url:
            return {
                "task_id": task_id,
                "mfu": 0.0,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": "Missing model_url or data_url",
                "seed": seed,
            }

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + timeout
        seed_value = abs(hash(seed)) % (2**32)
        _set_deterministic(seed_value)

        # Validate code structure
        code_ok, code_error = _validate_code_structure(code)
        if not code_ok:
            return {
                "task_id": task_id,
                "mfu": 0.0,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": code_error,
                "seed": seed,
                "code": code,
            }

        # Write code to temp file for loading as module
        train_path = CACHE_DIR / "miner_train.py"
        try:
            train_path.write_text(code)
        except Exception as e:
            return {
                "task_id": task_id,
                "mfu": 0.0,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": f"Failed to write train.py: {e}",
                "seed": seed,
            }

        if time.monotonic() > deadline:
            return {
                "task_id": task_id,
                "mfu": 0.0,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": "Timeout before evaluation",
                "seed": seed,
            }

        try:
            # Load miner's module
            miner_module = _load_miner_module(train_path)

            if not hasattr(miner_module, "inner_steps"):
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": 0.0,
                    "success": False,
                    "error": "train.py missing inner_steps function",
                    "seed": seed,
                    "code": code,
                }

            # Load model and data
            _log_vram("before-model-load")
            model = _get_cached_model(model_url, use_random_init=use_random_init)
            model_params = model_params_override or _count_model_params(model)
            logger.info(f"Model loaded: {model_params:,} parameters, random_init={use_random_init}")

            data = _load_hf_dataset(
                dataset_name=data_url,
                model_name=model_url,
                num_samples=data_samples,
                sequence_length=sequence_length or EVAL_SEQUENCE_LENGTH,
                validator_seed=seed,  # Unpredictable sampling
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            seq_len = sequence_length or EVAL_SEQUENCE_LENGTH

            # Run reference implementation (captures gradients for verification)
            data_iter_ref = _create_data_iterator(data, batch_size, seq_len)
            optimizer_ref = _create_optimizer(model)

            initial_state = _CACHE.get("initial_state")
            if initial_state is None:
                initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                _CACHE["initial_state"] = initial_state

            reference, reference_grad = _run_reference(
                model, data_iter_ref, optimizer_ref, steps, device, capture_final_gradients=True
            )

            # Capture reference final weights for verification
            reference_final_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            logger.info("Captured reference final model state for weight verification")
            _log_vram("after-reference-run")

            # Free reference optimizer VRAM before running miner code
            del optimizer_ref
            del data_iter_ref
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Reset model for miner's code
            model.load_state_dict(initial_state)
            warmup_steps = 2
            data_iter_warmup = _create_data_iterator(data, batch_size, seq_len)
            optimizer_warmup = _create_optimizer(model)

            logger.info(f"Running {warmup_steps} warmup step(s) to check for basic errors...")
            try:
                warmup_result = miner_module.inner_steps(
                    model=model,
                    data_iterator=data_iter_warmup,
                    optimizer=optimizer_warmup,
                    num_steps=warmup_steps,
                    device=device,
                )

                # Quick validation of warmup result
                if warmup_result is None:
                    return {
                        "task_id": task_id,
                        "mfu": 0.0,
                        "tps": 0.0,
                        "total_tokens": 0,
                        "wall_time_seconds": 0.0,
                        "success": False,
                        "error": "inner_steps returned None (warmup check)",
                        "seed": seed,
                        "code": code,
                    }

                # Check required attributes exist
                if not hasattr(warmup_result, "final_logits"):
                    return {
                        "task_id": task_id,
                        "mfu": 0.0,
                        "tps": 0.0,
                        "total_tokens": 0,
                        "wall_time_seconds": 0.0,
                        "success": False,
                        "error": "inner_steps result missing 'final_logits' attribute",
                        "seed": seed,
                        "code": code,
                    }

                # Check logits are present and valid shape
                logits = warmup_result.final_logits
                if logits is None:
                    return {
                        "task_id": task_id,
                        "mfu": 0.0,
                        "tps": 0.0,
                        "total_tokens": 0,
                        "wall_time_seconds": 0.0,
                        "success": False,
                        "error": "final_logits is None - must return actual logits for verification",
                        "error_code": "missing_logits",
                        "seed": seed,
                        "code": code,
                    }
                if len(logits.shape) != 3:
                    return {
                        "task_id": task_id,
                        "mfu": 0.0,
                        "tps": 0.0,
                        "total_tokens": 0,
                        "wall_time_seconds": 0.0,
                        "success": False,
                        "error": f"Logits shape mismatch: expected 3D (batch, seq, vocab), got {logits.shape}",
                        "error_code": "invalid_logits_shape",
                        "seed": seed,
                        "code": code,
                    }

                logger.info("Warmup passed - proceeding with verification checks")

                # Check trainable params AFTER warmup (miner code may modify requires_grad)
                trainable_ok, trainable_error, trainable_details = _verify_trainable_params(
                    model, min_trainable_params_ratio
                )
                if not trainable_ok:
                    return {
                        "task_id": task_id,
                        "mfu": 0.0,
                        "tps": 0.0,
                        "total_tokens": 0,
                        "wall_time_seconds": 0.0,
                        "success": False,
                        "error": trainable_error,
                        "error_code": "insufficient_trainable_params",
                        "seed": seed,
                        "code": code,
                        "diagnostics": {"trainable_params": trainable_details},
                    }

            except Exception as e:
                # Early termination - don't waste time on broken code
                error_msg = f"Early termination (warmup failed): {type(e).__name__}: {str(e)}"
                logger.error(error_msg)
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": 0.0,
                    "success": False,
                    "error": error_msg,
                    "seed": seed,
                    "code": code,
                }

            # Reset model state after warmup and free warmup optimizer VRAM
            model.load_state_dict(initial_state)
            del optimizer_warmup
            del data_iter_warmup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Full timed evaluation with gradient capture
            _log_vram("before-timed-eval")
            data_iter_miner = _create_data_iterator(data, batch_size, seq_len)
            base_optimizer = _create_optimizer(model)
            optimizer_miner = GradientCapturingOptimizer(base_optimizer, model, num_steps=steps)

            # Enforce backend settings to match reference run.
            # Miner warmup may have changed these for a speed boost;
            # reset them so the timed eval uses the same settings as reference.
            _enforce_backend_state()

            _cuda_synchronize()
            start = _perf_counter()

            miner_result = miner_module.inner_steps(
                model=model,
                data_iterator=data_iter_miner,
                optimizer=optimizer_miner,
                num_steps=steps,
                device=device,
            )

            _cuda_synchronize()
            wall_time = _perf_counter() - start
            logger.info(f"Timing: wall_time={wall_time:.2f}s (used for MFU/TPS)")

            # Convert GPU gradient snapshot to CPU (slow, but OUTSIDE the timer)
            optimizer_miner.finalize_gradients()

            # Verify backend settings were not tampered with during timed eval
            _backend_violations = _check_backend_state()
            if _backend_violations:
                logger.warning(f"Backend settings changed during eval: {_backend_violations}")
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": "forbidden pattern detected",
                    "error_code": "execution_failed",
                    "seed": seed,
                    "code": code,
                }

            # Verify optimizer wrapper integrity
            if type(optimizer_miner) is not GradientCapturingOptimizer:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": "Optimizer integrity check failed",
                    "error_code": "execution_failed",
                    "seed": seed,
                    "code": code,
                }

            # Get gradients captured from the final step (inside miner's code)
            candidate_grad = optimizer_miner.captured_gradients

            if candidate_grad is None:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": "No gradients captured - miner may not be calling optimizer.step()",
                    "error_code": "no_gradients_captured",
                    "seed": seed,
                    "code": code,
                }

            # Validate miner's returned result
            ok, error, parsed = _validate_return_type(miner_result)
            if not ok or parsed is None:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": error,
                    "seed": seed,
                    "code": code,
                }

            # Sequence length verification
            expected_seq_len = seq_len - 1  # causal LM uses batch[:, :-1] as input

            if parsed.final_logits is None:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": "final_logits is None - must return actual logits for verification",
                    "error_code": "missing_logits",
                    "seed": seed,
                    "code": code,
                }

            # Verify logits shape is 3D (batch, seq, vocab)
            if len(parsed.final_logits.shape) != 3:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": f"Logits shape invalid: expected 3D (batch, seq, vocab), got shape {parsed.final_logits.shape}",
                    "error_code": "invalid_logits_shape",
                    "seed": seed,
                    "code": code,
                }

            logits_seq_len = parsed.final_logits.shape[1]

            # Verify sequence length matches expected
            if logits_seq_len != expected_seq_len:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": f"Sequence length mismatch (possible truncation): expected {expected_seq_len}, got {logits_seq_len}",
                    "error_code": "sequence_truncation",
                    "seed": seed,
                    "code": code,
                }
            logger.info(f"[PASSED] Sequence length check: {logits_seq_len} == {expected_seq_len}")

            # Verify sufficient parameters changed during training
            params_ok, params_error, params_details = _verify_params_changed(
                model, initial_state, min_params_changed_ratio
            )
            if not params_ok:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": params_error,
                    "error_code": "insufficient_params_changed",
                    "seed": seed,
                    "code": code,
                    "diagnostics": {"params_changed": params_details},
                }

            # Calculate expected tokens
            expected_tokens = batch_size * seq_len * steps

            # Verify outputs using gradient-based AND weight-based verification
            verified, verify_error, verify_details = _verify_outputs(
                reference,
                parsed,
                expected_tokens,
                reference_grad=reference_grad,
                candidate_grad=candidate_grad,
                reference_final_state=reference_final_state,
                model=model,
                max_loss_difference=max_loss_difference,
                gradient_norm_ratio_max=gradient_norm_ratio_max,
                weight_relative_error_max=weight_relative_error_max,
            )

            total_tokens_int = expected_tokens  # Use validator-computed token count
            tps = float(total_tokens_int) / max(wall_time, 1e-6)
            mfu = _calculate_mfu(total_tokens_int, wall_time, model_params, gpu_peak_tflops)

            # Diagnostics
            diagnostics = {
                "verification": verify_details,
                "reference_loss": reference.final_loss,
                "candidate_loss": parsed.final_loss,
                "expected_tokens": expected_tokens,
                "actual_tokens": parsed.total_tokens,
                "model_params": model_params,
                "gpu_peak_tflops": gpu_peak_tflops,
                "trainable_params": trainable_details,
                "params_changed": params_details,
            }

            # Extract error_code from verification details if present
            error_code = None
            if not verified and verify_details:
                # Check gradient_verification sub-details
                grad_details = verify_details.get("gradient_verification", {})
                error_code = grad_details.get("error_code")
                # Check weight_verification sub-details
                if not error_code:
                    weight_details = verify_details.get("weight_verification", {})
                    error_code = weight_details.get("error_code")
                # Check other verification failures
                if not error_code:
                    for check in verify_details.get("checks_failed", []):
                        if check.get("check") == "token_count":
                            error_code = "token_count_mismatch"
                        elif check.get("check") == "loss_comparison":
                            error_code = "loss_mismatch"
                        elif check.get("check") == "weight_verification":
                            error_code = "weight_mismatch"

            return {
                "task_id": task_id,
                "mfu": mfu if verified else 0.0,
                "tps": tps if verified else 0.0,
                "total_tokens": total_tokens_int if verified else 0,
                "wall_time_seconds": wall_time,
                "success": verified,
                "error": verify_error,
                "error_code": error_code,
                "seed": seed,
                "diagnostics": diagnostics,
                "code": code,
            }

        except Exception as e:
            return {
                "task_id": task_id,
                "mfu": 0.0,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                "seed": seed,
                "code": code,
            }

        finally:
            # Free model gradients to release VRAM held by .grad tensors
            try:
                model.zero_grad(set_to_none=True)
            except Exception:
                pass

            # Explicitly delete large objects to free VRAM/RAM before GC.
            # Without this, local references keep optimizer states (~2x model
            # VRAM), data tensors, gradient copies, and reference state alive,
            # preventing gc.collect() + empty_cache() from reclaiming memory.
            # noinspection PyUnusedLocal
            optimizer_miner = base_optimizer = None  # type: ignore[assignment]
            data_iter_miner = data = None  # type: ignore[assignment]
            reference_final_state = candidate_grad = None  # type: ignore[assignment]
            miner_result = parsed = miner_module = None  # type: ignore[assignment]
            reference = reference_grad = None  # type: ignore[assignment]

            # Reset torch state (clears compile caches, removes miner module)
            _reset_torch_state()

            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _log_vram("after-cleanup")


# =============================================================================
# FastAPI HTTP Server (for Basilica custom Docker deployment)
# =============================================================================

app = FastAPI(title="Templar MFU Evaluation", version="2.0.0")

# Global actor instance (reused for efficiency)
_actor: Actor | None = None


def get_actor() -> Actor:
    """Get or create singleton Actor instance."""
    global _actor
    if _actor is None:
        _actor = Actor()
    return _actor


class EvaluateRequest(BaseModel):
    """Request body for /evaluate endpoint."""

    task_id: int = 0
    seed: str = "0:0:0"
    model_url: str
    data_url: str
    steps: int = 5
    batch_size: int = 8
    timeout: int = 600
    sequence_length: int | None = None
    data_samples: int = 10000
    code: str  # Miner's train.py code
    # Verification settings
    max_loss_difference: float = 0.5
    use_random_init: bool = True
    min_trainable_params_ratio: float = 1.0
    min_params_changed_ratio: float = 0.5
    # Gradient verification
    gradient_norm_ratio_max: float = 1.02
    # Weight verification
    weight_relative_error_max: float = 0.04
    # MFU calculation
    gpu_peak_tflops: float = 312.0


class EvaluateResponse(BaseModel):
    """Response body from /evaluate endpoint."""

    task_id: int
    mfu: float  # Model FLOPs Utilization (primary metric)
    tps: float  # Tokens per second (secondary metric)
    total_tokens: int
    wall_time_seconds: float
    success: bool
    error: str | None = None
    error_code: str | None = None
    seed: str
    diagnostics: dict = {}


@app.get("/health")
async def health():
    """Health check endpoint for Basilica."""
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    """Evaluate miner's train.py code and return MFU score.

    This endpoint is called by the validator via Basilica.
    """
    actor = get_actor()

    result = await actor.evaluate(
        task_id=request.task_id,
        seed=request.seed,
        model_url=request.model_url,
        data_url=request.data_url,
        steps=request.steps,
        batch_size=request.batch_size,
        timeout=request.timeout,
        sequence_length=request.sequence_length,
        data_samples=request.data_samples,
        code=request.code,
        max_loss_difference=request.max_loss_difference,
        use_random_init=request.use_random_init,
        min_trainable_params_ratio=request.min_trainable_params_ratio,
        min_params_changed_ratio=request.min_params_changed_ratio,
        gradient_norm_ratio_max=request.gradient_norm_ratio_max,
        weight_relative_error_max=request.weight_relative_error_max,
        gpu_peak_tflops=request.gpu_peak_tflops,
    )

    return EvaluateResponse(
        task_id=result.get("task_id", request.task_id),
        mfu=result.get("mfu", 0.0),
        tps=result.get("tps", 0.0),
        total_tokens=result.get("total_tokens", 0),
        wall_time_seconds=result.get("wall_time_seconds", 0.0),
        success=result.get("success", False),
        error=result.get("error"),
        error_code=result.get("error_code"),
        seed=result.get("seed", request.seed),
        diagnostics=result.get("diagnostics", {}),
    )


# Entry point when running directly (for local testing)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
