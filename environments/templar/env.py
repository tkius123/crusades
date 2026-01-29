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
4. This Actor runs benchmark and returns TPS
"""

import ast
import gc
import hashlib
import importlib.util
import logging
import os
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


# Global cache for model (data is NOT cached for validators)
_CACHE = {
    "model": None,
    "model_path": None,
    "initial_state": None,
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


def _load_hf_dataset(
    dataset_name: str,
    model_name: str,
    num_samples: int = 10000,
    sequence_length: int = 1024,
    split: str = "train",
    validator_seed: str | None = None,
) -> torch.Tensor:
    """Load and tokenize dataset from HuggingFace or local cache.

    SECURITY: Validators use unpredictable seeds so miners can't pre-compute
    which samples will be used for evaluation.

    When running with --network none, uses pre-cached dataset from Docker image.
    The validator seed is used to shuffle the cached data unpredictably.

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
    import random

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

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check for cached dataset (enables --network none operation)
    cached_path = os.getenv("CACHED_DATASET_PATH", "/home/appuser/.cache/templar/dataset.json")

    if Path(cached_path).exists():
        # Load from cache and shuffle with validator seed
        logger.info(f"Using cached dataset: {cached_path}")
        with open(cached_path) as f:
            all_samples = json.load(f)

        # Shuffle with validator seed for unpredictability
        rng = random.Random(actual_seed)
        rng.shuffle(all_samples)

        tokens_list = []
        for text in all_samples[:num_samples]:
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

        data = torch.stack(tokens_list)
        logger.info(f"Loaded cached data: shape={data.shape}, seed={actual_seed}")
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


def _validate_code_structure(code: str) -> tuple[bool, str | None]:
    """Validate that train.py has correct structure."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"Syntax error at line {exc.lineno}: {exc.msg}"

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


def _load_model(model_path: str):
    """Load model from HuggingFace."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


def _get_cached_model(model_path: str):
    """Get model from cache or load it."""
    cached = _CACHE.get("model")
    cached_path = _CACHE.get("model_path")

    if cached is not None and cached_path == model_path:
        if _CACHE.get("initial_state"):
            cached.load_state_dict(_CACHE["initial_state"])
        return cached

    model = _load_model(model_path)
    _CACHE["model"] = model
    _CACHE["model_path"] = model_path
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
    """Create standard AdamW optimizer."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )


def _run_reference(
    model: torch.nn.Module,
    data_iterator: Iterator[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    device: torch.device,
) -> InnerStepsResult:
    """Run reference implementation for comparison."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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


def _verify_outputs(
    reference: InnerStepsResult,
    candidate: InnerStepsResult,
    expected_tokens: int,
    output_tolerance: float = 0.05,
    loss_ratio_min: float = 0.8,
    loss_ratio_max: float = 1.2,
) -> tuple[bool, str | None, dict]:
    """Verify candidate outputs match reference within tolerance.

    Verification checks:
    1. Token count matches expected (miner processed correct amount of data)
    2. Loss is reasonable (not NaN/Inf) and matches reference within allowed ratio
    3. Logits are valid tensors (not NaN/Inf)
    4. Logits match reference within tolerance (prevents cheating)

    Args:
        reference: Reference implementation results
        candidate: Miner's implementation results
        expected_tokens: Expected token count (batch_size * seq_len * steps)
        output_tolerance: Maximum allowed relative difference for logits (default 5%)
        loss_ratio_min: Minimum allowed loss ratio candidate/reference (default 0.8)
        loss_ratio_max: Maximum allowed loss ratio candidate/reference (default 1.2)

    Returns:
        Tuple of (success, error_message, verification_details)
    """
    # Collect detailed verification info for logging
    details = {
        "expected_tokens": expected_tokens,
        "candidate_tokens": candidate.total_tokens,
        "candidate_loss": candidate.final_loss,
        "reference_loss": reference.final_loss if reference else None,
        "output_tolerance": output_tolerance,
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

    # 2. Verify loss is reasonable (typical loss range is 1.5-3 for trained LLMs)
    logger.info(f"[CHECK 2/4] Loss validity: candidate_loss={candidate.final_loss:.6f}")
    if candidate.final_loss != candidate.final_loss:  # NaN check
        error = "Loss is NaN"
        details["checks_failed"].append({"check": "loss_validity", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details
    if abs(candidate.final_loss) > 100:
        error = f"Loss unreasonable: {candidate.final_loss:.4f} (expected 1-10)"
        details["checks_failed"].append({"check": "loss_validity", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details
    details["checks_passed"].append("loss_validity")
    logger.info("[PASSED] Loss is valid (not NaN/Inf, in reasonable range)")

    # 3. Verify logits are valid
    logger.info("[CHECK 3/4] Logits validity")
    cand_logits = candidate.final_logits
    if cand_logits is None:
        error = "No logits returned"
        details["checks_failed"].append({"check": "logits_validity", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details

    if isinstance(cand_logits, str):
        cand_logits = torch.load(cand_logits, weights_only=True)

    details["candidate_logits_shape"] = list(cand_logits.shape)
    logger.info(f"   Candidate logits shape: {cand_logits.shape}")

    if torch.isnan(cand_logits).any():
        nan_count = torch.isnan(cand_logits).sum().item()
        error = f"Logits contain {nan_count} NaN values"
        details["checks_failed"].append(
            {"check": "logits_validity", "error": error, "nan_count": nan_count}
        )
        logger.error(f"[FAILED] {error}")
        return False, error, details
    if torch.isinf(cand_logits).any():
        inf_count = torch.isinf(cand_logits).sum().item()
        error = f"Logits contain {inf_count} Inf values"
        details["checks_failed"].append(
            {"check": "logits_validity", "error": error, "inf_count": inf_count}
        )
        logger.error(f"[FAILED] {error}")
        return False, error, details
    details["checks_passed"].append("logits_validity")
    logger.info("[PASSED] Logits are valid (no NaN/Inf)")

    # 4. Compare with reference
    logger.info("[CHECK 4/4] Reference comparison")

    # 4a. Compare losses - they should be within 20% of each other
    if reference is not None and reference.final_loss > 0:
        loss_ratio = candidate.final_loss / reference.final_loss
        details["loss_ratio"] = loss_ratio
        logger.info(
            f"   Loss comparison: candidate={candidate.final_loss:.6f}, reference={reference.final_loss:.6f}"
        )
        logger.info(f"   Loss ratio: {loss_ratio:.4f} (allowed: {loss_ratio_min}-{loss_ratio_max})")

        if loss_ratio < loss_ratio_min or loss_ratio > loss_ratio_max:
            error = (
                f"Loss mismatch: candidate={candidate.final_loss:.4f}, "
                f"reference={reference.final_loss:.4f}, ratio={loss_ratio:.2f} (allowed: {loss_ratio_min}-{loss_ratio_max})"
            )
            details["checks_failed"].append(
                {"check": "loss_comparison", "error": error, "loss_ratio": loss_ratio}
            )
            logger.error(f"[FAILED] {error}")
            return False, error, details
        details["checks_passed"].append("loss_comparison")
        logger.info("[PASSED] Loss matches reference within 20%")

    # 4b. Compare logits against reference
    if reference is not None and reference.final_logits is not None:
        ref_logits = reference.final_logits
        if isinstance(ref_logits, str):
            ref_logits = torch.load(ref_logits, weights_only=True)

        details["reference_logits_shape"] = list(ref_logits.shape)
        logger.info(f"   Reference logits shape: {ref_logits.shape}")

        # Ensure same device for comparison
        if cand_logits.device != ref_logits.device:
            cand_logits = cand_logits.to(ref_logits.device)

        # Check shape match
        if cand_logits.shape != ref_logits.shape:
            error = f"Logits shape mismatch: candidate={cand_logits.shape}, reference={ref_logits.shape}"
            details["checks_failed"].append({"check": "logits_shape", "error": error})
            logger.error(f"[FAILED] {error}")
            return False, error, details
        details["checks_passed"].append("logits_shape")

        # Calculate relative difference using mean absolute error
        # Normalize by the scale of the reference logits
        abs_diff = torch.abs(cand_logits - ref_logits)
        ref_scale = torch.abs(ref_logits).mean() + 1e-8  # Avoid division by zero
        relative_diff = (abs_diff.mean() / ref_scale).item()
        max_diff = abs_diff.max().item()

        details["logits_relative_diff"] = relative_diff
        details["logits_max_diff"] = max_diff
        details["logits_ref_scale"] = ref_scale.item()

        logger.info("   Logits comparison:")
        logger.info(f"      Reference scale (mean abs): {ref_scale.item():.6f}")
        logger.info(f"      Mean absolute difference: {abs_diff.mean().item():.6f}")
        logger.info(f"      Max absolute difference: {max_diff:.6f}")
        logger.info(f"      Relative difference: {relative_diff:.6f}")
        logger.info(f"      Tolerance: {output_tolerance}")

        if relative_diff > output_tolerance:
            error = (
                f"Logits mismatch: relative_diff={relative_diff:.6f} "
                f"exceeds tolerance={output_tolerance} (max_diff={max_diff:.6f})"
            )
            details["checks_failed"].append(
                {
                    "check": "logits_comparison",
                    "error": error,
                    "relative_diff": relative_diff,
                    "max_diff": max_diff,
                }
            )
            logger.error(f"[FAILED] {error}")
            return False, error, details

        details["checks_passed"].append("logits_comparison")
        logger.info(
            f"[PASSED] Logits match reference (relative_diff={relative_diff:.6f} <= {output_tolerance})"
        )

    logger.info("=" * 60)
    logger.info("VERIFICATION: ALL CHECKS PASSED")
    logger.info(f"   Checks passed: {details['checks_passed']}")
    logger.info("=" * 60)

    return True, None, details


class Actor:
    """Templar TPS Evaluation Actor for Affinetes (Validator-Owned).

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
        output_tolerance: float = 0.05,  # Tolerance for logits comparison
        loss_ratio_min: float = 0.8,  # Minimum allowed loss ratio
        loss_ratio_max: float = 1.2,  # Maximum allowed loss ratio
    ) -> dict:
        """
        Run TPS evaluation on miner's code.

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
            output_tolerance: Maximum allowed relative difference for logits verification
            loss_ratio_min: Minimum allowed loss ratio (candidate/reference)
            loss_ratio_max: Maximum allowed loss ratio (candidate/reference)

        Returns:
            Dict with: tps, total_tokens, wall_time_seconds, success, error, code
        """
        # Validate code
        if not code:
            return {
                "task_id": task_id,
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
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": 0.0,
                    "success": False,
                    "error": "train.py missing inner_steps function",
                    "seed": seed,
                    "code": code,
                }

            # Load model and data
            model = _get_cached_model(model_url)
            data = _load_hf_dataset(
                dataset_name=data_url,
                model_name=model_url,
                num_samples=data_samples,
                sequence_length=sequence_length or EVAL_SEQUENCE_LENGTH,
                validator_seed=seed,  # Unpredictable sampling
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            seq_len = sequence_length or EVAL_SEQUENCE_LENGTH

            # Run reference implementation
            data_iter_ref = _create_data_iterator(data, batch_size, seq_len)
            optimizer_ref = _create_optimizer(model)

            initial_state = _CACHE.get("initial_state")
            if initial_state is None:
                initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                _CACHE["initial_state"] = initial_state

            reference = _run_reference(model, data_iter_ref, optimizer_ref, steps, device)

            # Reset model for miner's code
            model.load_state_dict(initial_state)

            # =================================================================
            # EARLY TERMINATION - Run 1 warmup step to catch basic errors
            # =================================================================
            # This catches shape mismatches, missing attributes, etc. before
            # running the full evaluation (saves GPU time on broken code)
            data_iter_warmup = _create_data_iterator(data, batch_size, seq_len)
            optimizer_warmup = _create_optimizer(model)

            logger.info("Running warmup step to check for basic errors...")
            try:
                warmup_result = miner_module.inner_steps(
                    model=model,
                    data_iterator=data_iter_warmup,
                    optimizer=optimizer_warmup,
                    num_steps=1,  # Just 1 step for validation
                    device=device,
                )

                # Quick validation of warmup result
                if warmup_result is None:
                    return {
                        "task_id": task_id,
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
                        "tps": 0.0,
                        "total_tokens": 0,
                        "wall_time_seconds": 0.0,
                        "success": False,
                        "error": "inner_steps result missing 'final_logits' attribute",
                        "seed": seed,
                        "code": code,
                    }

                # Check logits shape (should be 3D: batch, seq, vocab)
                logits = warmup_result.final_logits
                if logits is not None and len(logits.shape) != 3:
                    return {
                        "task_id": task_id,
                        "tps": 0.0,
                        "total_tokens": 0,
                        "wall_time_seconds": 0.0,
                        "success": False,
                        "error": f"Logits shape mismatch: expected 3D (batch, seq, vocab), got {logits.shape}",
                        "seed": seed,
                        "code": code,
                    }

                logger.info("Warmup passed - proceeding with full evaluation")

            except Exception as e:
                # Early termination - don't waste time on broken code
                error_msg = f"Early termination (warmup failed): {type(e).__name__}: {str(e)}"
                logger.error(error_msg)
                return {
                    "task_id": task_id,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": 0.0,
                    "success": False,
                    "error": error_msg,
                    "seed": seed,
                    "code": code,
                }

            # Reset model state after warmup
            model.load_state_dict(initial_state)

            # =================================================================
            # FULL EVALUATION - Run actual timed evaluation
            # =================================================================
            data_iter_miner = _create_data_iterator(data, batch_size, seq_len)
            optimizer_miner = _create_optimizer(model)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()
            miner_result = miner_module.inner_steps(
                model=model,
                data_iterator=data_iter_miner,
                optimizer=optimizer_miner,
                num_steps=steps,
                device=device,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            wall_time = time.perf_counter() - start

            # Validate return type
            ok, error, parsed = _validate_return_type(miner_result)
            if not ok or parsed is None:
                return {
                    "task_id": task_id,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": error,
                    "seed": seed,
                    "code": code,
                }

            # Calculate expected tokens
            expected_tokens = batch_size * seq_len * steps

            # Verify outputs against reference (correctness check with tolerance)
            verified, verify_error, verify_details = _verify_outputs(
                reference, parsed, expected_tokens, output_tolerance, loss_ratio_min, loss_ratio_max
            )

            # Diagnostics (include verification details)
            diagnostics = {
                "verification": verify_details,
                "reference_loss": reference.final_loss,
                "candidate_loss": parsed.final_loss,
                "expected_tokens": expected_tokens,
                "actual_tokens": parsed.total_tokens,
            }

            total_tokens = int(parsed.total_tokens)
            tps = float(total_tokens) / max(wall_time, 1e-6)

            return {
                "task_id": task_id,
                "tps": tps if verified else 0.0,
                "total_tokens": total_tokens if verified else 0,
                "wall_time_seconds": wall_time,
                "success": verified,
                "error": verify_error,
                "seed": seed,
                "diagnostics": diagnostics,
                "code": code,
            }

        except Exception as e:
            return {
                "task_id": task_id,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                "seed": seed,
                "code": code,
            }

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# =============================================================================
# FastAPI HTTP Server (for Basilica custom Docker deployment)
# =============================================================================

app = FastAPI(title="Templar TPS Evaluation", version="1.0.0")

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
    output_tolerance: float = 0.05  # Tolerance for logits comparison
    loss_ratio_min: float = 0.8  # Minimum allowed loss ratio
    loss_ratio_max: float = 1.2  # Maximum allowed loss ratio


class EvaluateResponse(BaseModel):
    """Response body from /evaluate endpoint."""

    task_id: int
    tps: float
    total_tokens: int
    wall_time_seconds: float
    success: bool
    error: str | None = None
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
    """Evaluate miner's train.py code and return TPS score.

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
        output_tolerance=request.output_tolerance,
        loss_ratio_min=request.loss_ratio_min,
        loss_ratio_max=request.loss_ratio_max,
    )

    return EvaluateResponse(
        task_id=result.get("task_id", request.task_id),
        tps=result.get("tps", 0.0),
        total_tokens=result.get("total_tokens", 0),
        wall_time_seconds=result.get("wall_time_seconds", 0.0),
        success=result.get("success", False),
        error=result.get("error"),
        seed=result.get("seed", request.seed),
        diagnostics=result.get("diagnostics", {}),
    )


# Entry point when running directly (for local testing)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
