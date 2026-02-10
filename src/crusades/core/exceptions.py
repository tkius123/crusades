"""Custom exceptions for templar-crusades."""

from enum import StrEnum


class EvaluationErrorCode(StrEnum):
    """Structured error codes for evaluation failures.

    Use these codes instead of string matching for robust error handling.
    """

    # Code validation errors
    NO_CODE = "no_code"
    SYNTAX_ERROR = "syntax_error"
    MISSING_INNER_STEPS = "missing_inner_steps"
    INVALID_RETURN_TYPE = "invalid_return_type"

    # Verification failures
    INSUFFICIENT_TRAINABLE_PARAMS = "insufficient_trainable_params"
    INSUFFICIENT_PARAMS_CHANGED = "insufficient_params_changed"
    GRADIENT_COVERAGE_FAILED = "gradient_coverage_failed"
    GRADIENT_RELATIVE_ERROR_FAILED = "gradient_relative_error_failed"
    LOSS_MISMATCH = "loss_mismatch"
    TOKEN_COUNT_MISMATCH = "token_count_mismatch"
    NO_GRADIENTS_CAPTURED = "no_gradients_captured"
    MISSING_LOGITS = "missing_logits"
    INVALID_LOGITS_SHAPE = "invalid_logits_shape"
    SEQUENCE_TRUNCATION = "sequence_truncation"
    WEIGHT_MISMATCH = "weight_mismatch"

    # Runtime errors
    TIMEOUT = "timeout"
    OUT_OF_MEMORY = "out_of_memory"
    WARMUP_FAILED = "warmup_failed"
    EXECUTION_FAILED = "execution_failed"

    # Infrastructure errors
    MODEL_LOAD_FAILED = "model_load_failed"
    DATA_LOAD_FAILED = "data_load_failed"
    DOCKER_FAILED = "docker_failed"

    # Unknown
    UNKNOWN = "unknown"

    @classmethod
    def is_verification_failure(cls, code: "EvaluationErrorCode") -> bool:
        """Check if error code indicates a verification failure."""
        return code in {
            cls.INSUFFICIENT_TRAINABLE_PARAMS,
            cls.INSUFFICIENT_PARAMS_CHANGED,
            cls.GRADIENT_COVERAGE_FAILED,
            cls.GRADIENT_RELATIVE_ERROR_FAILED,
            cls.LOSS_MISMATCH,
            cls.TOKEN_COUNT_MISMATCH,
            cls.NO_GRADIENTS_CAPTURED,
            cls.MISSING_LOGITS,
            cls.INVALID_LOGITS_SHAPE,
            cls.SEQUENCE_TRUNCATION,
            cls.WEIGHT_MISMATCH,
        }

    @classmethod
    def is_fatal(cls, code: "EvaluationErrorCode") -> bool:
        """Check if error is fatal/deterministic (no point retrying).

        Only includes checks that are determined by the miner's CODE, not
        by the random seed/data. Data-dependent checks (gradient error,
        params changed, loss) can vary between runs and should be retried.
        """
        return code in {
            # Code validation errors - code won't magically fix itself
            cls.NO_CODE,
            cls.SYNTAX_ERROR,
            cls.MISSING_INNER_STEPS,
            cls.INVALID_RETURN_TYPE,
            # Code-level checks - deterministic regardless of data
            cls.INSUFFICIENT_TRAINABLE_PARAMS,  # Code freezes layers or not
            cls.NO_GRADIENTS_CAPTURED,  # Code calls optimizer.step() or not
            cls.MISSING_LOGITS,  # Code returns None or not
            cls.INVALID_LOGITS_SHAPE,  # Code returns wrong shape or not
            cls.SEQUENCE_TRUNCATION,  # Code truncates or not
            # NOT fatal (data-dependent, can vary between seeds):
            # - INSUFFICIENT_PARAMS_CHANGED (borderline with few steps)
            # - GRADIENT_RELATIVE_ERROR_FAILED (varies with data)
            # - GRADIENT_COVERAGE_FAILED (borderline)
            # - LOSS_MISMATCH (varies with data)
            # - TOKEN_COUNT_MISMATCH (should be deterministic but edge cases)
        }

    @classmethod
    def is_miner_fault(cls, code: "EvaluationErrorCode") -> bool:
        """Check if error is likely the miner's fault (vs infrastructure)."""
        return code not in {
            cls.MODEL_LOAD_FAILED,
            cls.DATA_LOAD_FAILED,
            cls.DOCKER_FAILED,
            cls.TIMEOUT,  # Could be either
        }


class CrusadesError(Exception):
    """Base exception for all crusades errors."""

    pass


class SandboxError(CrusadesError):
    """Error during sandbox/Docker execution."""

    pass


class EvaluationError(CrusadesError):
    """Error during evaluation with structured error code."""

    def __init__(self, message: str, code: EvaluationErrorCode = EvaluationErrorCode.UNKNOWN):
        super().__init__(message)
        self.code = code
        self.message = message


class StorageError(CrusadesError):
    """Error accessing storage (database)."""

    pass
