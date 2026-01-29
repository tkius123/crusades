"""Protocol definitions for templar-crusades.

These protocols define contracts between components, enabling loose coupling
and easy testing/mocking.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class SubmissionStatus(StrEnum):
    """Status of a code submission."""

    # In-progress statuses (hidden from dashboard for security)
    PENDING = "pending"  # Just submitted, awaiting validation
    VALIDATING = "validating"  # Being validated (syntax, imports, functions)
    EVALUATING = "evaluating"  # Passed validation, being evaluated in sandbox

    # Final statuses (shown in recent submissions)
    FINISHED = "finished"  #  Evaluation complete, has final TPS score
    FAILED_VALIDATION = (
        "failed_validation"  #  Code validation failed (syntax, imports, missing function)
    )
    FAILED_EVALUATION = (
        "failed_evaluation"  #  Sandbox ran but verification failed (logits mismatch, timeout)
    )
    ERROR = "error"  #  Unexpected system error


@runtime_checkable
class Submission(Protocol):
    """Protocol for submission data."""

    @property
    def submission_id(self) -> str: ...

    @property
    def miner_hotkey(self) -> str: ...

    @property
    def miner_uid(self) -> int: ...

    @property
    def code_hash(self) -> str: ...

    @property
    def status(self) -> SubmissionStatus: ...

    @property
    def created_at(self) -> datetime: ...

    @property
    def final_score(self) -> float | None: ...


@runtime_checkable
class EvaluationResult(Protocol):
    """Protocol for evaluation result data."""

    @property
    def submission_id(self) -> str: ...

    @property
    def evaluator_hotkey(self) -> str: ...

    @property
    def tokens_per_second(self) -> float: ...

    @property
    def total_tokens(self) -> int: ...

    @property
    def wall_time_seconds(self) -> float: ...

    @property
    def success(self) -> bool: ...

    @property
    def error(self) -> str | None: ...

    @property
    def timestamp(self) -> datetime: ...


@runtime_checkable
class SandboxRuntime(Protocol):
    """Protocol for sandbox execution environments."""

    async def initialize(self) -> None:
        """Initialize the sandbox runtime (build images, create networks, etc.)."""
        ...

    async def run(
        self,
        code_path: str,
        timeout_seconds: int,
        env_vars: dict[str, str] | None = None,
    ) -> "SandboxResult":
        """Run code in sandbox and return results."""
        ...

    async def cleanup(self) -> None:
        """Clean up sandbox resources."""
        ...


class SandboxResult:
    """Result from sandbox execution."""

    def __init__(
        self,
        success: bool,
        tokens_per_second: float,
        total_tokens: int,
        wall_time_seconds: float,
        exit_code: int,
        stdout: str,
        stderr: str,
        error: str | None = None,
        # Verification fields
        final_loss: float | None = None,
        final_logits: "torch.Tensor | None" = None,  # noqa: F821
        final_logits_path: str | None = None,
    ):
        self.success = success
        self.tokens_per_second = tokens_per_second
        self.total_tokens = total_tokens
        self.wall_time_seconds = wall_time_seconds
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.error = error
        # Verification fields
        self.final_loss = final_loss
        self.final_logits = final_logits
        self.final_logits_path = final_logits_path


@runtime_checkable
class CodeValidator(Protocol):
    """Protocol for code validation."""

    def validate(self, code: str) -> "ValidationResult":
        """Validate code before execution."""
        ...


class ValidationResult:
    """Result from code validation."""

    def __init__(
        self,
        valid: bool,
        errors: list[str] | None = None,
    ):
        self.valid = valid
        self.errors = errors or []


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for database operations."""

    async def save_submission(self, submission: Any) -> None: ...

    async def get_submission(self, submission_id: str) -> Any | None: ...

    async def update_submission_status(
        self, submission_id: str, status: SubmissionStatus
    ) -> None: ...

    async def save_evaluation(self, evaluation: Any) -> None: ...

    async def get_evaluations(self, submission_id: str) -> list[Any]: ...

    async def get_leaderboard(self, limit: int = 100) -> list[Any]: ...

    async def get_top_submission(self) -> Any | None: ...

    async def get_pending_submissions(self) -> list[Any]: ...
