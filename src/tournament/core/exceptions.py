"""Custom exceptions for templar-tournament."""


class TournamentError(Exception):
    """Base exception for all tournament errors."""

    pass


class SandboxError(TournamentError):
    """Error during sandbox execution."""

    pass


class SandboxTimeoutError(SandboxError):
    """Sandbox execution timed out."""

    pass


class SandboxCrashError(SandboxError):
    """Code crashed during sandbox execution."""

    pass


class EvaluationError(TournamentError):
    """Error during evaluation."""

    pass


class ValidationError(TournamentError):
    """Code validation failed."""

    pass


class StorageError(TournamentError):
    """Error accessing storage (database or R2)."""

    pass


class ChainError(TournamentError):
    """Error interacting with Bittensor chain."""

    pass
