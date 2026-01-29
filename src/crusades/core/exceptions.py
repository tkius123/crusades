"""Custom exceptions for templar-crusades."""


class CrusadesError(Exception):
    """Base exception for all crusades errors."""

    pass


class SandboxError(CrusadesError):
    """Error during sandbox execution."""

    pass


class SandboxTimeoutError(SandboxError):
    """Sandbox execution timed out."""

    pass


class SandboxCrashError(SandboxError):
    """Code crashed during sandbox execution."""

    pass


class EvaluationError(CrusadesError):
    """Error during evaluation."""

    pass


class ValidationError(CrusadesError):
    """Code validation failed."""

    pass


class StorageError(CrusadesError):
    """Error accessing storage (database)."""

    pass


class ChainError(CrusadesError):
    """Error interacting with Bittensor chain."""

    pass
