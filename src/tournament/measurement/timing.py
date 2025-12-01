"""External timing measurement for TPS calculation.

The key principle: timing is measured by the validator (host-side),
not by the miner's code. This prevents timing manipulation.
"""

import time
from dataclasses import dataclass


@dataclass
class TimingResult:
    """Result of timing measurement."""

    elapsed_seconds: float
    start_time: float
    end_time: float


class ExternalTimer:
    """High-precision timer for measuring sandbox execution time.

    Uses time.perf_counter() for maximum precision.
    Timing happens outside the sandbox container.
    """

    def __init__(self):
        self._start_time: float | None = None
        self._end_time: float | None = None

    def start(self) -> None:
        """Start the timer."""
        self._start_time = time.perf_counter()
        self._end_time = None

    def stop(self) -> float:
        """Stop the timer and return elapsed seconds."""
        if self._start_time is None:
            raise RuntimeError("Timer was not started")
        self._end_time = time.perf_counter()
        return self.elapsed

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self._start_time is None:
            return 0.0
        end = self._end_time if self._end_time is not None else time.perf_counter()
        return end - self._start_time

    @property
    def result(self) -> TimingResult:
        """Get full timing result."""
        if self._start_time is None or self._end_time is None:
            raise RuntimeError("Timer not properly started/stopped")
        return TimingResult(
            elapsed_seconds=self._end_time - self._start_time,
            start_time=self._start_time,
            end_time=self._end_time,
        )

    def __enter__(self) -> "ExternalTimer":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.stop()


def calculate_tps(total_tokens: int, elapsed_seconds: float) -> float:
    """Calculate tokens per second."""
    if elapsed_seconds <= 0:
        return 0.0
    return total_tokens / elapsed_seconds
