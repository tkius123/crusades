"""API client for tournament endpoints."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx


@dataclass
class TournamentData:
    """Container for all tournament data."""

    overview: dict[str, Any]
    validator: dict[str, Any]
    leaderboard: list[dict[str, Any]]
    recent: list[dict[str, Any]]
    queue: dict[str, Any]
    history: list[dict[str, Any]]


@dataclass
class SubmissionDetail:
    """Container for submission detail data."""

    submission: dict[str, Any]
    evaluations: list[dict[str, Any]]
    code: str | None


class MockClient:
    """Mock client that returns demo data."""

    def __init__(self):
        from tournament.tui.mock_data import (
            MOCK_CODE,
            MOCK_EVALUATIONS,
            MOCK_HISTORY,
            MOCK_LEADERBOARD,
            MOCK_OVERVIEW,
            MOCK_QUEUE,
            MOCK_RECENT,
            MOCK_SUBMISSIONS,
            MOCK_VALIDATOR,
            get_default_evaluations,
            get_default_submission,
        )

        self._overview = MOCK_OVERVIEW
        self._validator = MOCK_VALIDATOR
        self._queue = MOCK_QUEUE
        self._leaderboard = MOCK_LEADERBOARD
        self._recent = MOCK_RECENT
        self._history = MOCK_HISTORY
        self._submissions = MOCK_SUBMISSIONS
        self._evaluations = MOCK_EVALUATIONS
        self._code = MOCK_CODE
        self._get_default_submission = get_default_submission
        self._get_default_evaluations = get_default_evaluations

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get_overview(self) -> dict[str, Any]:
        return self._overview

    def get_validator_status(self) -> dict[str, Any]:
        return self._validator

    def get_leaderboard(self, limit: int = 10) -> list[dict[str, Any]]:
        return self._leaderboard[:limit]

    def get_recent_submissions(self) -> list[dict[str, Any]]:
        return self._recent

    def get_queue_stats(self) -> dict[str, Any]:
        return self._queue

    def get_history(self) -> list[dict[str, Any]]:
        return self._history

    def fetch_all(self) -> TournamentData:
        return TournamentData(
            overview=self.get_overview(),
            validator=self.get_validator_status(),
            leaderboard=self.get_leaderboard(),
            recent=self.get_recent_submissions(),
            queue=self.get_queue_stats(),
            history=self.get_history(),
        )

    def get_submission(self, submission_id: str) -> dict[str, Any]:
        return self._submissions.get(submission_id, self._get_default_submission(submission_id))

    def get_submission_evaluations(self, submission_id: str) -> list[dict[str, Any]]:
        return self._evaluations.get(submission_id, self._get_default_evaluations(submission_id))

    def get_submission_code(self, submission_id: str) -> str | None:
        return self._code

    def fetch_submission_detail(self, submission_id: str) -> SubmissionDetail:
        return SubmissionDetail(
            submission=self.get_submission(submission_id),
            evaluations=self.get_submission_evaluations(submission_id),
            code=self.get_submission_code(submission_id),
        )


class TournamentClient:
    """Client for fetching tournament data from API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=10.0)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _get(self, endpoint: str) -> dict[str, Any] | list[Any]:
        """Make a GET request to the API."""
        try:
            response = self._client.get(f"{self.base_url}{endpoint}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError:
            return {}

    def get_overview(self) -> dict[str, Any]:
        """Get dashboard overview stats."""
        data = self._get("/api/stats/overview")
        if not data:
            return {
                "submissions_24h": 0,
                "current_top_score": 0.0,
                "score_improvement_24h": 0.0,
                "total_submissions": 0,
                "active_miners": 0,
            }
        return data

    def get_validator_status(self) -> dict[str, Any]:
        """Get validator status."""
        data = self._get("/api/stats/validator")
        if not data:
            return {
                "status": "unknown",
                "evaluations_completed_1h": 0,
                "current_evaluation": None,
                "uptime": "N/A",
            }
        return data

    def get_leaderboard(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get leaderboard entries."""
        data = self._get(f"/leaderboard?limit={limit}")
        if not data or not isinstance(data, list):
            return []
        return data

    def get_recent_submissions(self) -> list[dict[str, Any]]:
        """Get recent submissions."""
        data = self._get("/api/stats/recent")
        if not data or not isinstance(data, list):
            return []
        return data

    def get_queue_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        data = self._get("/api/stats/queue")
        if not data:
            return {
                "pending_count": 0,
                "running_count": 0,
                "finished_count": 0,
                "avg_wait_time_seconds": 0.0,
                "avg_score": 0.0,
            }
        return data

    def get_history(self) -> list[dict[str, Any]]:
        """Get TPS history."""
        data = self._get("/api/stats/history")
        if not data or not isinstance(data, list):
            return []
        return data

    def fetch_all(self) -> TournamentData:
        """Fetch all tournament data."""
        return TournamentData(
            overview=self.get_overview(),
            validator=self.get_validator_status(),
            leaderboard=self.get_leaderboard(),
            recent=self.get_recent_submissions(),
            queue=self.get_queue_stats(),
            history=self.get_history(),
        )

    def get_submission(self, submission_id: str) -> dict[str, Any]:
        """Get submission details."""
        data = self._get(f"/api/submissions/{submission_id}")
        if not data:
            return {}
        return data

    def get_submission_evaluations(self, submission_id: str) -> list[dict[str, Any]]:
        """Get evaluations for a submission."""
        data = self._get(f"/api/submissions/{submission_id}/evaluations")
        if not data or not isinstance(data, list):
            return []
        return data

    def get_submission_code(self, submission_id: str) -> str | None:
        """Get code for a submission."""
        data = self._get(f"/api/submissions/{submission_id}/code")
        if not data or not isinstance(data, dict):
            return None
        return data.get("code")

    def fetch_submission_detail(self, submission_id: str) -> SubmissionDetail:
        """Fetch all details for a submission."""
        return SubmissionDetail(
            submission=self.get_submission(submission_id),
            evaluations=self.get_submission_evaluations(submission_id),
            code=self.get_submission_code(submission_id),
        )


def format_time_ago(timestamp_str: str | None) -> str:
    """Format a timestamp as 'X ago' string."""
    if not timestamp_str:
        return "N/A"
    try:
        if timestamp_str.endswith("Z"):
            timestamp_str = timestamp_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(timestamp_str)
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        delta = now - dt

        if delta.days > 0:
            return f"{delta.days}d ago"
        hours = delta.seconds // 3600
        if hours > 0:
            return f"{hours}h ago"
        minutes = delta.seconds // 60
        if minutes > 0:
            return f"{minutes}m ago"
        return "just now"
    except (ValueError, TypeError):
        return "N/A"
