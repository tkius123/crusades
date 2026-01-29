"""Database abstraction layer."""

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ..config import get_hparams
from ..core.protocols import SubmissionStatus
from .models import Base, EvaluationModel, SubmissionModel, ValidatorStateModel


class Database:
    """Async database interface."""

    def __init__(self, url: str | None = None):
        if url is None:
            url = get_hparams().storage.database_url
        self.engine = create_async_engine(url, echo=False)
        self.session_factory = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def initialize(self) -> None:
        """Create tables if they don't exist."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        """Close database connection."""
        await self.engine.dispose()

    # Submission operations

    async def save_submission(self, submission: SubmissionModel) -> None:
        """Save a new submission."""
        async with self.session_factory() as session:
            session.add(submission)
            await session.commit()

    async def get_submission(self, submission_id: str) -> SubmissionModel | None:
        """Get submission by ID."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(SubmissionModel).where(SubmissionModel.submission_id == submission_id)
            )
            return result.scalar_one_or_none()

    async def update_submission_status(
        self,
        submission_id: str,
        status: SubmissionStatus,
        error_message: str | None = None,
    ) -> None:
        """Update submission status."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(SubmissionModel).where(SubmissionModel.submission_id == submission_id)
            )
            submission = result.scalar_one_or_none()
            if submission:
                submission.status = status
                if error_message:
                    submission.error_message = error_message
                await session.commit()

    async def update_submission_score(self, submission_id: str, final_score: float) -> None:
        """Update submission final score."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(SubmissionModel).where(SubmissionModel.submission_id == submission_id)
            )
            submission = result.scalar_one_or_none()
            if submission:
                submission.final_score = final_score
                submission.status = SubmissionStatus.FINISHED
                await session.commit()

    async def update_submission_code(self, submission_id: str, code: str) -> None:
        """Store miner's code content after evaluation.

        This is called by the validator after downloading and evaluating
        the miner's code, so it can be displayed on the dashboard.

        Args:
            submission_id: Submission ID
            code: Miner's train.py code content
        """
        async with self.session_factory() as session:
            result = await session.execute(
                select(SubmissionModel).where(SubmissionModel.submission_id == submission_id)
            )
            submission = result.scalar_one_or_none()
            if submission:
                submission.code_content = code
                await session.commit()

    async def get_submission_code(self, submission_id: str) -> str | None:
        """Get miner's code content for a submission.

        Args:
            submission_id: Submission ID

        Returns:
            Code content or None if not stored
        """
        async with self.session_factory() as session:
            result = await session.execute(
                select(SubmissionModel.code_content).where(
                    SubmissionModel.submission_id == submission_id
                )
            )
            return result.scalar_one_or_none()

    async def get_all_submissions(self) -> list[SubmissionModel]:
        """Get all submissions."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(SubmissionModel).order_by(desc(SubmissionModel.created_at))
            )
            return list(result.scalars().all())

    async def get_pending_submissions(self) -> list[SubmissionModel]:
        """Get submissions pending validation."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(SubmissionModel)
                .where(SubmissionModel.status == SubmissionStatus.PENDING)
                .order_by(SubmissionModel.created_at)
            )
            return list(result.scalars().all())

    async def get_evaluating_submissions(self) -> list[SubmissionModel]:
        """Get submissions currently being evaluated."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(SubmissionModel)
                .where(SubmissionModel.status == SubmissionStatus.EVALUATING)
                .order_by(SubmissionModel.created_at)
            )
            return list(result.scalars().all())

    async def get_latest_submission_by_hotkey(self, hotkey: str) -> SubmissionModel | None:
        """Get the most recent submission from a miner.

        Used for rate limiting - checking when miner last submitted.

        Args:
            hotkey: Miner's hotkey

        Returns:
            Most recent submission or None
        """
        async with self.session_factory() as session:
            result = await session.execute(
                select(SubmissionModel)
                .where(SubmissionModel.miner_hotkey == hotkey)
                .order_by(desc(SubmissionModel.created_at))
                .limit(1)
            )
            return result.scalar_one_or_none()

    # Evaluation operations

    async def save_evaluation(self, evaluation: EvaluationModel) -> None:
        """Save a new evaluation result."""
        async with self.session_factory() as session:
            session.add(evaluation)
            await session.commit()

    async def get_evaluations(self, submission_id: str) -> list[EvaluationModel]:
        """Get all evaluations for a submission."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(EvaluationModel)
                .where(EvaluationModel.submission_id == submission_id)
                .order_by(EvaluationModel.created_at)
            )
            return list(result.scalars().all())

    async def count_evaluations(self, submission_id: str) -> int:
        """Count ALL evaluations for a submission (including failed)."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(func.count())
                .select_from(EvaluationModel)
                .where(
                    EvaluationModel.submission_id == submission_id,
                )
            )
            return result.scalar() or 0

    # Leaderboard operations

    async def get_top_submission(self) -> SubmissionModel | None:
        """Get the top-scoring finished submission."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(SubmissionModel)
                .where(
                    SubmissionModel.status == SubmissionStatus.FINISHED,
                    SubmissionModel.final_score.isnot(None),
                )
                .order_by(desc(SubmissionModel.final_score))
                .limit(1)
            )
            return result.scalar_one_or_none()

    async def get_leaderboard(self, limit: int = 100) -> list[SubmissionModel]:
        """Get top submissions by score."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(SubmissionModel)
                .where(
                    SubmissionModel.status == SubmissionStatus.FINISHED,
                    SubmissionModel.final_score.isnot(None),
                )
                .order_by(desc(SubmissionModel.final_score))
                .limit(limit)
            )
            return list(result.scalars().all())

    async def get_top_submissions(self, limit: int = 5) -> list[SubmissionModel]:
        """Get top N submissions by score for similarity checking.

        This is used during submission to check if new code is similar
        to existing top-performing code (anti-copying).

        Args:
            limit: Number of top submissions to return (default 5)

        Returns:
            List of top submissions sorted by score (highest first)
        """
        async with self.session_factory() as session:
            result = await session.execute(
                select(SubmissionModel)
                .where(
                    SubmissionModel.status == SubmissionStatus.FINISHED,
                    SubmissionModel.final_score.isnot(None),
                )
                .order_by(desc(SubmissionModel.final_score))
                .limit(limit)
            )
            return list(result.scalars().all())

    # Validator state operations

    async def get_validator_state(self, key: str) -> str | None:
        """Get a validator state value."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(ValidatorStateModel.value).where(ValidatorStateModel.key == key)
            )
            return result.scalar_one_or_none()

    async def set_validator_state(self, key: str, value: str) -> None:
        """Set a validator state value (upsert)."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(ValidatorStateModel).where(ValidatorStateModel.key == key)
            )
            state = result.scalar_one_or_none()
            if state:
                state.value = value
            else:
                session.add(ValidatorStateModel(key=key, value=value))
            await session.commit()


# Global instance
_database: Database | None = None


async def get_database() -> Database:
    """Get or create global database instance."""
    global _database
    if _database is None:
        _database = Database()
        await _database.initialize()
    return _database
