"""Configuration management for templar-tournament."""

import json
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SandboxConfig(BaseModel):
    """Sandbox execution settings."""

    memory_limit: str = "16g"
    cpu_count: int = 4
    gpu_count: int = 1
    pids_limit: int = 256


class StorageConfig(BaseModel):
    """Storage settings."""

    database_url: str = "sqlite+aiosqlite:///tournament.db"


class HParams(BaseModel):
    """Hyperparameters loaded from hparams.json."""

    netuid: int = 3

    # Evaluation settings
    num_evals_per_submission: int = 3
    eval_steps: int = 100
    eval_timeout: int = 600

    # Benchmark settings
    benchmark_model_size: str = "150M"
    benchmark_sequence_length: int = 1024
    benchmark_batch_size: int = 8

    # Timing settings
    set_weights_interval_seconds: int = 600

    # Nested configs
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @classmethod
    def load(cls, path: Path | str | None = None) -> Self:
        """Load hyperparameters from JSON file."""
        if path is None:
            # Default to hparams/hparams.json relative to project root
            path = Path(__file__).parent.parent.parent.parent / "hparams" / "hparams.json"
        else:
            path = Path(path)

        if not path.exists():
            return cls()

        with open(path) as f:
            data = json.load(f)

        return cls.model_validate(data)


class Config(BaseSettings):
    """Runtime configuration from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="TOURNAMENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Bittensor settings
    wallet_name: str = "default"
    wallet_hotkey: str = "default"
    subtensor_network: str = "finney"

    # R2/S3 storage (for code submissions)
    r2_account_id: str = ""
    r2_bucket_name: str = "tournament-submissions"
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Paths
    hparams_path: str = "hparams/hparams.json"
    benchmark_model_path: str = "benchmark/model"
    benchmark_data_path: str = "benchmark/data"

    # Debug
    debug: bool = False


# Global instances (lazy loaded)
_hparams: HParams | None = None
_config: Config | None = None


def get_hparams() -> HParams:
    """Get or create global HParams instance."""
    global _hparams
    if _hparams is None:
        config = get_config()
        _hparams = HParams.load(config.hparams_path)
    return _hparams


def get_config() -> Config:
    """Get or create global Config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config
