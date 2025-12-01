# Templar Tournament

A Bittensor subnet where miners compete on training code efficiency, measured by **tokens per second (TPS)**.

## Overview

Miners submit optimized training code that is evaluated in isolated Docker sandboxes. The **winner-takes-all**: 100% of subnet emissions go to the single top-scoring submission.

### Key Features

- **Competition Metric**: Tokens per second (TPS)
- **Incentive Model**: Winner-takes-all (top scorer gets 100%)
- **Evaluation**: Isolated Docker sandbox with external timing
- **Security**: Network disabled, resource limits, code validation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         MINERS                              │
│  Submit train.py with:                                      │
│  - setup_model(path) -> model                               │
│  - setup_data(path, batch_size, seq_len) -> iterator        │
│  - train_step(model, batch) -> tokens_processed             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       VALIDATORS                            │
│  1. Validate code (syntax, required functions)              │
│  2. Run in Docker sandbox (network disabled)                │
│  3. Measure TPS externally (host-side timing)               │
│  4. Set weights: 100% to top scorer                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    BITTENSOR NETWORK                        │
│  Winner receives 100% of subnet emissions                   │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
git clone https://github.com/your-org/templar-tournament.git
cd templar-tournament
uv sync
```

## Usage

### Running a Validator

```bash
uv run python -m neurons.validator \
    --wallet.name default \
    --wallet.hotkey default \
    --burn-hotkey <fallback_hotkey>
```

### Submitting Code (Miner)

```bash
uv run python -m neurons.miner train.py \
    --wallet.name default \
    --wallet.hotkey default
```

### Running the API

```bash
uv run python -m api.app
```

## Miner Submission Format

Create a `train.py` with these required functions:

```python
import torch

def setup_model(model_path: str) -> torch.nn.Module:
    """Load the benchmark model. Apply your optimizations here."""
    # Example: torch.compile, mixed precision, etc.
    pass

def setup_data(data_path: str, batch_size: int, seq_len: int) -> Iterator:
    """Create a data iterator. Optimize prefetching, pinned memory, etc."""
    pass

def train_step(model: torch.nn.Module, batch: torch.Tensor) -> int:
    """Execute one training step. Return number of tokens processed."""
    # Forward pass, backward pass, optimizer step
    # Return: batch_size * sequence_length (tokens processed)
    pass
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TOURNAMENT_WALLET_NAME` | Wallet name | `default` |
| `TOURNAMENT_WALLET_HOTKEY` | Wallet hotkey | `default` |
| `TOURNAMENT_SUBTENSOR_NETWORK` | Network (finney/test) | `finney` |
| `TOURNAMENT_API_HOST` | API host | `0.0.0.0` |
| `TOURNAMENT_API_PORT` | API port | `8000` |

### Hyperparameters (`hparams/hparams.json`)

```json
{
    "netuid": 3,
    "num_evals_per_submission": 3,
    "eval_steps": 100,
    "eval_timeout": 600,
    "set_weights_interval_seconds": 600
}
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /leaderboard` | Current rankings |
| `GET /submissions/{id}` | Submission status |
| `GET /submissions/{id}/evaluations` | Evaluation results |
| `GET /health` | Health check |

## Project Structure

```
templar-tournament/
├── src/tournament/          # Core library
│   ├── config.py           # Configuration
│   ├── schemas.py          # Pydantic models
│   ├── core/               # Protocols & exceptions
│   ├── chain/              # Bittensor integration
│   ├── sandbox/            # Docker sandbox
│   ├── pipeline/           # Code validation
│   ├── measurement/        # TPS timing
│   └── storage/            # Database
├── neurons/                 # Node implementations
│   ├── validator.py        # Validator
│   └── miner.py            # Miner CLI
├── api/                     # FastAPI application
├── hparams/                 # Configuration files
└── benchmark/               # Model & data for benchmarking
```

## Documentation

- [Validator Guide](docs/validator.md) - How to run a validator
- [Miner Guide](docs/miner.md) - How to submit optimized training code
- [API Guide](docs/api.md) - REST API reference

## License

MIT
