# Validator Guide

This guide explains how to run a validator for the Templar Tournament subnet.

## Overview

Validators are responsible for:
1. **Validating** miner code submissions (syntax, required `inner_steps` function)
2. **Verifying** that submissions produce correct outputs (logits, loss, token count)
3. **Benchmarking** tokens per second (TPS) in isolated Docker sandboxes
4. **Setting weights** on the Bittensor network (winner-takes-all)

### Verification System

The validator ensures miners are actually training by:
- Running a **reference implementation** to compute expected outputs
- Running the **miner's code** in a sandbox with the same inputs
- **Comparing outputs**: logits, loss, and token count must match within tolerance

This prevents miners from submitting fake results or skipping actual computation.

## Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA 12.8 support
- **RAM**: 32GB+ recommended
- **Storage**: 50GB+ for Docker images and benchmark data

### Software
- Docker with NVIDIA Container Toolkit
- Python 3.12+
- uv package manager
- Bittensor wallet with registered hotkey on subnet 3

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/templar-tournament.git
cd templar-tournament

# Install dependencies
uv sync

# Build the sandbox Docker image
cd src/tournament/sandbox
docker build -t tournament-sandbox:latest .
cd ../../..
```

## Configuration

### Environment Variables

Create a `.env` file or export these variables:

```bash
# Bittensor
TOURNAMENT_WALLET_NAME=default
TOURNAMENT_WALLET_HOTKEY=default
TOURNAMENT_SUBTENSOR_NETWORK=finney

# Paths
TOURNAMENT_BENCHMARK_MODEL_PATH=benchmark/model
TOURNAMENT_BENCHMARK_DATA_PATH=benchmark/data

# Optional
TOURNAMENT_DEBUG=false
```

### Hyperparameters

Edit `hparams/hparams.json` if needed:

```json
{
    "netuid": 3,
    "num_evals_per_submission": 3,
    "eval_steps": 100,
    "eval_timeout": 600,
    "set_weights_interval_seconds": 600,
    "verification": {
        "logits_atol": 1e-3,
        "logits_rtol": 1e-3,
        "loss_tolerance": 1e-3,
        "verify_model_state": false,
        "deterministic_mode": true,
        "random_seed": 42
    }
}
```

**Verification tolerances:**
- `logits_atol/rtol`: Absolute/relative tolerance for logits comparison (1e-3 allows bf16 variance)
- `loss_tolerance`: Maximum allowed difference in final loss value
- `deterministic_mode`: Ensures reproducible execution with fixed seeds

## Running the Validator

### Basic Usage

```bash
uv run python -m neurons.validator \
    --wallet.name <your_wallet_name> \
    --wallet.hotkey <your_hotkey_name>
```

### With Burn Fallback

If no valid submissions exist, emissions go to the burn hotkey:

```bash
uv run python -m neurons.validator \
    --wallet.name default \
    --wallet.hotkey default \
    --burn-hotkey <fallback_ss58_address>
```

### Burn Mode (Testing)

Send all emissions to burn hotkey (useful for testing):

```bash
uv run python -m neurons.validator \
    --wallet.name default \
    --wallet.hotkey default \
    --burn-hotkey <burn_ss58_address> \
    --burn-enabled
```

## Validator Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                    VALIDATOR LOOP                            │
│                                                              │
│  1. Process pending submissions                              │
│     └─ Validate code (syntax, inner_steps function)         │
│                                                              │
│  2. Evaluate validated submissions                           │
│     ├─ Run REFERENCE inner_steps → expected outputs         │
│     ├─ Run MINER code in sandbox → actual outputs           │
│     ├─ VERIFY: Compare logits, loss, token count            │
│     │   └─ If mismatch → mark as FAILED with verbose error  │
│     ├─ Calculate TPS (tokens / wall_time)                   │
│     └─ Save evaluation results                              │
│                                                              │
│  3. Set weights (every 10 minutes)                          │
│     └─ Top verified scorer gets 100%                        │
│     └─ Or burn if no valid winner                           │
│                                                              │
│  4. Sync metagraph (every 5 minutes)                        │
│                                                              │
│  5. Sleep 10 seconds, repeat                                │
└─────────────────────────────────────────────────────────────┘
```

## Monitoring

### Logs

The validator logs to stdout with timestamps:

```
2024-01-15 10:30:00 | INFO | neurons.validator | Starting Validator...
2024-01-15 10:30:01 | INFO | neurons.validator | Hotkey: 5ABC...XYZ
2024-01-15 10:30:02 | INFO | neurons.validator | UID: 42
2024-01-15 10:30:15 | INFO | neurons.validator | Validating submission abc123
2024-01-15 10:30:16 | INFO | neurons.validator | Submission abc123 passed validation
2024-01-15 10:31:00 | INFO | neurons.validator | Evaluating submission abc123
2024-01-15 10:31:01 | INFO | verification.verifier | Step 1/4: Running reference execution (seed=42)...
2024-01-15 10:31:30 | INFO | verification.verifier |   Reference complete: 819,200 tokens, loss=2.3456
2024-01-15 10:31:31 | INFO | verification.verifier | Step 2/4: Preparing sandbox inputs...
2024-01-15 10:31:32 | INFO | verification.verifier | Step 3/4: Running miner code in sandbox...
2024-01-15 10:32:45 | INFO | verification.verifier |   Sandbox complete: 819,200 tokens in 54.32s
2024-01-15 10:32:46 | INFO | verification.verifier | Step 4/4: Verifying outputs...
2024-01-15 10:32:46 | INFO | verification.verifier |   [OK] Token count matches: 819,200
2024-01-15 10:32:46 | INFO | verification.verifier |   [OK] Loss matches: 2.3457 (diff=0.000100)
2024-01-15 10:32:46 | INFO | verification.verifier |   [OK] Logits match (atol=0.001, rtol=0.001)
2024-01-15 10:32:46 | INFO | verification.verifier | VERIFICATION PASSED - TPS: 15,082.35
2024-01-15 10:32:46 | INFO | neurons.validator | Verification PASSED for abc123
2024-01-15 10:32:46 | INFO | neurons.validator |   TPS: 15,082.35
2024-01-15 10:32:46 | INFO | neurons.validator |   Tokens: 819,200
2024-01-15 10:32:46 | INFO | neurons.validator |   Time: 54.32s
```

**Verification failure example:**
```
2024-01-15 10:35:00 | WARNING | neurons.validator | Verification FAILED for xyz789
2024-01-15 10:35:00 | WARNING | neurons.validator |   Error type: LogitsMismatchError
2024-01-15 10:35:00 | WARNING | neurons.validator |   Message: VERIFICATION FAILED: Output logits don't match expected values.
                                                        Max difference: 0.015234
                                                        Tolerance: 0.001000
```

### Database

Submissions and evaluations are stored in SQLite:

```bash
# Default location
tournament.db

# Query leaderboard
sqlite3 tournament.db "SELECT * FROM submissions ORDER BY final_score DESC LIMIT 10"
```

## Troubleshooting

### Docker Issues

```bash
# Verify Docker is running
docker info

# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu24.04 nvidia-smi

# Rebuild sandbox image
docker build -t tournament-sandbox:latest src/tournament/sandbox/
```

### Bittensor Issues

```bash
# Check wallet
btcli wallet list

# Check registration
btcli subnet metagraph --netuid 3
```

### Permission Issues

```bash
# Ensure Docker socket is accessible
sudo usermod -aG docker $USER
# Log out and back in
```

## Graceful Shutdown

The validator handles SIGINT and SIGTERM for graceful shutdown:

```bash
# Ctrl+C or
kill -SIGTERM <pid>
```

This ensures:
- Current evaluation completes
- Database connections close properly
- Docker containers are cleaned up
