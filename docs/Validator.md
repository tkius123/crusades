# Validator Guide

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              YOUR LOCAL MACHINE                                  │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐    │
│   │                         validator.py main loop                          │    │
│   │                                                                         │    │
│   │   1. Read miner commitments from Bittensor blockchain                   │    │
│   │   2. Decrypt timelock-encrypted code URLs                               │    │
│   │   3. Download miner's train.py from URL                                 │    │
│   │   4. Evaluate via affinetes_runner.evaluate(code=...)                   │    │
│   │   5. Calculate median TPS from multiple runs                            │    │
│   │   6. Set weights on blockchain (winner-takes-all)                       │    │
│   └─────────────────────────────────┬───────────────────────────────────────┘    │
│                                     │                                            │
│                           ┌─────────┴─────────┐                                  │
│                           │  --affinetes-mode │                                  │
│                           └─────────┬─────────┘                                  │
│                                     │                                            │
│               ┌─────────────────────┴─────────────────────┐                      │
│               │                                           │                      │
│               ▼                                           ▼                      │
│      ┌────────────────┐                         ┌────────────────┐               │
│      │  DOCKER MODE   │                         │ BASILICA MODE  │               │
│      └───────┬────────┘                         └───────┬────────┘               │
│              │                                          │                        │
└──────────────┼──────────────────────────────────────────┼────────────────────────┘
               │                                          │
               ▼                                          ▼
      ┌─────────────────────┐               ┌──────────────────────────────┐
      │   LOCAL DOCKER      │               │      BASILICA CLOUD          │
      │                     │               │                              │
      │  • Uses YOUR GPU    │               │  • Rents remote A100 GPU     │
      │  • Spawns container │               │  • Pulls your Docker image   │
      │  • templar-eval     │               │  • Runs FastAPI server       │
      │    image            │               │  • POST /evaluate            │
      │  • Logs streamed    │               │  • Returns TPS results       │
      │    with [DOCKER]    │               │  • Auto-terminates (TTL)     │
      │    prefix           │               │                              │
      └─────────────────────┘               └──────────────────────────────┘
```


---

## Quick Start

### Docker Mode (Local GPU)

```bash
# 1. Build evaluation image
cd environments/templar
docker build --no-cache -t templar-eval:latest .

# 2. Run validator
SUBTENSOR_NETWORK=finney \
PYTHONUNBUFFERED=1 \
uv run -m neurons.validator \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey \
    --affinetes-mode docker \
    2>&1 | tee logs/validator.log
```

### Basilica Mode (Remote GPU)

```bash
# 1. Build and push image to registry
cd environments/templar
docker build -t ghcr.io/YOUR_ORG/templar-eval:latest .
docker push ghcr.io/YOUR_ORG/templar-eval:latest

# 2. Run validator (no local GPU needed!)
SUBTENSOR_NETWORK=finney \
BASILICA_API_TOKEN="your-token" \
PYTHONUNBUFFERED=1 \
uv run -m neurons.validator \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey \
    --affinetes-mode basilica \
    2>&1 | tee logs/validator.log
```

---

## Docker Mode (Detailed)

### Prerequisites

1. **NVIDIA GPU** with CUDA support
2. **Docker** with GPU support (`nvidia-container-toolkit`)
3. **Built evaluation image**

### Step 1: Verify GPU Access

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Step 2: Build Evaluation Image

```bash
cd environments/templar
docker build -t templar-eval:latest .
```

### Step 3: Configure hparams.json

Edit `hparams/hparams.json`:

```json
{
    "netuid": 3,
    "docker": {
        "gpu_devices": "0",
        "memory_limit": "32g",
        "shm_size": "8g"
    }
}
```

| Setting | Description | Examples |
|---------|-------------|----------|
| `netuid` | Subnet ID for Crusades | `3` |
| `gpu_devices` | Which GPUs to use | `"0"`, `"0,1"`, `"all"` |
| `memory_limit` | Container memory limit | `"32g"`, `"64g"` |
| `shm_size` | Shared memory (PyTorch) | `"8g"`, `"16g"` |

### Step 4: Run Validator

```bash
# Mainnet (Production)
SUBTENSOR_NETWORK=finney \
PYTHONUNBUFFERED=1 \
uv run -m neurons.validator \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey \
    --affinetes-mode docker \
    2>&1 | tee logs/validator.log

# Localnet (Testing)
SUBTENSOR_NETWORK=local \
PYTHONUNBUFFERED=1 \
uv run -m neurons.validator \
    --wallet.name templar_test \
    --wallet.hotkey V1 \
    --affinetes-mode docker \
    2>&1 | tee logs/validator.log
```

---

## Basilica Mode (Detailed)

### How Basilica Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BASILICA FLOW                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   FIRST EVALUATION (~2-4 minutes)                                           │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │ 1. Validator requests deployment from Basilica API               │      │
│   │ 2. Basilica provisions A100 GPU                                  │      │
│   │ 3. Basilica pulls Docker image from ghcr.io                      │      │
│   │ 4. Basilica starts FastAPI server (uvicorn)                      │      │
│   │ 5. Validator calls POST /evaluate with miner's code              │      │
│   │ 6. Basilica returns TPS results                                  │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   SUBSEQUENT EVALUATIONS (instant)                                          │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │ 1. Validator reuses existing deployment                          │      │
│   │ 2. Validator calls POST /evaluate with miner's code              │      │
│   │ 3. Basilica returns TPS results                                  │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   AFTER TTL EXPIRES (default: 1 hour)                                       │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │ 1. Deployment auto-terminates                                    │      │
│   │ 2. GPU released                                                  │      │
│   │ 3. Next evaluation creates new deployment                        │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Prerequisites

1. **Basilica API Token** (contact Basilica team)
2. **Docker image pushed to public registry** (ghcr.io)
3. **No local GPU required!**

### Step 1: Get Basilica API Token

Contact the Basilica team to get your `BASILICA_API_TOKEN`.

### Step 2: Build and Push Docker Image

```bash
# Build the evaluation image
cd environments/templar
docker build -t ghcr.io/YOUR_ORG/templar-eval:latest .

# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin

# Push the image (must be PUBLIC for Basilica to pull)
docker push ghcr.io/YOUR_ORG/templar-eval:latest
```

> ⚠️ **Important**: The image must be **public** in ghcr.io settings for Basilica to pull it.

### Step 3: Configure Environment

Add to your `.env` file:

```bash
BASILICA_API_TOKEN="basilica_xxxxx..."
SUBTENSOR_NETWORK=finney
```

### Step 4: Configure hparams.json

Edit `hparams/hparams.json`:

```json
{
    "netuid": 3,
    "basilica": {
        "image": "ghcr.io/YOUR_ORG/templar-eval:latest",
        "ttl_seconds": 3600,
        "gpu_count": 1,
        "gpu_models": ["A100"],
        "min_gpu_memory_gb": 40,
        "cpu": "6",
        "memory": "32Gi"
    }
}
```

| Setting | Description | Default |
|---------|-------------|---------|
| `image` | Docker image URL | Required |
| `ttl_seconds` | Deployment lifetime | 3600 (1 hour) |
| `gpu_count` | Number of GPUs | 1 |
| `gpu_models` | Acceptable GPU types | ["A100"] |
| `min_gpu_memory_gb` | Minimum VRAM | 40 |
| `cpu` | CPU cores | "6" |
| `memory` | Memory limit | "32Gi" |

### Step 5: Run Validator

```bash
# Load environment and run
source .venv/bin/activate
set -a && source .env && set +a

PYTHONUNBUFFERED=1 \
uv run -m neurons.validator \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey \
    --affinetes-mode basilica \
    2>&1 | tee logs/validator.log
```

### Step 6 (Optional): Run Without Local GPU

If you want zero local GPU memory usage:

```bash
CUDA_VISIBLE_DEVICES="" \
PYTHONUNBUFFERED=1 \
uv run -m neurons.validator \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey \
    --affinetes-mode basilica \
    2>&1 | tee logs/validator.log
```

---

## Management

### View Logs

```bash
# Real-time logs
tail -f logs/validator.log

# Search for evaluations
grep -E "BASILICA|DOCKER|TPS|PASSED|FAILED" logs/validator.log
```

### Check Status

```bash
# Check if running
ps aux | grep neurons.validator

# Check GPU usage (Docker mode)
nvidia-smi
```

### Stop Validator

```bash
# Graceful stop
pkill -f neurons.validator

# Or Ctrl+C if running in foreground
```

---

## Monitor (TUI)

The TUI dashboard provides real-time monitoring of crusades activity from your local database.

### Local Database (Validators Only)

Read directly from your validator's SQLite database:

```bash
uv run -m crusades.tui --db crusades.db
```

### TUI Features

- Leaderboard with TPS scores
- Recent submissions and their status
- TPS history chart
- Evaluation queue
- View submission code (after evaluation)

## View Submission Code

Miner code is stored in the database after evaluation.

```bash
# List all submissions
uv run scripts/view_submission.py

# View specific submission
uv run scripts/view_submission.py commit_9303_1

# Save code to file
uv run scripts/view_submission.py commit_9303_1 --save
```
