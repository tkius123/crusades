# ⚡ Templar Tournament

**Compete to write the fastest PyTorch training code. Winner takes all!**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TOURNAMENT FLOW                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   MINER                          VALIDATOR                    BLOCKCHAIN     │
│                                                                              │
│   ┌──────────┐                                                               │
│   │ train.py │                                                               │
│   └────┬─────┘                                                               │
│        │                                                                     │
│        ▼                                                                     │
│   ┌──────────┐     pay 0.1 TAO      ┌──────────────┐                        │
│   │  Submit  │ ──────────────────▶  │   Receive    │                        │
│   │  + Sign  │                      │   & Verify   │                        │
│   └────┬─────┘                      └──────┬───────┘                        │
│        │                                   │                                 │
│        │       code + signature            │                                 │
│        └──────────────────────────────────▶│                                 │
│                                            │                                 │
│                                   ┌────────▼────────┐                        │
│                                   │   EVALUATION    │                        │
│                                   │   (5x runs)     │                        │
│                                   │                 │                        │
│                                   │  Run 1: 1250 ──┐│                        │
│                                   │  Run 2: 1180   ││                        │
│                                   │  Run 3: 1220 ◄─┤│ median                 │
│                                   │  Run 4: 1195   ││                        │
│                                   │  Run 5: 1240 ──┘│                        │
│                                   └────────┬────────┘                        │
│                                            │                                 │
│                                            ▼                                 │
│                                   ┌─────────────────┐    set weights        │
│   ┌──────────┐                    │   LEADERBOARD   │ ─────────────────────▶│
│   │  Check   │ ◄──────────────────│   #1 = Winner   │                       │
│   │  Score   │                    └─────────────────┘                       │
│   └──────────┘                                                              │
│                                                                              │
│        ▼                                               ┌──────────────────┐  │
│   ┌──────────┐                                         │   EMISSIONS      │  │
│   │  WINNER  │ ◄───────────────────────────────────────│   95% validator  │  │
│   │  gets 5% │                                         │    5% winner     │  │
│   └──────────┘                                         └──────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Miners

```bash
# 1. Setup
git clone https://github.com/tplr-ai/templar-tournament.git && cd templar-tournament
uv sync && uv run python scripts/setup_miner.py

# 2. Test locally (free!)
uv run python -m local_test train.py --steps 5

# 3. Submit (costs 0.1 TAO)
uv run python -m neurons.miner train.py \
    --wallet.name mywallet --wallet.hotkey myhotkey \
    --payment-recipient <VALIDATOR_HOTKEY> \
    --validator-api http://<VALIDATOR_IP>:8000
```

### Validators

```bash
# 1. Setup
git clone https://github.com/tplr-ai/templar-tournament.git && cd templar-tournament
uv sync && uv run python scripts/setup_validator.py

# 2. Build sandbox
cd src/tournament/sandbox && docker build -t tournament-sandbox:latest . && cd ../../..

# 3. Run (two terminals)
uv run python -m api.app                    # API + Dashboard
uv run python -m neurons.validator          # Evaluation loop
```

---

## How It Works

| Step | What Happens |
|------|-------------|
| 1. **Pay** | Miner pays 0.1 TAO to validator (anti-spam) |
| 2. **Submit** | Code uploaded with cryptographic signature |
| 3. **Evaluate** | Sandbox runs code 5 times with different seeds |
| 4. **Score** | Median TPS = final score (protects against outliers) |
| 5. **Rank** | Highest TPS → #1 on leaderboard |
| 6. **Reward** | Winner receives 5% of emissions, validator 95% |

**TPS = Tokens Per Second = (batch_size × seq_length × steps) / time**

---

## Code Requirements

Your `train.py` must implement:

```python
def inner_steps(model, data_iterator, optimizer, num_steps, device):
    """Run training steps.
    
    Returns: InnerStepsResult(final_logits, total_tokens, final_loss)
    """
    total_tokens = 0
    for step in range(num_steps):
        batch = next(data_iterator).to(device)  # Shape: (8, 1024)
        
        # Your optimized training here
        # - Process input_ids[:, :-1] → predict labels[:, 1:]
        # - Use autocast for bfloat16
        # - backward() + optimizer.step()
        
        total_tokens += batch.numel()
    
    return InnerStepsResult(final_logits, total_tokens, final_loss)
```

**Rules:**
- ✅ Output logits must match reference within 2%
- ✅ Process correct token count
- ✅ Complete within 10 minutes
- ❌ No network access, subprocess, or filesystem writes

**Sandbox includes:** PyTorch + CUDA, Transformers, flash-attn, torchtitan

---

## Anti-Copying Protection

| Protection | How It Works |
|-----------|--------------|
| **Duplicate Detection** | Same code hash → rejected |
| **Similarity Check** | >90% similar to existing → rejected |
| **Cooldown** | 5 minute wait between submissions |
| **Hidden Until Done** | Code not visible until evaluated |

---

## Dashboard

**Web:** `http://<VALIDATOR_IP>:8000/`

**TUI:** `uv run python -m tournament.tui`

| Key | Action |
|-----|--------|
| `j/k` | Navigate/scroll |
| `h/l` | Switch panels |
| `Enter` | View details |
| `c` | Toggle code |
| `q` | Quit |

---

## API Endpoints

```bash
GET  /leaderboard                    # Rankings
GET  /api/stats/overview             # Network stats
GET  /api/stats/recent               # Recent submissions
GET  /api/submissions/{id}           # Submission details
GET  /api/submissions/{id}/code      # View code (after eval)
```

---

## Configuration (hparams.json)

```json
{
    "netuid": 2,
    "evaluation_runs": 5,
    "benchmark_model_name": "Qwen/Qwen2.5-3B",
    "benchmark_batch_size": 8,
    "benchmark_sequence_length": 1024,
    "submission_cost_rao": 100000000,
    "burn_rate": 0.95,
    "burn_uid": 0
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `evaluation_runs` | 5 | Runs per submission |
| `benchmark_batch_size` | 8 | Tokens per batch |
| `submission_cost_rao` | 1e8 | 0.1 TAO in RAO |
| `burn_rate` | 0.95 | % to validator |
| `verification.output_vector_tolerance` | 0.02 | 2% max diff |

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| "Output logits don't match" | Check loss function, autocast settings |
| "Timeout exceeded" | Optimize your loop |
| "Forbidden import" | Remove os/socket/subprocess |
| "Too similar" | Make significant code changes |
| "Cooldown active" | Wait 5 minutes |

---

**Ready to compete? Optimize your code and claim #1! ⚡**
