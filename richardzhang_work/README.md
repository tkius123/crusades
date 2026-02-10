# richardzhang_work

Automated Bittensor Crusades miner management suite. Five Python services orchestrated by a single master loop that:

1. Fetches top leaderboard submissions and tracks your evaluation results
2. Detects gaming/cheating in top submissions using AI
3. Generates improved `train.py` code and auto-submits it on-chain
4. Commits and pushes your local changes
5. Syncs upstream repository changes into your fork

---

## Architecture overview

```
run_all.py  (master loop — runs steps 1-5 forever)
  |
  ├── 1. fetch_top_submissions.py --recent
  |       Fetches leaderboard + top N code + recent submissions.
  |       Updates improved/submissions.json with eval results for your UIDs.
  |
  ├── 2. check_gaming.py
  |       Sends new top-N submissions to a Cursor Background Agent.
  |       Classifies each as gaming (YES) or honest (NO).
  |
  ├── 3. improve_and_submit.py --submit
  |       Takes the #1 honest submission + your last eval result.
  |       Asks a Cursor Agent to generate an improved train.py.
  |       Submits via GitHub Gist + `neurons.miner submit`.
  |       Rotates across 3 wallets with cooldown-based selection.
  |
  ├── 4. git add + commit + push
  |       Stages richardzhang_work/ changes and pushes to origin.
  |
  └── 5. sync_upstream.py --push
          Fetches upstream, merges into main, runs uv sync, pushes.
```

---

## Environment setup

All scripts read `richardzhang_work/.env` automatically (not committed to git).

```bash
# richardzhang_work/.env
CURSOR_API_KEY=key-REPLACE_ME
CURSOR_REPO=https://github.com/your-user/crusades

# For auto-submit:
GITHUB_TOKEN=ghp_REPLACE_ME
WALLETS=[{"name":"coldkey1","hotkey":"hotkey1","uid":42},{"name":"coldkey2","hotkey":"hotkey2","uid":241},{"name":"coldkey3","hotkey":"hotkey3","uid":109}]
NETWORK=finney
```

| Variable | Required by | Description |
|----------|-------------|-------------|
| `CURSOR_API_KEY` | check_gaming, improve_and_submit | Cursor API token from [cursor.com/dashboard/integration](https://cursor.com/dashboard/integration) |
| `CURSOR_REPO` | check_gaming, improve_and_submit | Your GitHub fork URL |
| `GITHUB_TOKEN` | improve_and_submit (--submit) | GitHub PAT with `gist` scope |
| `WALLETS` | improve_and_submit (--submit) | JSON array of `{name, hotkey, uid}` wallet objects |
| `NETWORK` | improve_and_submit (--submit) | Bittensor network (default: `finney`) |

---

## Quick start — run everything

```bash
cd /root/workspace/crusades

# Full automation: fetch, check gaming, generate, submit, push, sync upstream
uv run python richardzhang_work/run_all.py

# Single cycle (no loop):
uv run python richardzhang_work/run_all.py --once

# Generate but don't submit to chain:
uv run python richardzhang_work/run_all.py --no-submit

# Don't push git changes:
uv run python richardzhang_work/run_all.py --once --no-push
```

| Flag | Effect |
|------|--------|
| `--once` | Run one cycle and exit |
| `--no-submit` | Generate improvements but don't submit to chain |
| `--no-push` | Skip git add/commit/push step |
| `--interval N` | Seconds between cycles (default: 60) |

> **Note:** Submission is **enabled by default**. Use `--no-submit` to disable it.

---

## 1. `fetch_top_submissions.py` — leaderboard + code fetcher

Connects to the Crusades API (`69.19.137.219:8080`) and downloads leaderboard stats, submission details, and source code.

```bash
# One-shot, top 5
uv run python richardzhang_work/fetch_top_submissions.py --top 5

# Also fetch recent submissions and update your eval results in submissions.json
uv run python richardzhang_work/fetch_top_submissions.py --top 5 --recent

# Service mode (fetch every 5 min)
uv run python richardzhang_work/fetch_top_submissions.py --service --interval 300
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--top N` | `5` | Number of top submissions to fetch |
| `--recent` | off | Also fetch recent submissions and update `improved/submissions.json` with eval results for your UIDs |
| `--api-url` | `http://69.19.137.219:8080` | Crusades API base URL |
| `--out-dir` | `richardzhang_work/top_submissions` | Output directory |
| `--service` | off | Long-lived mode with periodic fetching |
| `--interval` | `300` | Seconds between fetches in service mode |

**Env overrides:** `FETCH_SUBS_API_URL`, `FETCH_SUBS_TOP_N`, `FETCH_SUBS_INTERVAL`, `FETCH_SUBS_LOG_FILE`

**Smart skipping:** If the submission ID for a given rank hasn't changed since last fetch, it skips re-downloading details and code.

**Recent submission matching:** When `--recent` is used, the script fetches your UIDs' recent submissions, matches them to `improved/submissions.json` entries by `gist_url` (extracted from the submission's `code_hash`), and updates each entry's `submission_id`, `status`, and `mfu`. Entries with status already `finished` are not updated. `last_checked` is only updated when `status` changes.

**Output:**

```
top_submissions/
  leaderboard_20260209_195055.json   # timestamped leaderboard snapshots
  rank01_v3_commit_7501050_79/
    stats.json                        # rank, detail, evaluations
    train.py                          # miner's source code
  rank02_.../
  ...
```

---

## 2. `check_gaming.py` — AI gaming detection

Scans new top-N submissions and uses a Cursor Background Agent to classify each as **gaming** (YES) or **honest** (NO). Includes your manual notes from `gaming_notes.md` in the prompt.

```bash
# One-shot (uses .env for CURSOR_API_KEY and CURSOR_REPO)
uv run python richardzhang_work/check_gaming.py

# With specific model
uv run python richardzhang_work/check_gaming.py --model claude-4.6-opus-high-thinking

# Service mode
uv run python richardzhang_work/check_gaming.py --service --interval 300
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--top N` | `5` | Consider top N submissions |
| `--model` | `claude-4.6-opus-high-thinking` | Cursor agent model |
| `--service` | off | Long-lived mode |
| `--interval` | `300` | Seconds between checks |

**Env overrides:** `GAMING_CHECK_INTERVAL`, `GAMING_CHECK_TOP_N`, `GAMING_CHECK_LOG_FILE`

**How it knows what's new:** Maintains `gaming_checks/state.json` with all previously checked submission IDs. Only unchecked submissions trigger an agent call.

**Batch processing:** All new submissions are sent in a single agent call to reduce overhead. Verdicts are parsed from the structured agent reply.

**Manual notes:** Edit `richardzhang_work/gaming_notes.md` with your observations about gaming patterns. The file is included verbatim in the agent prompt.

**Output:**

```
gaming_checks/
  state.json                          # checked IDs + all verdicts
  v3_commit_7501050_79.json           # per-submission: verdict, reason, full_reply
  batch_reply_20260209_213041.txt     # raw agent reply
```

---

## 3. `improve_and_submit.py` — AI code generation + auto-submission

Analyzes the #1 honest submission (from gaming check results) and generates an improved `train.py` using a Cursor Background Agent. Optionally submits via GitHub Gist and `neurons.miner submit`.

```bash
# Generate only (no submission)
uv run python richardzhang_work/improve_and_submit.py

# Generate + submit
uv run python richardzhang_work/improve_and_submit.py --submit

# Service mode (every 1 hour)
uv run python richardzhang_work/improve_and_submit.py --submit --service
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--submit` | off | Auto-submit via gist + `neurons.miner submit` |
| `--model` | `claude-4.6-opus-high-thinking` | Cursor agent model |
| `--service` | off | Long-lived mode |
| `--interval` | `3600` | Seconds between runs in service mode |

**Env overrides:** `IMPROVE_INTERVAL`, `IMPROVE_LOG_FILE`, `IMPROVE_NO_COMMENT`, `IMPROVE_POLICY`

**Improvement policy** (`richardzhang_work/improve_config.json`):

| Key | Default | Description |
|-----|---------|-------------|
| `no_comment` | `true` | If true, generated code must have no comments; if false, comments allowed. |
| `improvement_policy` | `"circular"` | How much to change vs the #1 submission. |

| `improvement_policy` | Behavior |
|----------------------|----------|
| `circular` (default) | Cycle copycat → minor → major → copycat … each run uses the next in the cycle (state in `improved/policy_cycle.json`). You can manually set `"index"` to 0, 1, or 2 to choose the next policy (0=copycat, 1=minor, 2=major); no other state depends on it. |
| `copycat` | Exactly copy the top submission; only change variable names (no logic change). |
| `minor` | 1 or 2 small modifications from the top submission. |
| `major` | One significant change that the AI recommends (e.g. different optimization or structure). |

Override via env: `IMPROVE_NO_COMMENT=true|false`, `IMPROVE_POLICY=copycat|minor|major|circular`.

For **minor** and **major**, the script records what changed vs the top submission (a unified diff) in `improved/last_applied_changes.json`. On the next run, that diff is included in the prompt so the agent can avoid repeating the same edits or build on them. **Copycat** does not use or update this.

### Key behaviors

**Wallet rotation:** With `--submit`, the script reads `WALLETS` (JSON array with `name`, `hotkey`, `uid`). It picks the wallet with the longest time since its last submission, requiring at least 1.2 hours (4320 seconds) cooldown. If no wallet is ready, the entire cycle is skipped (no agent call).

**Gist revisioning:** All submissions use a single shared Gist (ID stored in `improved/gist.json`). Each submission creates a new revision of this Gist, and the revision-specific raw URL is submitted to the chain.

**Previous result feedback:** The prompt includes the most recent *evaluated* submission from `improved/submissions.json` (walks backwards to find one with status `finished` or `failed_*`). If the previous submission failed, the prompt instructs the agent to fix issues and be more conservative. If it succeeded, the prompt asks to beat its MFU.

**Last run record:** Saves `{top_sid, policy}` to `improved/last_gen_inputs.json` after each generation (reference only; not used for skip logic).

**Code retrieval:** The prompt tells the agent to write code to `richardzhang_work/improved/train_agent_output.py` on its branch. The script fetches this file from the agent's branch via GitHub API. Falls back to extracting code from the conversation reply.

**Validation gate:** If the generated code doesn't contain `def inner_steps`, it's saved as `failed_reply_<timestamp>.txt` and submission is aborted.

**Output:**

```
improved/
  train_20260209_232843.py     # timestamped generated code
  train_latest.py              # always the most recent generated code
  agent_reply_20260209_232843.txt  # full agent reply for reference
  failed_reply_20260209_231131.txt # saved when validation fails
  submissions.json             # all submission attempts (success + failure)
  wallet_history.json          # last submission time per wallet
  gist.json                    # shared gist ID
  last_gen_inputs.json         # last run: {top_sid, policy} (policy = copycat|minor|major)
  last_applied_changes.json    # (minor/major) last diff vs top submission, used in next prompt
```

**`submissions.json` entry format:**

```json
{
  "wallet_name": "mycoldkey1",
  "wallet_hotkey": "myhotkey1",
  "uid": 42,
  "gist_url": "https://gist.githubusercontent.com/.../train.py",
  "code_file": "train_20260209_232843.py",
  "policy": "minor",
  "top_sid": "v3_commit_7501050_79",
  "submit_status": "submitted",
  "submitted_at": "2026-02-09T23:29:29.276964",
  "submission_id": "v3_commit_7509610_241",
  "status": "finished",
  "mfu": 45.82,
  "last_checked": "2026-02-10T00:24:12.810913"
}
```

`policy` = improvement policy used (copycat | minor | major). `top_sid` = #1 honest submission we improved from.

---

## 4. `sync_upstream.py` — upstream merge + dependency sync

Fetches upstream changes (from `one-covenant/crusades`), merges into your `main` branch, runs `uv sync` to update dependencies, and optionally pushes.

```bash
# Fetch + merge (no push)
uv run python richardzhang_work/sync_upstream.py

# Merge + push
uv run python richardzhang_work/sync_upstream.py --push

# Dry run (only report)
uv run python richardzhang_work/sync_upstream.py --dry-run

# Service mode (sync every 60s)
uv run python richardzhang_work/sync_upstream.py --service
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--push` | off | Push `main` to `origin` after merge |
| `--dry-run` | off | Only fetch and report; don't merge |
| `--service` | off | Long-lived mode |
| `--interval` | `60` | Seconds between syncs in service mode |

**Env overrides:** `SYNC_UPSTREAM_INTERVAL`, `SYNC_UPSTREAM_LOG_FILE`, `SYNC_UPSTREAM_PUSH`

**Requires a clean working tree.** If there are uncommitted changes, the merge is skipped with an error message.

**After merge,** runs `uv sync` to update dependencies if `pyproject.toml` or `uv.lock` changed.

**Log:** `richardzhang_work/sync-upstream.log` — each merged commit's hash, author, date, and subject.

---

## 5. `run_all.py` — master orchestrator

Runs all services in sequence in an infinite loop:

1. `fetch_top_submissions.py --recent` — fetch leaderboard + update eval results
2. `check_gaming.py` — classify new submissions as gaming/honest
3. `improve_and_submit.py [--submit]` — generate improvement + submit
4. `git add/commit/push` — push `richardzhang_work/` changes
5. `sync_upstream.py --push` — merge upstream changes

```bash
# Full automation
uv run python richardzhang_work/run_all.py --submit

# Single cycle
uv run python richardzhang_work/run_all.py --submit --once

# Without submission
uv run python richardzhang_work/run_all.py --once

# Without git push
uv run python richardzhang_work/run_all.py --once --no-push

# Custom interval (30 min between cycles)
uv run python richardzhang_work/run_all.py --submit --interval 1800
```

**Graceful shutdown:** Ctrl+C stops the loop after the current step finishes.

**Logging:** All subprocess output is captured and logged with timestamps.

---

## Systemd services (optional)

Each script can also be installed as an independent systemd service. This is an alternative to `run_all.py` for running scripts individually.

| Script | Service name | Setup script |
|--------|-------------|-------------|
| `sync_upstream.py` | `crusades-sync-upstream` | `setup-sync-upstream-service.sh` |
| `fetch_top_submissions.py` | `crusades-fetch-submissions` | `setup-fetch-submissions-service.sh` |
| `check_gaming.py` | `crusades-check-gaming` | `setup-check-gaming-service.sh` |
| `improve_and_submit.py` | `crusades-improve` | `setup-improve-service.sh` |

**Install any service:**

```bash
# Example: install the fetch-submissions service
chmod +x richardzhang_work/setup-fetch-submissions-service.sh
./richardzhang_work/setup-fetch-submissions-service.sh
```

**Common systemctl commands:**

```bash
sudo systemctl status <service-name>
sudo systemctl start <service-name>
sudo systemctl stop <service-name>
sudo systemctl restart <service-name>
sudo journalctl -u <service-name> -f
```

**Prerequisite:** [uv](https://docs.astral.sh/uv/) must be installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

---

## Directory structure

```
richardzhang_work/
  .env                          # secrets (not in git)
  README.md                     # this file
  improve_config.json           # no_comment, improvement_policy (copycat | minor | major | circular)
  gaming_notes.md               # manual notes about gaming patterns (included in AI prompt)

  # Python scripts
  sync_upstream.py              # upstream sync
  fetch_top_submissions.py      # leaderboard + code fetcher
  check_gaming.py               # AI gaming detection
  improve_and_submit.py         # AI code generation + submission
  run_all.py                    # master orchestrator

  # Systemd service files
  crusades-sync-upstream.service
  crusades-fetch-submissions.service
  crusades-check-gaming.service
  crusades-improve.service

  # Setup scripts (install systemd services)
  setup-sync-upstream-service.sh
  setup-fetch-submissions-service.sh
  setup-check-gaming-service.sh
  setup-improve-service.sh

  # Data directories (generated at runtime)
  top_submissions/              # leaderboard snapshots + ranked submission code
    leaderboard_*.json
    rank01_<sid>/stats.json, train.py
    ...
  gaming_checks/                # AI verdicts on submissions
    state.json
    <sid>.json
    batch_reply_*.txt
  improved/                     # generated code + submission history
    train_*.py                  # timestamped generated code files
    train_latest.py             # most recent generated code
    agent_reply_*.txt           # full agent replies
    failed_reply_*.txt          # saved when validation fails
    submissions.json            # all submission attempts + eval results
    wallet_history.json         # cooldown tracking per wallet
    gist.json                   # shared gist ID
    last_gen_inputs.json        # last run inputs (reference); top_attempts.json = attempt counts
```

---

## Quick reference

| Goal | Command |
|------|---------|
| **Run everything (loop)** | `uv run python richardzhang_work/run_all.py` |
| **Run everything (once)** | `uv run python richardzhang_work/run_all.py --once` |
| **Fetch leaderboard** | `uv run python richardzhang_work/fetch_top_submissions.py --top 5` |
| **Fetch + update my results** | `uv run python richardzhang_work/fetch_top_submissions.py --top 5 --recent` |
| **Check for gaming** | `uv run python richardzhang_work/check_gaming.py` |
| **Generate improvement** | `uv run python richardzhang_work/improve_and_submit.py` |
| **Generate + submit** | `uv run python richardzhang_work/improve_and_submit.py --submit` |
| **Sync upstream** | `uv run python richardzhang_work/sync_upstream.py --push` |
| **Reset attempt count for same top** | Edit or delete `improved/top_attempts.json` (e.g. remove or lower the entry for that top_sid) |
| **Force re-check gaming** | Delete `gaming_checks/state.json`, then run check_gaming |
| **View submission history** | `cat richardzhang_work/improved/submissions.json` |
| **View gaming verdicts** | `cat richardzhang_work/gaming_checks/state.json` |
| **View wallet cooldowns** | `cat richardzhang_work/improved/wallet_history.json` |
