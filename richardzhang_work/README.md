# richardzhang_work

Scripts and services for syncing upstream (one-covenant/crusades) into your fork's `main` branch and fetching top tournament submissions.

---

## `sync_upstream.py` — one-shot or service

**One-shot (run once, then exit):**

```bash
# From repo root
cd /root/workspace/crusades

# Fetch + merge into main (no push)
python richardzhang_work/sync_upstream.py

# Merge then push to origin
python richardzhang_work/sync_upstream.py --push

# Only show what would be merged (no merge, no push)
python richardzhang_work/sync_upstream.py --dry-run
```

**Service (run in foreground, sync every 60s until you stop it):**

```bash
cd /root/workspace/crusades

# Sync every 60 seconds (default)
python richardzhang_work/sync_upstream.py --service

# Sync every 2 minutes
python richardzhang_work/sync_upstream.py --service --interval 120
```

**Options:**

| Option        | Meaning |
|---------------|--------|
| `--push`      | After merging, push `main` to `origin`. |
| `--dry-run`   | Only fetch and report; do not merge or push. |
| `--service`   | Run as a long-lived process; sync every `--interval` seconds until SIGTERM/SIGINT. |
| `--interval N`| When using `--service`, sync every N seconds (default: 60). |
| `--log-file P`| Write log lines to path P (default: `richardzhang_work/sync-upstream.log`). |

**Where merges are logged:**
`richardzhang_work/sync-upstream.log` (or path given by `--log-file`). Each run appends lines like:

- `[timestamp] Found N new commit(s) on upstream/main.`
- `[timestamp]   <hash> <author> <date> <subject>` for each commit merged.

---

## `.service` file — run as a systemd service

The file `crusades-sync-upstream.service` is a **template**. You don't run it directly; you install it with the setup script below. After that, you use `systemctl` to control the service.

---

## Default config when run as a service (via `.sh`)

When you install and run the service with `./richardzhang_work/setup-sync-upstream-service.sh`, the unit file uses these defaults:

| Setting    | Default | Notes |
|-----------|---------|--------|
| **Interval** | `60` seconds | Sync runs every 60 seconds. Set in `.service` as `--interval 60`. |
| **Log path** | `richardzhang_work/sync-upstream.log` | Script default; not overridden in the unit. Path is relative to repo root. |
| **Push**     | Off | Merges stay local; no `--push` in the unit. To push after merge, add `--push` to `ExecStart` in the `.service` file. |
| **Working directory** | Repo root | Filled in as `REPLACE_WORKDIR` by the setup script. |
| **User**     | Current user | Filled in as `REPLACE_USER` when you run the setup script. |
| **Runner**   | `uv run python` | Requires [uv](https://docs.astral.sh/uv/) installed (e.g. in `~/.local/bin`). PATH in the unit includes `~/.local/bin`. |

**Changing config without editing code:** edit the **installed** unit file only (no changes to the Python script or the template in the repo).

**Option A — Environment variables (recommended)**
Edit `/etc/systemd/system/crusades-sync-upstream.service` and add or uncomment `Environment=` lines. The script reads these:

| Env var | Effect | Example |
|---------|--------|--------|
| `SYNC_UPSTREAM_INTERVAL` | Sync interval in seconds | `Environment="SYNC_UPSTREAM_INTERVAL=120"` |
| `SYNC_UPSTREAM_LOG_FILE` | Log file path | `Environment="SYNC_UPSTREAM_LOG_FILE=/var/log/crusades-sync.log"` |
| `SYNC_UPSTREAM_PUSH` | Push to origin after merge (set to `1`, `true`, or `yes`) | `Environment="SYNC_UPSTREAM_PUSH=1"` |

Then run:

```bash
sudo systemctl daemon-reload
sudo systemctl restart crusades-sync-upstream
```

**Option B — Edit ExecStart**
Change the command line in the unit (e.g. add `--interval 120`, `--log-file /path/to/log`, or `--push` to the `ExecStart` line), then `daemon-reload` and `restart` as above.

---

## `setup-sync-upstream-service.sh` — install and run the service

**Prerequisite:** [uv](https://docs.astral.sh/uv/) must be installed (e.g. `curl -LsSf https://astral.sh/uv/install.sh | sh`). The script checks for `uv` and exits if not found.

**One-time setup (installs the service and starts it):**

```bash
# From repo root; may need: chmod +x richardzhang_work/setup-sync-upstream-service.sh
./richardzhang_work/setup-sync-upstream-service.sh
```

This script:

1. Checks that `uv` is on PATH; copies the `.service` template to `/etc/systemd/system/crusades-sync-upstream.service` and fills in your user, repo path, and home (for `uv` in `~/.local/bin`).
2. Runs `systemctl daemon-reload`, `enable`, and `start` so the service runs now and after reboot.

**After setup, use these commands:**

| Command | Purpose |
|--------|--------|
| `sudo systemctl status crusades-sync-upstream` | See if the service is running. |
| `sudo systemctl start crusades-sync-upstream`  | Start the service. |
| `sudo systemctl stop crusades-sync-upstream`   | Stop the service. |
| `sudo systemctl restart crusades-sync-upstream`| Restart the service. |
| `sudo systemctl enable crusades-sync-upstream` | Start on boot (usually done by setup script). |
| `sudo systemctl disable crusades-sync-upstream`| Do not start on boot. |
| `sudo journalctl -u crusades-sync-upstream -f`  | Stream service logs (stdout/stderr). |

**Merge log (what was merged):**
`richardzhang_work/sync-upstream.log` (same as when running the script by hand).

---

## `fetch_top_submissions.py` — fetch top submissions from Crusades API

Connects to the [Crusades tournament API](https://www.tplr.ai/tournament) and downloads the top N submissions (leaderboard stats, evaluation details, and source code).

**One-shot:**

```bash
cd /root/workspace/crusades

# Fetch top 10 (default)
uv run python richardzhang_work/fetch_top_submissions.py

# Fetch top 5
uv run python richardzhang_work/fetch_top_submissions.py --top 5

# Custom API URL
uv run python richardzhang_work/fetch_top_submissions.py --api-url http://your-api:8080
```

**Service (fetch every 5 minutes by default):**

```bash
uv run python richardzhang_work/fetch_top_submissions.py --service
uv run python richardzhang_work/fetch_top_submissions.py --service --interval 120 --top 5
```

**Options:**

| Option        | Default | Meaning |
|---------------|---------|--------|
| `--api-url`   | `http://69.19.137.219:8080` | Crusades API base URL. |
| `--top N`     | `10` | Number of top submissions to fetch. |
| `--out-dir`   | `richardzhang_work/top_submissions` | Output directory. |
| `--log-file`  | `richardzhang_work/fetch-top-submissions.log` | Log file path. |
| `--service`   | off | Run as long-lived process; fetch every `--interval` seconds. |
| `--interval`  | `300` (5 min) | Fetch interval in seconds for `--service`. |

**Env overrides (for systemd, no code change):**

| Env var | Effect |
|---------|--------|
| `FETCH_SUBS_API_URL` | API base URL |
| `FETCH_SUBS_TOP_N` | Number of top submissions |
| `FETCH_SUBS_INTERVAL` | Interval in seconds |
| `FETCH_SUBS_LOG_FILE` | Log file path |

**What gets saved to `richardzhang_work/top_submissions/`:**

```
top_submissions/
  leaderboard.json           # full leaderboard snapshot
  rank01_<submission_id>/
    stats.json               # rank, leaderboard entry, full detail, evaluations
    train.py                 # miner's source code
  rank02_<submission_id>/
    stats.json
    train.py
  ...
```

**Install as systemd service:**

```bash
./richardzhang_work/setup-fetch-submissions-service.sh
```

| Command | Purpose |
|--------|--------|
| `sudo systemctl status crusades-fetch-submissions` | Service status |
| `sudo systemctl stop crusades-fetch-submissions` | Stop |
| `sudo systemctl restart crusades-fetch-submissions` | Restart |
| `sudo journalctl -u crusades-fetch-submissions -f` | Stream logs |

---

## `check_gaming.py` — classify new top-5 as gaming or not (Cursor Agent)

When new submissions appear in the top 5 (from `top_submissions/`), this script launches a **Cursor Background Agent** (`api.cursor.com/v0/agents`) to analyze the `train.py` and judge whether the code is **gaming** the benchmark or legitimate. Results are stored in `richardzhang_work/gaming_checks/`.

**Prerequisites:**

| Env var | Required | Description |
|---------|----------|-------------|
| `CURSOR_API_KEY` | yes | Cursor API token (`key-...`) from [cursor.com/dashboard/integration](https://cursor.com/dashboard/integration) |
| `CURSOR_REPO` | yes | GitHub repo URL for agent context (e.g. `https://github.com/tkius123/crusades`) |

**One-shot:**

```bash
cd /root/workspace/crusades
export CURSOR_API_KEY="key-..."
export CURSOR_REPO="https://github.com/tkius123/crusades"
uv run python richardzhang_work/check_gaming.py
```

**Service (check every 5 min):**

```bash
export CURSOR_API_KEY="key-..."
export CURSOR_REPO="https://github.com/tkius123/crusades"
uv run python richardzhang_work/check_gaming.py --service
uv run python richardzhang_work/check_gaming.py --service --interval 120 --top 5
```

**Options:**

| Option | Default | Meaning |
|--------|---------|--------|
| `--top N` | `5` | Consider top N submissions. |
| `--top-submissions-dir` | `richardzhang_work/top_submissions` | Where leaderboard and rank* folders live. |
| `--gaming-checks-dir` | `richardzhang_work/gaming_checks` | Where state and per-submission results are saved. |
| `--log-file` | `richardzhang_work/check-gaming.log` | Log file path. |
| `--service` | off | Run as long-lived process; check every `--interval` seconds. |
| `--interval` | `300` (5 min) | Check interval. |

**Env overrides:** `GAMING_CHECK_INTERVAL`, `GAMING_CHECK_TOP_N`, `GAMING_CHECK_LOG_FILE`.

**Output in `richardzhang_work/gaming_checks/`:**

- `state.json` — list of checked submission IDs and all results.
- `<submission_id>.json` — per-submission: `verdict` (YES/NO), `reason`, `full_reply`, `checked_at`, `rank`.

**Install as systemd service:**

1. Create an env file with your Cursor token and repo (do not commit it):
   ```bash
   sudo mkdir -p /etc/crusades
   printf 'CURSOR_API_KEY=key-...\nCURSOR_REPO=https://github.com/tkius123/crusades\n' | sudo tee /etc/crusades/gaming-check.env
   sudo chmod 600 /etc/crusades/gaming-check.env
   ```
2. Run the setup script (it will add `EnvironmentFile` if that file exists):
   ```bash
   ./richardzhang_work/setup-check-gaming-service.sh
   ```

| Command | Purpose |
|--------|--------|
| `sudo systemctl status crusades-check-gaming` | Service status |
| `sudo systemctl restart crusades-check-gaming` | Restart |
| `sudo journalctl -u crusades-check-gaming -f` | Stream logs |

---

## Quick reference

| Goal | Command |
|------|--------|
| Run sync once (no push) | `python richardzhang_work/sync_upstream.py` |
| Run sync once and push | `python richardzhang_work/sync_upstream.py --push` |
| See what would be merged | `python richardzhang_work/sync_upstream.py --dry-run` |
| Run sync as service (60s) | `python richardzhang_work/sync_upstream.py --service` |
| Install sync service | `./richardzhang_work/setup-sync-upstream-service.sh` |
| Sync service status | `sudo systemctl status crusades-sync-upstream` |
| Sync service logs | `sudo journalctl -u crusades-sync-upstream -f` |
| View merged commits log | `cat richardzhang_work/sync-upstream.log` |
| Fetch top submissions once | `uv run python richardzhang_work/fetch_top_submissions.py` |
| Fetch top 5 submissions | `uv run python richardzhang_work/fetch_top_submissions.py --top 5` |
| Run fetch as service (5 min) | `uv run python richardzhang_work/fetch_top_submissions.py --service` |
| Install fetch service | `./richardzhang_work/setup-fetch-submissions-service.sh` |
| Fetch service status | `sudo systemctl status crusades-fetch-submissions` |
| Fetch service logs | `sudo journalctl -u crusades-fetch-submissions -f` |
| View fetched submissions | `ls richardzhang_work/top_submissions/` |
| Run gaming check once | `CURSOR_API_KEY=key-... CURSOR_REPO=... uv run python richardzhang_work/check_gaming.py` |
| Run gaming check as service | `uv run python richardzhang_work/check_gaming.py --service` |
| Install gaming-check service | `./richardzhang_work/setup-check-gaming-service.sh` (set keys in /etc/crusades/gaming-check.env first) |
| Gaming check results | `ls richardzhang_work/gaming_checks/` |
