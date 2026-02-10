#!/usr/bin/env python3
"""
Analyze top honest miners' submissions, generate an improved version using
Cursor's Background Agent API, and optionally submit it.

Reads gaming_checks/state.json for verdicts, filters to honest (NO) submissions,
sends their code to a Cursor agent to synthesize an improved train.py, saves
the result, and optionally submits via GitHub Gist + neurons.miner.

Requires:
  CURSOR_API_KEY  — Cursor API token (key-...) from cursor.com/dashboard/integration
  CURSOR_REPO     — GitHub repo URL (e.g. https://github.com/tkius123/crusades)

For auto-submit (--submit):
  GITHUB_TOKEN    — GitHub PAT with gist scope (to create a secret gist)
  WALLETS         — JSON list of wallets, e.g. '[{"name":"w1","hotkey":"h1","uid":42},{"name":"w2","hotkey":"h2","uid":241}]'
                    The script picks the wallet with the oldest last-submission time (cooldown-based).
  NETWORK         — Bittensor network (default: "finney")

Usage:
  uv run python richardzhang_work/improve_and_submit.py                # analyze + generate
  uv run python richardzhang_work/improve_and_submit.py --submit       # + auto-submit
  uv run python richardzhang_work/improve_and_submit.py --service      # run periodically

Config (richardzhang_work/improve_config.json):
  no_comment          — true (default) = output must have no comments; false = comments allowed
  improvement_policy — "circular" (default) | "copycat" | "minor" | "major"
    circular: cycle copycat → minor → major each run
    copycat:  exactly copy top submission, only change variable names
    minor:    1 or 2 modifications from the top submission
    major:    one significant change that the AI recommends

Env overrides:
  IMPROVE_INTERVAL   — seconds between runs in --service (default: 3600)
  IMPROVE_LOG_FILE   — log file path
  IMPROVE_NO_COMMENT — true | false (overrides no_comment from config)
  IMPROVE_POLICY     — circular | copycat | minor | major (overrides improvement_policy)
"""

import argparse
import base64
import difflib
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

CURSOR_API_BASE = "https://api.cursor.com/v0"
DEFAULT_MODEL = "claude-4.6-opus-high-thinking"
DEFAULT_INTERVAL_SEC = 3600  # 1 hour
WALLET_COOLDOWN_SEC = 4320  # 1.2 hours = 72 minutes
AGENT_POLL_INTERVAL = 5
AGENT_POLL_MAX = 180  # ~15 min (generation is more complex)

IMPROVE_PROMPT = """Write the complete improved train.py code to the file: richardzhang_work/improved/train_agent_output.py
Do NOT modify any other files. Do NOT run any commands. Only write the one file above.

You are an expert PyTorch performance engineer competing in the Templar Crusades MFU benchmark.

RULES (must follow — violations cause disqualification):
- Must implement inner_steps(model, data_iterator, optimizer, num_steps, device) -> InnerStepsResult
- Must use the PROVIDED optimizer directly: call optimizer.step() and optimizer.zero_grad() every step
- MUST NOT access optimizer.optimizer or any internal attributes
- MUST NOT detect or bypass any optimizer wrapper (e.g. checking for captured_gradients)
- Must process ALL tokens in each batch (no truncation)
- Must return actual final_logits tensor (not None)
- Must train all model parameters (don't freeze layers)
- Gradient relative error must be < 2% vs reference
- Loss must be close to reference (within 0.3)

ALLOWED optimizations (legitimate speed-ups):
- torch.compile with max-autotune
- CUDA stream prefetching (overlap data transfer with compute)
- Fused optimizer (via param_groups, not by accessing internals)
- TF32 matmul / cuDNN settings
- Flash/mem-efficient SDP backends
- Disabling gradient checkpointing (if VRAM allows)
- Freeing stale optimizer states via gc
- Any other optimization that doesn't change the math or bypass the optimizer

{strategy_section}

{applied_changes_section}
{notes_section}
=== Current #1 submission ===
{top_code}

Write the complete improved code to richardzhang_work/improved/train_agent_output.py.
{code_style_section}
"""


def script_dir() -> Path:
    return Path(__file__).resolve().parent


IMPROVE_CONFIG_DEFAULTS: dict[str, Any] = {
    "no_comment": True,
    "improvement_policy": "circular",
}

VALID_POLICIES = ("copycat", "minor", "major", "circular")
POLICY_CYCLE = ("copycat", "minor", "major")  # order for circular mode
POLICY_CYCLE_FILE = "policy_cycle.json"  # Safe to edit: set "index" to 0=copycat, 1=minor, 2=major for next run


def load_improve_config(work_dir: Path) -> dict[str, Any]:
    """Load improve_config.json from work_dir; env overrides: IMPROVE_NO_COMMENT, IMPROVE_POLICY."""
    config = dict(IMPROVE_CONFIG_DEFAULTS)
    config_path = work_dir / "improve_config.json"
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text())
            if isinstance(data.get("no_comment"), bool):
                config["no_comment"] = data["no_comment"]
            if data.get("improvement_policy") in VALID_POLICIES:
                config["improvement_policy"] = data["improvement_policy"]
        except (json.JSONDecodeError, OSError):
            pass
    env_no_comment = os.environ.get("IMPROVE_NO_COMMENT", "").strip().lower()
    if env_no_comment in ("1", "true", "yes"):
        config["no_comment"] = True
    elif env_no_comment in ("0", "false", "no"):
        config["no_comment"] = False
    env_policy = os.environ.get("IMPROVE_POLICY", "").strip().lower()
    if env_policy in VALID_POLICIES:
        config["improvement_policy"] = env_policy
    return config


def _load_policy_cycle_index(output_dir: Path) -> int:
    """Load current index for circular policy (0..2). Default 0."""
    path = output_dir / POLICY_CYCLE_FILE
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text())
        i = int(data.get("index", 0))
        if 0 <= i < len(POLICY_CYCLE):
            return i
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        pass
    return 0


def _save_policy_cycle_index(output_dir: Path, index: int) -> None:
    """Persist next cycle index for circular mode."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / POLICY_CYCLE_FILE).write_text(json.dumps({"index": index % len(POLICY_CYCLE)}, indent=2))


def resolve_effective_policy(config_policy: str, output_dir: Path) -> tuple[str, int]:
    """Resolve config policy to effective policy and next cycle index.
    For circular: effective = CYCLE[current_index], next_index = (current+1) % 3.
    Otherwise: effective = config_policy, next_index = 0.
    Returns (effective_policy, next_cycle_index).
    """
    if config_policy == "circular":
        idx = _load_policy_cycle_index(output_dir)
        effective = POLICY_CYCLE[idx]
        next_idx = (idx + 1) % len(POLICY_CYCLE)
        return effective, next_idx
    base = config_policy or "copycat"
    if base not in POLICY_CYCLE:
        base = "copycat"
    return base, 0


def build_strategy_section(policy: str, top_mfu_section: str) -> str:
    """Build the strategy block for the prompt based on effective improvement_policy."""
    if policy == "copycat":
        return f"""{top_mfu_section}
STRATEGY — COPYCAT:
- Exactly copy the #1 submission code below. Only change some variable names (e.g. rename locals or globals for clarity).
- Do NOT change any logic, control flow, or behavior. Output must be functionally identical to the top submission."""
    if policy == "minor":
        return f"""{top_mfu_section}
STRATEGY — MINOR (1 or 2 modifications only):
- Start by copying the ENTIRE #1 submission code below. Do not rewrite it.
- Then make only ONE or TWO small changes so we can see how each change affects MFU.
- Do not rewrite the training loop; only adjust settings or add one optimization at a time.
- Your improved code must match or exceed the #1 submission's MFU while passing all verification rules."""
    if policy == "major":
        return f"""{top_mfu_section}
STRATEGY — MAJOR (significant change):
- Use the #1 submission as the base. You may introduce one significant change that you recommend (e.g. different compile mode, optimization, or structure).
- All RULES must still be satisfied. The code must be expected to match or exceed the #1 submission's MFU.
- Prefer a single, well-motivated change over many small tweaks."""
    return f"{top_mfu_section}\nSTRATEGY: Use the #1 submission as the base and improve within the rules above."


def build_code_style_section(config: dict[str, Any]) -> str:
    """Build the code style line (no comments vs comments allowed)."""
    if config.get("no_comment", True):
        return "The code must have no comments. Only pure Python."
    return "Comments are allowed if they help explain non-obvious choices."


LAST_APPLIED_CHANGES_FILE = "last_applied_changes.json"
APPLIED_DIFF_MAX_LEN = 8000


def _compute_diff(top_code: str, improved_code: str) -> str:
    """Return a unified diff of improved_code vs top_code (what was applied)."""
    top_lines = top_code.splitlines(keepends=True)
    improved_lines = improved_code.splitlines(keepends=True)
    diff = list(difflib.unified_diff(top_lines, improved_lines, fromfile="top", tofile="improved", lineterm=""))
    return "\n".join(diff)


def save_last_applied_changes(output_dir: Path, top_sid: str, top_code: str, improved_code: str, generated_file: str) -> None:
    """Record what changes were applied to the top submission (for minor/major)."""
    diff = _compute_diff(top_code, improved_code)
    if len(diff) > APPLIED_DIFF_MAX_LEN:
        diff = diff[:APPLIED_DIFF_MAX_LEN] + "\n... (truncated)"
    data = {
        "top_sid": top_sid,
        "generated_file": generated_file,
        "applied_diff": diff,
        "saved_at": datetime.now().isoformat(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / LAST_APPLIED_CHANGES_FILE).write_text(json.dumps(data, indent=2))


def load_applied_changes_section(output_dir: Path, current_top_sid: str) -> str:
    """Load last applied changes and return prompt section if top_sid matches; else empty."""
    path = output_dir / LAST_APPLIED_CHANGES_FILE
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text())
        if data.get("top_sid") != current_top_sid:
            return ""
        diff = data.get("applied_diff", "")
        if not diff:
            return ""
        return (
            "Last time you applied the following changes to the top submission (unified diff: top -> improved):\n"
            "```\n" + diff + "\n```\n"
            "Use this to avoid repeating the same edits or to build on them.\n\n"
        )
    except (json.JSONDecodeError, OSError):
        return ""


def load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def log(msg: str, file: Path | None = None) -> None:
    line = f"[{datetime.now().isoformat()}] {msg}"
    print(line, flush=True)
    if file:
        with open(file, "a") as f:
            f.write(line + "\n")


# ── Cursor Agent API ─────────────────────────────────────────

def auth_headers(api_key: str) -> dict[str, str]:
    b64 = base64.b64encode(f"{api_key}:".encode()).decode()
    return {"Authorization": f"Basic {b64}"}


AGENT_OUTPUT_FILE = "richardzhang_work/improved/train_agent_output.py"


def call_cursor_agent(api_key: str, repo: str, prompt: str, model: str | None = None) -> str | None:
    """Launch agent, let it write to AGENT_OUTPUT_FILE, then fetch from its branch via GitHub API."""
    body: dict[str, Any] = {
        "prompt": {"text": prompt},
        "source": {"repository": repo, "ref": "main"},
        "target": {"autoCreatePr": False},
    }
    if model and model.lower() != "auto":
        body["model"] = model
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            f"{CURSOR_API_BASE}/agents",
            headers={"Content-Type": "application/json", **auth_headers(api_key)},
            json=body,
        )
        if resp.status_code not in (200, 201):
            print(f"Agent launch failed ({resp.status_code}): {resp.text}", flush=True)
            return None
        launch_data = resp.json()
        agent_id = launch_data["id"]
        branch_name = launch_data.get("target", {}).get("branchName", "")

        for _ in range(AGENT_POLL_MAX):
            time.sleep(AGENT_POLL_INTERVAL)
            status = client.get(
                f"{CURSOR_API_BASE}/agents/{agent_id}",
                headers=auth_headers(api_key),
            ).json().get("status", "UNKNOWN")
            if status in ("FINISHED", "FAILED"):
                break
        else:
            return None

        if status != "FINISHED":
            return None

        # Read the file from the agent's branch via GitHub API
        if branch_name:
            code = _fetch_file_from_branch(client, repo, branch_name, AGENT_OUTPUT_FILE)
            if code:
                return code

        # Fallback: try to extract code from conversation (reply should be short; we prefer file)
        conv = client.get(
            f"{CURSOR_API_BASE}/agents/{agent_id}/conversation",
            headers=auth_headers(api_key),
        ).json()
        for m in reversed(conv.get("messages", [])):
            if m.get("type") == "assistant_message" and m.get("text"):
                text = m["text"].strip()
                # If reply looks like a short ack (no code), don't use it as code
                if len(text) < 400 or "def " not in text:
                    return None
                return m["text"]
    return None


def _fetch_file_from_branch(client: httpx.Client, repo_url: str, branch: str, file_path: str) -> str | None:
    """Fetch a file from a GitHub branch. repo_url is like https://github.com/owner/repo."""
    # Parse owner/repo from URL
    parts = repo_url.rstrip("/").split("/")
    if len(parts) < 2:
        return None
    owner, repo_name = parts[-2], parts[-1]
    github_token = os.environ.get("GITHUB_TOKEN", "")
    headers = {"Accept": "application/vnd.github.raw+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    resp = client.get(
        f"https://api.github.com/repos/{owner}/{repo_name}/contents/{file_path}?ref={branch}",
        headers=headers,
        timeout=15,
    )
    if resp.status_code == 200:
        return resp.text
    return None


# ── Helpers ──────────────────────────────────────────────────

def _find_latest_evaluated(output_dir: Path) -> dict | None:
    """Walk submissions.json backwards to find the most recent entry with an eval result."""
    submissions_path = output_dir / "submissions.json"
    if not submissions_path.exists():
        return None
    try:
        submissions = json.loads(submissions_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    for entry in reversed(submissions):
        status = entry.get("status", "")
        if status and status not in ("submitted", "unknown", ""):
            return entry
    return None


def get_previous_result_section(output_dir: Path) -> str:
    """Navigate backwards through submissions to find the most recent one with an eval result."""
    entry = _find_latest_evaluated(output_dir)
    if entry is None:
        return ""

    status = entry.get("status", "")
    uid = entry.get("uid", "?")
    mfu = entry.get("mfu")
    code_file = entry.get("code_file", "")
    error = entry.get("error", "")

    lines = [f"IMPORTANT — Your most recent evaluated submission (UID {uid}):"]
    lines.append(f"  Status: {status}")
    if mfu is not None:
        lines.append(f"  MFU: {mfu}")
    if error:
        lines.append(f"  Error: {error}")

    if "failed" in status.lower():
        lines.append("")
        lines.append("The previous code FAILED evaluation. You MUST fix the issues.")
        lines.append("Common failure reasons: gradient error > 2%, loss too different from reference,")
        lines.append("returning None for final_logits, wrong token count, or crashing.")
        lines.append("Make the new version more conservative — prioritize correctness over speed.")
        if code_file:
            code_path = output_dir / code_file
            if code_path.exists():
                failed_code = code_path.read_text()
                lines.append("")
                lines.append(f"Previous failed code ({code_file}):")
                lines.append(failed_code)
    elif status == "finished" and mfu is not None:
        lines.append("")
        lines.append(f"The previous code succeeded with MFU={mfu}. Try to beat it while keeping correctness.")
        if code_file:
            code_path = output_dir / code_file
            if code_path.exists():
                success_code = code_path.read_text()
                lines.append("")
                lines.append(f"Previous successful code ({code_file}):")
                lines.append(success_code)

    return "\n".join(lines) + "\n"


def _get_latest_eval_sid(output_dir: Path) -> str | None:
    """Get identifier of the most recent submission with an eval result (walks backwards)."""
    entry = _find_latest_evaluated(output_dir)
    if entry is None:
        return None
    return entry.get("submission_id") or entry.get("code_file")


def _load_last_gen_inputs(output_dir: Path) -> dict | None:
    path = output_dir / "last_gen_inputs.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _save_last_gen_inputs(output_dir: Path, inputs: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "last_gen_inputs.json").write_text(json.dumps(inputs, indent=2))


def load_notes(notes_path: Path) -> str:
    if not notes_path.exists():
        return ""
    text = notes_path.read_text().strip()
    if not text:
        return ""
    return f"The reviewer has left the following notes:\n{text}\n"


def _get_top_submission_mfu(top_submissions_dir: Path, top_rank: int, top_sid: str) -> float | None:
    """Read #1 submission's MFU from its rank folder stats.json. Returns None if not found."""
    folder = top_submissions_dir / f"rank{top_rank:02d}_{top_sid}"
    stats_file = folder / "stats.json"
    if not stats_file.exists():
        return None
    try:
        data = json.loads(stats_file.read_text())
        entry = data.get("leaderboard_entry") or data.get("submission_detail") or {}
        score = entry.get("final_score")
        if score is not None:
            return float(score)
        return None
    except (json.JSONDecodeError, OSError, TypeError):
        return None


def _get_top_submission_uid(top_submissions_dir: Path, top_rank: int, top_sid: str) -> int | None:
    """Read #1 submission's miner_uid from its rank folder stats.json. Returns None if not found."""
    folder = top_submissions_dir / f"rank{top_rank:02d}_{top_sid}"
    stats_file = folder / "stats.json"
    if not stats_file.exists():
        return None
    try:
        data = json.loads(stats_file.read_text())
        entry = data.get("leaderboard_entry") or data.get("submission_detail") or {}
        uid = entry.get("miner_uid")
        if uid is not None:
            return int(uid)
        return None
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return None


def _get_our_uids(wallets: list[dict] | None) -> set[int]:
    """Return set of our UIDs from wallets. If wallets is None, try parsing WALLETS env."""
    if wallets:
        return {int(w["uid"]) for w in wallets if w.get("uid") is not None}
    raw = os.environ.get("WALLETS")
    if not raw:
        return set()
    try:
        lst = json.loads(raw)
        if isinstance(lst, list):
            return {int(w["uid"]) for w in lst if isinstance(w, dict) and w.get("uid") is not None}
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return set()


TOP_ATTEMPTS_FILE = "top_attempts.json"
MAX_ATTEMPTS_PER_TOP = 6


def _load_top_attempts(output_dir: Path) -> dict[str, int]:
    """Load {top_sid: attempt_count} from top_attempts.json."""
    path = output_dir / TOP_ATTEMPTS_FILE
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return {k: int(v) for k, v in data.items() if isinstance(v, (int, float))}
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        pass
    return {}


def _increment_top_attempt(output_dir: Path, top_sid: str) -> None:
    """Increment attempt count for this top_sid and persist."""
    counts = _load_top_attempts(output_dir)
    counts[top_sid] = counts.get(top_sid, 0) + 1
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / TOP_ATTEMPTS_FILE).write_text(json.dumps(counts, indent=2))


def get_honest_submissions(
    gaming_checks_dir: Path,
    top_submissions_dir: Path,
) -> list[tuple[int, str, str]]:
    """Return [(rank, sid, code)] for honest (verdict=NO) submissions, sorted by rank."""
    state_file = gaming_checks_dir / "state.json"
    if not state_file.exists():
        return []
    try:
        state = json.loads(state_file.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    results = state.get("results", {})
    honest = []
    for sid, info in results.items():
        if info.get("verdict", "").upper() != "NO":
            continue
        rank = info.get("rank", 99)
        folder = top_submissions_dir / f"rank{rank:02d}_{sid}"
        train_py = folder / "train.py"
        if train_py.exists():
            honest.append((rank, sid, train_py.read_text()))
    honest.sort(key=lambda x: x[0])
    return honest


def extract_code(reply: str) -> str:
    """Extract Python code from agent reply. Handles markdown fences, leading commentary, etc."""
    import re as _re
    text = reply.strip()

    # Try to extract from markdown code fence first (```python ... ```)
    m = _re.search(r"```(?:python)?\s*\n(.*?)```", text, _re.DOTALL)
    if m:
        return m.group(1).strip()

    # If the reply starts with commentary (not Python), find the first import/from/def/class line
    lines = text.split("\n")
    first_code_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ", "def ", "class ", '"""', "# ", "@dataclass")):
            first_code_idx = i
            break
    if first_code_idx is not None and first_code_idx > 0:
        return "\n".join(lines[first_code_idx:]).strip()

    return text


# ── Wallet selection (cooldown-based) ─────────────────────────

def load_wallet_history(output_dir: Path) -> dict[str, str]:
    """Load {wallet_key: last_submission_iso} from wallet_history.json."""
    hist_file = output_dir / "wallet_history.json"
    if not hist_file.exists():
        return {}
    try:
        return json.loads(hist_file.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_wallet_history(output_dir: Path, history: dict[str, str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "wallet_history.json").write_text(json.dumps(history, indent=2))


def wallet_key(w: dict) -> str:
    return f"{w['name']}/{w['hotkey']}"


def wallet_label(w: dict) -> str:
    uid = w.get("uid", "?")
    return f"{w['name']}/{w['hotkey']} (UID {uid})"


def pick_wallet(wallets: list[dict], output_dir: Path, cooldown_sec: int = WALLET_COOLDOWN_SEC) -> dict | None:
    """Pick a wallet that has cooled down (>= cooldown_sec since last submission).
    Among eligible wallets, picks the one with the oldest submission.
    Returns None if no wallet is ready."""
    if not wallets:
        return None
    history = load_wallet_history(output_dir)
    now = datetime.now()
    eligible = []
    for w in wallets:
        last_sub = history.get(wallet_key(w))
        if not last_sub:
            eligible.append((w, "1970-01-01T00:00:00"))  # never submitted
            continue
        try:
            last_dt = datetime.fromisoformat(last_sub)
            elapsed = (now - last_dt).total_seconds()
            if elapsed >= cooldown_sec:
                eligible.append((w, last_sub))
        except (ValueError, TypeError):
            eligible.append((w, "1970-01-01T00:00:00"))
    if not eligible:
        return None
    # Pick the one with the oldest submission
    eligible.sort(key=lambda x: x[1])
    return eligible[0][0]


def record_submission(
    output_dir: Path,
    wallet: dict,
    gist_url: str = "",
    code_file: str = "",
    submit_status: str = "submitted",
    error: str = "",
    policy: str = "",
) -> None:
    """Record a submission attempt (success or failure) in cooldown history + submissions log."""
    now = datetime.now().isoformat()
    # Update cooldown history (even on failure — we used the wallet's turn)
    history = load_wallet_history(output_dir)
    history[wallet_key(wallet)] = now
    save_wallet_history(output_dir, history)
    # Append to submissions log
    submissions_file = output_dir / "submissions.json"
    submissions: list[dict] = []
    if submissions_file.exists():
        try:
            submissions = json.loads(submissions_file.read_text())
        except (json.JSONDecodeError, OSError):
            submissions = []
    entry: dict = {
        "wallet_name": wallet["name"],
        "wallet_hotkey": wallet["hotkey"],
        "uid": wallet.get("uid"),
        "gist_url": gist_url,
        "code_file": code_file,
        "submit_status": submit_status,
        "submitted_at": now,
    }
    if policy:
        entry["policy"] = policy
    if error:
        entry["error"] = error
    submissions.append(entry)
    submissions_file.write_text(json.dumps(submissions, indent=2))


def load_gist_id(output_dir: Path) -> str | None:
    """Load saved gist ID from gist.json."""
    gist_file = output_dir / "gist.json"
    if not gist_file.exists():
        return None
    try:
        data = json.loads(gist_file.read_text())
        return data.get("gist_id")
    except (json.JSONDecodeError, OSError):
        return None


def save_gist_id(output_dir: Path, gist_id: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gist.json").write_text(json.dumps({"gist_id": gist_id}, indent=2))


def _github_headers(github_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
    }


def _get_revision_raw_url(data: dict) -> str | None:
    """Extract the raw URL for train.py from a gist API response.
    The raw_url includes a revision hash, so it points to this exact version."""
    files = data.get("files", {})
    train_file = files.get("train.py", {})
    return train_file.get("raw_url")


def update_or_create_gist(
    github_token: str,
    code: str,
    output_dir: Path,
    description: str = "Crusades train.py",
) -> str | None:
    """Update the shared gist (new revision) or create it on first run.
    Returns the revision-specific raw URL for train.py, or None on error."""
    gist_id = load_gist_id(output_dir)

    with httpx.Client() as client:
        if gist_id:
            # Update existing gist → new revision
            resp = client.patch(
                f"https://api.github.com/gists/{gist_id}",
                headers=_github_headers(github_token),
                json={
                    "description": description,
                    "files": {"train.py": {"content": code}},
                },
                timeout=15,
            )
            if resp.status_code == 200:
                return _get_revision_raw_url(resp.json())
            # If gist was deleted, fall through to create
            if resp.status_code != 404:
                return None

        # Create new gist
        resp = client.post(
            "https://api.github.com/gists",
            headers=_github_headers(github_token),
            json={
                "description": description,
                "public": False,
                "files": {"train.py": {"content": code}},
            },
            timeout=15,
        )
        if resp.status_code != 201:
            return None
        data = resp.json()
        new_id = data.get("id")
        if new_id:
            save_gist_id(output_dir, new_id)
        return _get_revision_raw_url(data)


# ── Main logic ───────────────────────────────────────────────

def run_improve(
    top_submissions_dir: Path,
    gaming_checks_dir: Path,
    output_dir: Path,
    api_key: str,
    repo: str,
    log_path: Path | None,
    notes_path: Path | None = None,
    model: str | None = None,
    submit: bool = False,
    wallets: list[dict] | None = None,
    network: str = "finney",
) -> int:
    log("Analyzing honest submissions and generating improved version...", log_path)

    # If submitting, check wallet availability BEFORE spending time on generation
    if submit:
        if not wallets:
            log("ERROR: --submit requires WALLETS env var.", log_path)
            return 1
        available = pick_wallet(wallets, output_dir)
        if available is None:
            log(f"Skip reason: no wallet available (all submitted within the last {WALLET_COOLDOWN_SEC}s / {WALLET_COOLDOWN_SEC / 3600:.1f}h).", log_path)
            return 0

    honest = get_honest_submissions(gaming_checks_dir, top_submissions_dir)
    if not honest:
        log("No honest submissions found. Run check_gaming.py first.", log_path)
        return 1

    # Use only the top 1 honest submission (smallest rank)
    top_rank, top_sid, top_code = honest[0]
    log(f"Using #1 honest submission: {top_sid} (rank {top_rank})", log_path)

    # Skip if the top submission is ours (nothing to improve from)
    top_uid = _get_top_submission_uid(top_submissions_dir, top_rank, top_sid)
    our_uids = _get_our_uids(wallets)
    if top_uid is not None and our_uids and top_uid in our_uids:
        log(f"Skip reason: top submission is ours (UID {top_uid}).", log_path)
        return 0

    # Skip if we've already tried 6+ times for this same top submission
    attempts = _load_top_attempts(output_dir)
    if attempts.get(top_sid, 0) >= MAX_ATTEMPTS_PER_TOP:
        log(f"Skip reason: already tried {attempts.get(top_sid)} times for top_sid={top_sid} (max={MAX_ATTEMPTS_PER_TOP}).", log_path)
        return 0

    # Top submission MFU (so agent knows the bar to match or exceed)
    top_mfu = _get_top_submission_mfu(top_submissions_dir, top_rank, top_sid)
    if top_mfu is not None:
        top_mfu_section = f"Current #1 submission achieved MFU: {top_mfu:.2f}%. Your improved code must match or exceed this while passing all verification rules."
        log(f"Top submission MFU: {top_mfu:.2f}%", log_path)
    else:
        top_mfu_section = "Current #1 submission (MFU not available in this run). Your improved code must pass all verification rules."

    # Improvement policy and code style from config
    work_dir = script_dir()
    improve_config = load_improve_config(work_dir)
    config_policy = improve_config.get("improvement_policy") or "circular"
    policy, next_cycle_index = resolve_effective_policy(config_policy, output_dir)
    cycle_idx = _load_policy_cycle_index(output_dir) if config_policy == "circular" else None
    strategy_section = build_strategy_section(policy, top_mfu_section)
    code_style_section = build_code_style_section(improve_config)
    log(f"Config: no_comment={improve_config.get('no_comment')}, policy={config_policy} -> effective={policy}", log_path)
    if config_policy == "circular" and cycle_idx is not None:
        log(f"Circular cycle index this run: {cycle_idx} -> policy={policy}", log_path)

    # Build prompt inputs (do not reference our own improvement submissions)
    notes_section = load_notes(notes_path) if notes_path else ""

    if policy in ("minor", "major"):
        applied_changes_section = load_applied_changes_section(output_dir, top_sid)
    else:
        applied_changes_section = ""

    current_inputs = {"top_sid": top_sid, "policy": policy}

    prompt = IMPROVE_PROMPT.format(
        strategy_section=strategy_section,
        applied_changes_section=applied_changes_section,
        notes_section=notes_section,
        top_code=top_code,
        code_style_section=code_style_section,
    )

    log(f"Launching Cursor agent to generate improved train.py (model={model or 'auto'})...", log_path)
    reply = call_cursor_agent(api_key, repo, prompt, model=model)
    if reply is None:
        log("Cursor agent error (FAILED/TIMEOUT).", log_path)
        return 1

    _increment_top_attempt(output_dir, top_sid)

    improved_code = extract_code(reply)

    # Validate basic structure
    if "def inner_steps" not in improved_code:
        log("ERROR: Generated code doesn't contain 'def inner_steps'. Skipping submit.", log_path)
        log("  Agent may have returned an explanation instead of code. Saving for inspection.", log_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        (output_dir / f"failed_reply_{ts}.txt").write_text(reply)
        return 1

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    code_path = output_dir / f"train_{ts}.py"
    code_path.write_text(improved_code)
    # Also save as latest
    (output_dir / "train_latest.py").write_text(improved_code)
    log(f"Saved improved code to {code_path}", log_path)

    if policy in ("minor", "major"):
        save_last_applied_changes(output_dir, top_sid, top_code, improved_code, code_path.name)

    if config_policy == "circular":
        _save_policy_cycle_index(output_dir, next_cycle_index)
        log(f"Circular: next run will use policy={POLICY_CYCLE[next_cycle_index]}", log_path)

    # Record what inputs were used for this generation
    _save_last_gen_inputs(output_dir, current_inputs)

    # Save full reply for reference
    (output_dir / f"agent_reply_{ts}.txt").write_text(reply)

    # Auto-submit
    if submit:
        github_token = os.environ.get("GITHUB_TOKEN")
        if not github_token:
            log("ERROR: --submit requires GITHUB_TOKEN env var (PAT with gist scope).", log_path)
            return 1

        # Pick wallet with longest cooldown (already verified availability above)
        wallet = pick_wallet(wallets, output_dir)
        if not wallet:
            log("Skip reason: no wallet available after generation (cooldown not met).", log_path)
            return 0

        wname = wallet["name"]
        whotkey = wallet["hotkey"]
        log(f"Selected wallet: {wallet_label(wallet)} (longest cooldown)", log_path)

        log("Updating gist (new revision)...", log_path)
        raw_url = update_or_create_gist(github_token, improved_code, output_dir, f"Crusades train.py {ts}")
        if not raw_url:
            log("ERROR: Failed to update/create gist.", log_path)
            return 1
        log(f"Gist revision URL: {raw_url}", log_path)

        log(f"Submitting to {network} (wallet={wname}/{whotkey})...", log_path)
        import subprocess
        result = subprocess.run(
            [
                "uv", "run", "-m", "neurons.miner", "submit", raw_url,
                "--wallet.name", wname,
                "--wallet.hotkey", whotkey,
                "--network", network,
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        if result.returncode == 0:
            record_submission(output_dir, wallet, gist_url=raw_url, code_file=code_path.name, submit_status="submitted")
            log("Submission successful!", log_path)
            log(result.stdout, log_path)
        else:
            err_msg = result.stderr.strip()[:500]
            record_submission(output_dir, wallet, gist_url=raw_url, code_file=code_path.name, submit_status="failed", error=err_msg, policy=policy)
            log(f"Submission failed (exit {result.returncode}): {err_msg}", log_path)
            return 1

    return 0


def main() -> int:
    load_dotenv(script_dir() / ".env")

    parser = argparse.ArgumentParser(
        description="Analyze top honest submissions, generate improved train.py, optionally submit."
    )
    parser.add_argument("--top-submissions-dir", type=Path, default=None)
    parser.add_argument("--gaming-checks-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to save improved code (default: richardzhang_work/improved)")
    parser.add_argument("--notes-file", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--submit", action="store_true", help="Auto-submit: create gist and call neurons.miner submit")
    parser.add_argument("--service", action="store_true", help="Run periodically")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SEC, metavar="SEC", help="Interval in seconds for --service (default: 3600)")
    args = parser.parse_args()

    if os.environ.get("IMPROVE_INTERVAL"):
        args.interval = int(os.environ["IMPROVE_INTERVAL"])
    if os.environ.get("IMPROVE_LOG_FILE"):
        args.log_file = Path(os.environ["IMPROVE_LOG_FILE"])

    sdir = script_dir()
    top_submissions_dir = args.top_submissions_dir or sdir / "top_submissions"
    gaming_checks_dir = args.gaming_checks_dir or sdir / "gaming_checks"
    output_dir = args.output_dir or sdir / "improved"
    notes_path = args.notes_file or sdir / "gaming_notes.md"
    log_path = args.log_file or sdir / "improve.log"

    api_key = os.environ.get("CURSOR_API_KEY")
    if not api_key:
        log("ERROR: Set CURSOR_API_KEY.", log_path)
        return 1
    repo = os.environ.get("CURSOR_REPO")
    if not repo:
        log("ERROR: Set CURSOR_REPO.", log_path)
        return 1

    # Parse wallets from WALLETS env (JSON list)
    wallets: list[dict] | None = None
    wallets_raw = os.environ.get("WALLETS")
    if wallets_raw:
        try:
            wallets = json.loads(wallets_raw)
            if not isinstance(wallets, list) or not all("name" in w and "hotkey" in w for w in wallets):
                log("ERROR: WALLETS must be a JSON list of {\"name\":..., \"hotkey\":...} objects.", log_path)
                return 1
        except json.JSONDecodeError:
            log("ERROR: WALLETS is not valid JSON.", log_path)
            return 1

    network = os.environ.get("NETWORK", "finney")

    run_kwargs = dict(
        top_submissions_dir=top_submissions_dir,
        gaming_checks_dir=gaming_checks_dir,
        output_dir=output_dir,
        api_key=api_key,
        repo=repo,
        log_path=log_path,
        notes_path=notes_path,
        model=args.model,
        submit=args.submit,
        wallets=wallets,
        network=network,
    )

    if args.service:
        shutdown = False

        def handler(signum: int, frame: object) -> None:
            nonlocal shutdown
            shutdown = True

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)
        log(f"Service started; improving every {args.interval}s.", log_path)
        while not shutdown:
            try:
                run_improve(**run_kwargs)
            except Exception as e:
                log(f"ERROR: {e}", log_path)
            if shutdown:
                break
            for _ in range(args.interval):
                if shutdown:
                    break
                time.sleep(1)
        log("Service stopped.", log_path)
        return 0

    return run_improve(**run_kwargs)


if __name__ == "__main__":
    sys.exit(main())
