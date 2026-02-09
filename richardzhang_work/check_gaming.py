#!/usr/bin/env python3
"""
Check if new top N Crusades submissions are "gaming" the benchmark using
Cursor's Background Agent API.

Reads current top N from richardzhang_work/top_submissions/. All new (unchecked)
submissions are batched into a single Cursor agent call for speed, then verdicts
are parsed and persisted under richardzhang_work/gaming_checks/.

Requires:
  CURSOR_API_KEY  — Cursor API token (key-...) from cursor.com/dashboard/integration
  CURSOR_REPO     — GitHub repo URL for the agent source (e.g. https://github.com/tkius123/crusades)

Usage:
  uv run python richardzhang_work/check_gaming.py                                  # one-shot
  uv run python richardzhang_work/check_gaming.py --model claude-4-opus-thinking   # pick model
  uv run python richardzhang_work/check_gaming.py --service                        # run every 5 min

Notes file:
  richardzhang_work/gaming_notes.md — manual notes about gaming patterns; included in
  the prompt so the model can use your observations when judging submissions.

Env overrides:
  GAMING_CHECK_INTERVAL  — seconds between runs (default: 300)
  GAMING_CHECK_LOG_FILE  — log path
  GAMING_CHECK_TOP_N     — consider top N (default: 5)
"""

import argparse
import base64
import json
import os
import re
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx


def load_dotenv(env_path: Path) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (does not override existing)."""
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

CURSOR_API_BASE = "https://api.cursor.com/v0"
DEFAULT_MODEL = "claude-4.6-opus-high-thinking"  # Opus 4.6 max thinking; use "auto" to let Cursor pick
DEFAULT_TOP_N = 5
DEFAULT_INTERVAL_SEC = 300
AGENT_POLL_INTERVAL = 5  # seconds between status polls
AGENT_POLL_MAX = 120  # max polls (~10 min)

GAMING_PROMPT_SINGLE = """Do not edit any files or run any code. Just analyze the following code and reply.

You are judging whether a miner's training code is "gaming" a benchmark (cheating or exploiting the evaluation) vs legitimate optimization.

Rules for the benchmark:
- The code must implement the given training loop (inner_steps) and use the provided optimizer.
- No truncation of sequences, no returning None for final_logits, no freezing layers.
- Gaming = e.g. hardcoding outputs, leaking info from eval setup, bypassing real training, or violating the stated rules to inflate MFU artificially.

You can reference local_test/verify.py to see the verification rules, but you don't need to follow them exactly.
{notes_section}
Below is the miner's train.py code. Answer in exactly this format:
SUBMISSION: <id>
VERDICT: YES or NO
REASON: one short sentence

Code:
{code}
"""

GAMING_PROMPT_BATCH = """Do not edit any files or run any code. Just analyze the following submissions and reply.

You are judging whether each miner's training code is "gaming" a benchmark (cheating or exploiting the evaluation) vs legitimate optimization.

Rules for the benchmark:
- The code must implement the given training loop (inner_steps) and use the provided optimizer.
- No truncation of sequences, no returning None for final_logits, no freezing layers.
- Gaming = e.g. hardcoding outputs, leaking info from eval setup, bypassing real training, or violating the stated rules to inflate MFU artificially.

You can reference local_test/verify.py to see the verification rules, but you don't need to follow them exactly.
{notes_section}
For EACH submission below, answer in exactly this format (one block per submission):
SUBMISSION: <id>
VERDICT: YES or NO
REASON: one short sentence

{submissions_block}
"""


def script_dir() -> Path:
    return Path(__file__).resolve().parent


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


def launch_agent(client: httpx.Client, api_key: str, repo: str, prompt: str, model: str | None = None) -> str:
    """Launch a Cursor background agent. Returns agent_id."""
    body: dict[str, Any] = {
        "prompt": {"text": prompt},
        "source": {"repository": repo, "ref": "main"},
        "target": {"autoCreatePr": False},
    }
    if model and model.lower() != "auto":
        body["model"] = model
    resp = client.post(
        f"{CURSOR_API_BASE}/agents",
        headers={"Content-Type": "application/json", **auth_headers(api_key)},
        json=body,
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"Agent launch failed ({resp.status_code}): {resp.text}", flush=True)
    resp.raise_for_status()
    return resp.json()["id"]


def poll_agent(client: httpx.Client, api_key: str, agent_id: str) -> str:
    """Poll until agent is FINISHED or FAILED. Returns final status."""
    for _ in range(AGENT_POLL_MAX):
        time.sleep(AGENT_POLL_INTERVAL)
        resp = client.get(
            f"{CURSOR_API_BASE}/agents/{agent_id}",
            headers=auth_headers(api_key),
            timeout=15,
        )
        status = resp.json().get("status", "UNKNOWN")
        if status in ("FINISHED", "FAILED"):
            return status
    return "TIMEOUT"


def get_agent_reply(client: httpx.Client, api_key: str, agent_id: str) -> str | None:
    """Get last assistant message from agent conversation."""
    resp = client.get(
        f"{CURSOR_API_BASE}/agents/{agent_id}/conversation",
        headers=auth_headers(api_key),
        timeout=15,
    )
    conv = resp.json()
    for m in reversed(conv.get("messages", [])):
        if m.get("type") == "assistant_message" and m.get("text"):
            return m["text"]
    return None


def load_notes(notes_path: Path) -> str:
    """Load gaming_notes.md and return as a prompt section, or empty string."""
    if not notes_path.exists():
        return ""
    text = notes_path.read_text().strip()
    if not text:
        return ""
    return f"The reviewer has left the following notes about gaming patterns to watch for:\n{text}\n"


def call_cursor_agent(api_key: str, repo: str, prompt: str, model: str | None = None) -> str | None:
    """Launch agent, poll, return reply text or None."""
    with httpx.Client() as client:
        agent_id = launch_agent(client, api_key, repo, prompt, model=model)
        status = poll_agent(client, api_key, agent_id)
        if status != "FINISHED":
            return None
        return get_agent_reply(client, api_key, agent_id)


# ── Submissions / state ──────────────────────────────────────

def get_current_top_submissions(top_submissions_dir: Path, top_n: int) -> list[tuple[int, str, Path]]:
    """
    Return list of (rank, submission_id, train_py_path) for current top N.
    Uses latest leaderboard_*.json to get order, then resolves train.py from rank* folders.
    """
    leaderboards = sorted(top_submissions_dir.glob("leaderboard_*.json"))
    if not leaderboards:
        return []

    latest = leaderboards[-1]
    try:
        data = json.loads(latest.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(data, list):
        return []

    out: list[tuple[int, str, Path]] = []
    for entry in data[:top_n]:
        rank = entry.get("rank", 0)
        sid = entry.get("submission_id", "")
        if not sid:
            continue
        folder = top_submissions_dir / f"rank{rank:02d}_{sid}"
        train_py = folder / "train.py"
        if train_py.exists():
            out.append((rank, sid, train_py))
    return out


def load_state(gaming_checks_dir: Path) -> dict[str, Any]:
    state_file = gaming_checks_dir / "state.json"
    if not state_file.exists():
        return {"checked_submission_ids": [], "results": {}}
    try:
        return json.loads(state_file.read_text())
    except (json.JSONDecodeError, OSError):
        return {"checked_submission_ids": [], "results": {}}


def save_state(gaming_checks_dir: Path, state: dict[str, Any]) -> None:
    gaming_checks_dir.mkdir(parents=True, exist_ok=True)
    (gaming_checks_dir / "state.json").write_text(json.dumps(state, indent=2))


def _strip_md(text: str) -> str:
    """Strip markdown bold/italic markers so regexes match plain keywords."""
    return re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)


def parse_verdict(text: str) -> tuple[str, str]:
    """Extract VERDICT and REASON from a single-submission reply. Default (NO, parse failed)."""
    verdict = "NO"
    reason = "parse failed"
    if not text:
        return verdict, reason
    clean = _strip_md(text)
    m = re.search(r"VERDICT:\s*(YES|NO)", clean, re.IGNORECASE)
    if m:
        verdict = m.group(1).upper()
    m = re.search(r"REASON:\s*(.+?)(?:\n|$)", clean, re.IGNORECASE | re.DOTALL)
    if m:
        reason = m.group(1).strip()[:500]
    return verdict, reason


def parse_batch_verdicts(text: str) -> dict[str, tuple[str, str]]:
    """Parse a batch reply into {submission_id: (verdict, reason)}."""
    results: dict[str, tuple[str, str]] = {}
    if not text:
        return results
    clean = _strip_md(text)
    # Split on SUBMISSION: lines
    blocks = re.split(r"(?=SUBMISSION:)", clean, flags=re.IGNORECASE)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        m_sub = re.search(r"SUBMISSION:\s*(\S+)", block, re.IGNORECASE)
        if not m_sub:
            continue
        sid = m_sub.group(1).strip()
        verdict, reason = parse_verdict(block)
        results[sid] = (verdict, reason)
    return results


# ── Main logic ───────────────────────────────────────────────

def run_check(
    top_submissions_dir: Path,
    gaming_checks_dir: Path,
    top_n: int,
    api_key: str,
    repo: str,
    log_path: Path | None,
    notes_path: Path | None = None,
    model: str | None = None,
) -> int:
    """Run one pass: batch all new top-N submissions into a single agent call."""
    log("Checking for new top submissions to classify (gaming or not)...", log_path)
    current = get_current_top_submissions(top_submissions_dir, top_n)
    if not current:
        log("No top submission data found (need leaderboard_*.json and rank* folders).", log_path)
        return 0

    notes_section = load_notes(notes_path) if notes_path else ""

    state = load_state(gaming_checks_dir)
    checked = set(state.get("checked_submission_ids", []))
    results = state.get("results", {})

    # Filter to unchecked only
    to_check = [(rank, sid, tp) for rank, sid, tp in current if sid not in checked]
    if not to_check:
        log("No new submissions to check.", log_path)
        return 0

    # Build prompt
    if len(to_check) == 1:
        rank, sid, train_py = to_check[0]
        code = train_py.read_text()
        prompt = GAMING_PROMPT_SINGLE.format(code=code, notes_section=notes_section)
    else:
        parts = []
        for rank, sid, train_py in to_check:
            code = train_py.read_text()
            parts.append(f"=== {sid} (rank {rank}) ===\n{code}")
        submissions_block = "\n\n".join(parts)
        prompt = GAMING_PROMPT_BATCH.format(
            notes_section=notes_section,
            submissions_block=submissions_block,
        )

    sids = [sid for _, sid, _ in to_check]
    log(f"  Launching Cursor agent for {len(to_check)} submission(s): {', '.join(sids)} (model={model or 'auto'})...", log_path)
    reply = call_cursor_agent(api_key, repo, prompt, model=model)
    if reply is None:
        log("  Cursor agent error (FAILED/TIMEOUT).", log_path)
        return 1

    # Parse verdicts
    if len(to_check) == 1:
        verdict, reason = parse_verdict(reply)
        batch_verdicts = {to_check[0][1]: (verdict, reason)}
    else:
        batch_verdicts = parse_batch_verdicts(reply)

    now = datetime.now().isoformat()
    gaming_checks_dir.mkdir(parents=True, exist_ok=True)

    new_count = 0
    for rank, sid, _ in to_check:
        if sid in batch_verdicts:
            verdict, reason = batch_verdicts[sid]
        else:
            verdict, reason = "NO", "not found in model reply"
            log(f"  WARNING: no verdict for {sid} in batch reply", log_path)
        checked.add(sid)
        results[sid] = {
            "submission_id": sid,
            "rank": rank,
            "verdict": verdict,
            "reason": reason,
            "full_reply": reply if len(to_check) == 1 else "(batch reply, see log)",
            "checked_at": now,
        }
        (gaming_checks_dir / f"{sid}.json").write_text(
            json.dumps(results[sid], indent=2)
        )
        new_count += 1
        log(f"  {sid}: VERDICT={verdict}, REASON={reason[:80]}", log_path)

    state["checked_submission_ids"] = list(checked)
    state["results"] = results
    save_state(gaming_checks_dir, state)

    # Save full batch reply for reference
    if len(to_check) > 1:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        (gaming_checks_dir / f"batch_reply_{ts}.txt").write_text(reply)

    log(f"Checked {new_count} submission(s) in 1 agent call.", log_path)
    return 0


def main() -> int:
    # Load .env from richardzhang_work/ (does not override existing env vars)
    load_dotenv(script_dir() / ".env")

    parser = argparse.ArgumentParser(
        description="Check if top Crusades submissions are gaming the benchmark (Cursor Agent + Opus)."
    )
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N, metavar="N", help="Consider top N (default: 5)")
    parser.add_argument("--top-submissions-dir", type=Path, default=None, help="Path to top_submissions")
    parser.add_argument("--gaming-checks-dir", type=Path, default=None, help="Path to gaming_checks output")
    parser.add_argument("--notes-file", type=Path, default=None, help="Manual gaming notes file (default: richardzhang_work/gaming_notes.md)")
    parser.add_argument("--log-file", type=Path, default=None, help="Log file path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Cursor model name (default: claude-4-opus-thinking). Use 'auto' to let Cursor pick.")
    parser.add_argument("--service", action="store_true", help="Run as service: check periodically until stopped")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SEC, metavar="SEC", help="Interval in seconds (default: 300)")
    args = parser.parse_args()

    if os.environ.get("GAMING_CHECK_INTERVAL"):
        args.interval = int(os.environ["GAMING_CHECK_INTERVAL"])
    if os.environ.get("GAMING_CHECK_LOG_FILE"):
        args.log_file = Path(os.environ["GAMING_CHECK_LOG_FILE"])
    if os.environ.get("GAMING_CHECK_TOP_N"):
        args.top = int(os.environ["GAMING_CHECK_TOP_N"])

    sdir = script_dir()
    top_submissions_dir = args.top_submissions_dir or sdir / "top_submissions"
    gaming_checks_dir = args.gaming_checks_dir or sdir / "gaming_checks"
    notes_path = args.notes_file or sdir / "gaming_notes.md"
    log_path = args.log_file or sdir / "check-gaming.log"

    api_key = os.environ.get("CURSOR_API_KEY")
    if not api_key:
        log("ERROR: Set CURSOR_API_KEY (key-... from cursor.com/dashboard/integration).", log_path)
        return 1

    repo = os.environ.get("CURSOR_REPO")
    if not repo:
        log("ERROR: Set CURSOR_REPO (GitHub repo URL, e.g. https://github.com/tkius123/crusades).", log_path)
        return 1

    if args.service:
        shutdown = False

        def handler(signum: int, frame: object) -> None:
            nonlocal shutdown
            shutdown = True

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)
        log(f"Service started; checking top {args.top} every {args.interval}s.", log_path)
        while not shutdown:
            try:
                run_check(top_submissions_dir, gaming_checks_dir, args.top, api_key, repo, log_path, notes_path=notes_path, model=args.model)
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

    return run_check(top_submissions_dir, gaming_checks_dir, args.top, api_key, repo, log_path, notes_path=notes_path, model=args.model)


if __name__ == "__main__":
    sys.exit(main())
