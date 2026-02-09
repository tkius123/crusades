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

Env overrides:
  IMPROVE_INTERVAL   — seconds between runs in --service (default: 3600)
  IMPROVE_LOG_FILE   — log file path
"""

import argparse
import base64
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

IMPROVE_PROMPT = """CRITICAL INSTRUCTIONS:
- Do NOT edit, create, or modify any files in the repository.
- Do NOT run any commands or code.
- Do NOT use any tools.
- ONLY reply in this conversation with the complete Python code.
- Your ENTIRE reply must be valid Python code and nothing else — no explanations, no markdown, no commentary.

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

Below is the current #1 honest submission (not gaming). Use it as the base and improve it to be faster.

{notes_section}
{previous_result_section}
=== Current #1 submission ===
{top_code}

Your reply must be ONLY the complete improved train.py Python code.
No explanation. No markdown fences. No commentary. No comments in the code. Just the raw Python code starting with imports.
"""


def script_dir() -> Path:
    return Path(__file__).resolve().parent


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


def call_cursor_agent(api_key: str, repo: str, prompt: str, model: str | None = None) -> str | None:
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
        agent_id = resp.json()["id"]

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

        conv = client.get(
            f"{CURSOR_API_BASE}/agents/{agent_id}/conversation",
            headers=auth_headers(api_key),
        ).json()
        for m in reversed(conv.get("messages", [])):
            if m.get("type") == "assistant_message" and m.get("text"):
                return m["text"]
    return None


# ── Helpers ──────────────────────────────────────────────────

def get_previous_result_section(output_dir: Path) -> str:
    """Find the most recent submission with an eval result and include it in the prompt."""
    submissions_path = output_dir / "submissions.json"
    if not submissions_path.exists():
        return ""
    try:
        submissions = json.loads(submissions_path.read_text())
    except (json.JSONDecodeError, OSError):
        return ""
    if not submissions:
        return ""

    # Find the most recent entry that has an actual eval result (finished or failed_*)
    evaluated = None
    for entry in reversed(submissions):
        s = entry.get("status", "")
        if s and (s == "finished" or "failed" in s.lower()):
            evaluated = entry
            break

    if not evaluated:
        return ""

    status = evaluated.get("status", "")
    mfu = evaluated.get("mfu")
    code_file = evaluated.get("code_file", "")
    error = evaluated.get("error", "")
    uid = evaluated.get("uid", "?")

    lines = [f"IMPORTANT — Most recent evaluated submission (UID {uid}):"]
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


def load_notes(notes_path: Path) -> str:
    if not notes_path.exists():
        return ""
    text = notes_path.read_text().strip()
    if not text:
        return ""
    return f"The reviewer has left the following notes:\n{text}\n"


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
            log(f"No wallet available (all submitted within the last {WALLET_COOLDOWN_SEC}s / {WALLET_COOLDOWN_SEC / 3600:.1f}h). Skipping this turn.", log_path)
            return 0

    honest = get_honest_submissions(gaming_checks_dir, top_submissions_dir)
    if not honest:
        log("No honest submissions found. Run check_gaming.py first.", log_path)
        return 1

    # Use only the top 1 honest submission (smallest rank)
    top_rank, top_sid, top_code = honest[0]
    log(f"Using #1 honest submission: {top_sid} (rank {top_rank})", log_path)

    # Build prompt
    notes_section = load_notes(notes_path) if notes_path else ""
    previous_result_section = get_previous_result_section(output_dir)
    prompt = IMPROVE_PROMPT.format(
        notes_section=notes_section,
        previous_result_section=previous_result_section,
        top_code=top_code,
    )

    log(f"Launching Cursor agent to generate improved train.py (model={model or 'auto'})...", log_path)
    reply = call_cursor_agent(api_key, repo, prompt, model=model)
    if reply is None:
        log("Cursor agent error (FAILED/TIMEOUT).", log_path)
        return 1

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
            log("No wallet available after generation. Skipping submit.", log_path)
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
            record_submission(output_dir, wallet, gist_url=raw_url, code_file=code_path.name, submit_status="failed", error=err_msg)
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
