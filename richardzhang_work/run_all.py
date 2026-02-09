#!/usr/bin/env python3
"""
Master loop: runs all 4 services in sequence, forever.

Each cycle:
  1. Fetch top submissions + recent (update my results)
  2. Check for gaming in new top-5 submissions
  3. Generate improved train.py and submit (if wallet available)
  4. Stage and push richardzhang_work changes to git
  5. Sync upstream (fetch + merge upstream into main)

Usage:
  uv run python richardzhang_work/run_all.py
  uv run python richardzhang_work/run_all.py --interval 1800    # 30 min between cycles
  uv run python richardzhang_work/run_all.py --no-submit        # generate but don't submit
  uv run python richardzhang_work/run_all.py --no-push          # don't push git changes

Env:
  Reads richardzhang_work/.env for CURSOR_API_KEY, CURSOR_REPO, GITHUB_TOKEN, WALLETS, NETWORK.
  RUN_ALL_INTERVAL — override cycle interval (default: 3600)
"""

import signal
import subprocess
import sys
import os
import time
from datetime import datetime
from pathlib import Path


DEFAULT_INTERVAL_SEC = 60  # 1 minute between full cycles


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    return script_dir().parent


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


def run_step(name: str, cmd: list[str], log_path: Path | None) -> bool:
    """Run a subprocess step. Returns True on success."""
    log(f"── {name} ──", log_path)
    result = subprocess.run(
        cmd,
        cwd=str(repo_root()),
        capture_output=True,
        text=True,
    )
    # Log stdout (last 20 lines to keep it concise)
    stdout_lines = result.stdout.strip().splitlines()
    for line in stdout_lines[-20:]:
        log(f"  {line}", log_path)
    if result.returncode != 0:
        stderr_lines = result.stderr.strip().splitlines()
        for line in stderr_lines[-10:]:
            log(f"  ERROR: {line}", log_path)
        log(f"  {name} failed (exit {result.returncode})", log_path)
        return False
    log(f"  {name} done.", log_path)
    return True


def git_push(log_path: Path | None) -> bool:
    """Stage richardzhang_work changes and push to origin."""
    root = repo_root()
    # Check if there are any changes to commit
    result = subprocess.run(
        ["git", "status", "--porcelain", "richardzhang_work/"],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    changes = result.stdout.strip()
    if not changes:
        log("  No changes to commit.", log_path)
        return True

    log(f"  {len(changes.splitlines())} file(s) changed.", log_path)

    cmds = [
        ["git", "add", "richardzhang_work/"],
        ["git", "commit", "-m", f"auto: update richardzhang_work {datetime.now().strftime('%Y-%m-%d %H:%M')}"],
        ["git", "push", "origin", "HEAD"],
    ]
    for cmd in cmds:
        r = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
        if r.returncode != 0:
            log(f"  git error: {r.stderr.strip()}", log_path)
            return False
    log("  Committed and pushed.", log_path)
    return True


def run_cycle(
    log_path: Path | None,
    do_submit: bool = True,
    do_push: bool = True,
) -> None:
    log("=" * 60, log_path)
    log("CYCLE START", log_path)
    log("=" * 60, log_path)

    # 1. Fetch top submissions + recent
    run_step(
        "Fetch top submissions + recent",
        ["uv", "run", "python", "richardzhang_work/fetch_top_submissions.py", "--recent"],
        log_path,
    )

    # 2. Check gaming
    run_step(
        "Check gaming",
        ["uv", "run", "python", "richardzhang_work/check_gaming.py"],
        log_path,
    )

    # 3. Generate improvement (+ submit if enabled)
    improve_cmd = ["uv", "run", "python", "richardzhang_work/improve_and_submit.py"]
    if do_submit:
        improve_cmd.append("--submit")
    run_step(
        "Generate improvement" + (" + submit" if do_submit else ""),
        improve_cmd,
        log_path,
    )

    # 4. Stage and push git changes
    if do_push:
        log("── Git push ──", log_path)
        git_push(log_path)
    else:
        log("── Git push skipped (--no-push) ──", log_path)

    # 5. Sync upstream (fetch + merge upstream into main)
    run_step(
        "Sync upstream",
        ["uv", "run", "python", "richardzhang_work/sync_upstream.py", "--push"],
        log_path,
    )

    log("CYCLE DONE", log_path)
    log("", log_path)


def main() -> int:
    load_dotenv(script_dir() / ".env")

    import argparse
    parser = argparse.ArgumentParser(description="Master loop: fetch, check gaming, improve, submit, push.")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SEC, metavar="SEC", help="Seconds between cycles (default: 3600)")
    parser.add_argument("--no-submit", action="store_true", help="Generate improvements but don't submit")
    parser.add_argument("--no-push", action="store_true", help="Don't push git changes")
    parser.add_argument("--once", action="store_true", help="Run one cycle then exit")
    args = parser.parse_args()

    if os.environ.get("RUN_ALL_INTERVAL"):
        args.interval = int(os.environ["RUN_ALL_INTERVAL"])

    log_path = script_dir() / "run_all.log"

    if args.once:
        run_cycle(log_path, do_submit=not args.no_submit, do_push=not args.no_push)
        return 0

    shutdown = False

    def handler(signum: int, frame: object) -> None:
        nonlocal shutdown
        shutdown = True

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)

    log(f"Master loop started; cycle every {args.interval}s.", log_path)
    while not shutdown:
        try:
            run_cycle(log_path, do_submit=not args.no_submit, do_push=not args.no_push)
        except Exception as e:
            log(f"CYCLE ERROR: {e}", log_path)
        if shutdown:
            break
        log(f"Sleeping {args.interval}s until next cycle...", log_path)
        for _ in range(args.interval):
            if shutdown:
                break
            time.sleep(1)
    log("Master loop stopped.", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
