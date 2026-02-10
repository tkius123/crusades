#!/usr/bin/env python3
"""
Sync upstream changes into main: fetch upstream, merge if there are new commits,
and log what was merged (commits with hash, author, date, subject).

Usage:
  python richardzhang_work/sync_upstream.py              # fetch + merge, no push (one shot)
  python richardzhang_work/sync_upstream.py --push      # merge then push to origin
  python richardzhang_work/sync_upstream.py --dry-run   # only fetch and report
  python richardzhang_work/sync_upstream.py --service   # run as service: sync every 60s

As a systemd service (sync every minute):
  See richardzhang_work/crusades-sync-upstream.service
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Env overrides (so you can configure the service without editing ExecStart):
#   SYNC_UPSTREAM_INTERVAL   — interval in seconds (e.g. 120)
#   SYNC_UPSTREAM_LOG_FILE  — log file path
#   SYNC_UPSTREAM_PUSH      — set to 1, true or yes to push after merge

DEFAULT_INTERVAL_SEC = 60


def run(cmd: list[str], capture: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        cwd=repo_root(),
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr or result.stdout}")
    return result


def repo_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent


def log(msg: str, file: Path | None = None) -> None:
    line = f"[{datetime.now().isoformat()}] {msg}"
    print(line)
    if file:
        with open(file, "a") as f:
            f.write(line + "\n")


def run_sync(
    root: Path,
    log_path: Path,
    *,
    push: bool = False,
    dry_run: bool = False,
) -> int:
    """Run one sync: fetch upstream, merge if needed, log changes. Returns 0 on success, 1 on failure."""

    def log_both(msg: str) -> None:
        log(msg, log_path)

    try:
        run(["git", "rev-parse", "--is-inside-work-tree"])
    except RuntimeError:
        log_both("ERROR: Not inside a git repository.")
        return 1

    try:
        run(["git", "diff-index", "--quiet", "HEAD", "--"])
    except RuntimeError:
        log_both("ERROR: Working tree is dirty. Commit or stash changes first.")
        return 1

    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    if branch != "main":
        log_both(f"Checking out main (was on {branch}).")
        run(["git", "checkout", "main"])

    run(["git", "fetch", "upstream"])

    for candidate in ["upstream/main", "upstream/master"]:
        r = subprocess.run(
            ["git", "rev-parse", candidate],
            capture_output=True,
            text=True,
            cwd=root,
        )
        if r.returncode == 0:
            upstream_branch = candidate
            break
    else:
        log_both("ERROR: No upstream/main or upstream/master found.")
        return 1

    rev_list = run(
        ["git", "rev-list", "--reverse", f"main..{upstream_branch}"],
    ).stdout.strip()
    commits = [s for s in rev_list.splitlines() if s]

    if not commits:
        log_both(f"Up to date with {upstream_branch}. Nothing to merge.")
        return 0

    log_both(f"Found {len(commits)} new commit(s) on {upstream_branch}.")
    for rev in commits:
        fmt = "%h %an %ad %s"
        info = run(["git", "log", "-1", f"--format={fmt}", "--date=short", rev]).stdout.strip()
        log_both(f"  {info}")

    if dry_run:
        log_both("Dry run: would merge these commits into main.")
        return 0

    run(["git", "merge", upstream_branch, "--no-edit", "-m", f"Merge {upstream_branch} into main"])

    log_both("Merged successfully. Commits brought in:")
    for rev in commits:
        fmt = "%h %an %ad %s"
        info = run(["git", "log", "-1", f"--format={fmt}", "--date=short", rev]).stdout.strip()
        log_both(f"  {info}")

    # Keep deps in sync with merged code (per project README: uv sync)
    log_both("Running uv sync...")
    run(["uv", "sync"])
    log_both("uv sync done.")

    if push:
        push_cmd = ["git", "push", "origin", "main"]
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        if token:
            url_result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=root,
                capture_output=True,
                text=True,
            )
            if url_result.returncode == 0 and url_result.stdout.strip():
                remote = url_result.stdout.strip()
                if remote.startswith("https://") and "@" not in remote:
                    push_url = remote.replace("https://", f"https://{token}@", 1)
                    push_cmd = ["git", "push", push_url, "main"]
        run(push_cmd)
        log_both("Pushed main to origin.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch upstream and merge into main; log merged changes.")
    parser.add_argument("--push", action="store_true", help="Push main to origin after merging")
    parser.add_argument("--dry-run", action="store_true", help="Only fetch and report; do not merge or push")
    parser.add_argument(
        "--service",
        action="store_true",
        help="Run as service: sync every 60s until stopped (SIGTERM/SIGINT)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL_SEC,
        metavar="SEC",
        help="Sync interval in seconds when using --service (default: 60)",
    )
    parser.add_argument("--log-file", type=Path, default=None, help="Log file path (default: richardzhang_work/sync-upstream.log)")
    args = parser.parse_args()

    # Env overrides (for systemd Environment=)
    if os.environ.get("SYNC_UPSTREAM_INTERVAL"):
        args.interval = int(os.environ["SYNC_UPSTREAM_INTERVAL"])
    if os.environ.get("SYNC_UPSTREAM_LOG_FILE"):
        args.log_file = Path(os.environ["SYNC_UPSTREAM_LOG_FILE"])
    if os.environ.get("SYNC_UPSTREAM_PUSH", "").lower() in ("1", "true", "yes"):
        args.push = True

    root = repo_root()
    log_dir = Path(__file__).resolve().parent
    log_path = args.log_file or log_dir / "sync-upstream.log"

    if args.service:
        shutdown = False

        def handler(signum: int, frame: object) -> None:
            nonlocal shutdown
            shutdown = True

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

        log("Service started; syncing every %s second(s)." % args.interval, log_path)
        while not shutdown:
            try:
                run_sync(root, log_path, push=args.push, dry_run=False)
            except RuntimeError as e:
                log(str(e), log_path)
            if shutdown:
                break
            for _ in range(args.interval):
                if shutdown:
                    break
                time.sleep(1)
        log("Service stopped.", log_path)
        return 0

    return run_sync(root, log_path, push=args.push, dry_run=args.dry_run)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
