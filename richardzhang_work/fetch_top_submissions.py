#!/usr/bin/env python3
"""
Fetch top N Crusades submissions (stats + source code) and persist to disk.

Uses the Crusades API to get leaderboard data, submission details, and source code.
Saves each submission as a directory under richardzhang_work/top_submissions/.

Usage:
  python richardzhang_work/fetch_top_submissions.py                    # one-shot, top 10
  python richardzhang_work/fetch_top_submissions.py --top 5            # one-shot, top 5
  python richardzhang_work/fetch_top_submissions.py --service          # fetch every 5 min
  python richardzhang_work/fetch_top_submissions.py --service --interval 120

Env overrides (for systemd, no code change):
  FETCH_SUBS_API_URL     — API base URL  (default: http://69.19.137.219:8080)
  FETCH_SUBS_TOP_N       — number of top submissions (default: 10)
  FETCH_SUBS_INTERVAL    — interval in seconds for --service (default: 300)
  FETCH_SUBS_LOG_FILE    — log file path
"""

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

DEFAULT_API_URL = "http://69.19.137.219:8080"
DEFAULT_TOP_N = 10
DEFAULT_INTERVAL_SEC = 300  # 5 minutes


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def log(msg: str, file: Path | None = None) -> None:
    line = f"[{datetime.now().isoformat()}] {msg}"
    print(line, flush=True)
    if file:
        with open(file, "a") as f:
            f.write(line + "\n")


# ── API helpers ──────────────────────────────────────────────

def api_get(client: httpx.Client, base_url: str, endpoint: str) -> dict[str, Any] | list[Any] | None:
    try:
        resp = client.get(f"{base_url.rstrip('/')}{endpoint}")
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as e:
        return None


def fetch_leaderboard(client: httpx.Client, base_url: str, limit: int) -> list[dict[str, Any]]:
    data = api_get(client, base_url, f"/leaderboard?limit={limit}")
    if isinstance(data, list):
        return data
    return []


def fetch_submission(client: httpx.Client, base_url: str, submission_id: str) -> dict[str, Any]:
    data = api_get(client, base_url, f"/api/submissions/{submission_id}")
    return data if isinstance(data, dict) else {}


def fetch_evaluations(client: httpx.Client, base_url: str, submission_id: str) -> list[dict[str, Any]]:
    data = api_get(client, base_url, f"/api/submissions/{submission_id}/evaluations")
    return data if isinstance(data, list) else []


def fetch_code(client: httpx.Client, base_url: str, submission_id: str) -> str | None:
    data = api_get(client, base_url, f"/api/submissions/{submission_id}/code")
    if isinstance(data, dict):
        return data.get("code")
    return None


# ── Persistence ──────────────────────────────────────────────

def read_existing_submission_id(out_dir: Path, rank: int) -> str | None:
    """Read the submission_id currently saved for a given rank, if any."""
    folder_pattern = f"rank{rank:02d}_*"
    matches = sorted(out_dir.glob(folder_pattern))
    if not matches:
        return None
    # Extract submission_id from folder name: rank01_<submission_id>
    name = matches[-1].name  # e.g. "rank01_abc123"
    parts = name.split("_", 1)
    return parts[1] if len(parts) == 2 else None


def save_submission(
    out_dir: Path,
    rank: int,
    entry: dict[str, Any],
    detail: dict[str, Any],
    evaluations: list[dict[str, Any]],
    code: str | None,
    log_path: Path | None,
) -> None:
    """Save one submission to out_dir/<rank>_<submission_id>/."""
    sid = entry.get("submission_id", "unknown")
    folder = out_dir / f"rank{rank:02d}_{sid}"
    folder.mkdir(parents=True, exist_ok=True)

    # stats.json — leaderboard entry + full submission detail
    stats = {
        "rank": rank,
        "leaderboard_entry": entry,
        "submission_detail": detail,
        "evaluations": evaluations,
        "fetched_at": datetime.now().isoformat(),
    }
    (folder / "stats.json").write_text(json.dumps(stats, indent=2, default=str))

    # train.py — source code
    if code:
        (folder / "train.py").write_text(code)
    else:
        log(f"  rank {rank} ({sid}): code not available", log_path)


# ── Main logic ───────────────────────────────────────────────

def run_fetch(
    api_url: str,
    top_n: int,
    out_dir: Path,
    log_path: Path | None,
) -> int:
    """Fetch top N submissions once and persist. Returns 0 on success."""
    log(f"Fetching top {top_n} from {api_url} ...", log_path)
    with httpx.Client(timeout=30.0) as client:
        leaderboard = fetch_leaderboard(client, api_url, top_n)
        if not leaderboard:
            log("No leaderboard data returned.", log_path)
            return 1

        log(f"Leaderboard has {len(leaderboard)} entries.", log_path)

        # Save a timestamped leaderboard snapshot
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        (out_dir / f"leaderboard_{ts}.json").write_text(
            json.dumps(leaderboard, indent=2, default=str)
        )

        saved = 0
        skipped = 0
        for entry in leaderboard:
            rank = entry.get("rank", 0)
            sid = entry.get("submission_id", "")
            mfu = entry.get("final_score", 0.0)
            uid = entry.get("miner_uid", "?")

            # Skip if this rank already has the same submission_id on disk
            existing_sid = read_existing_submission_id(out_dir, rank)
            if existing_sid == sid:
                skipped += 1
                log(f"  rank {rank}: unchanged (id={sid}), skipped", log_path)
                continue

            detail = fetch_submission(client, api_url, sid)
            evaluations = fetch_evaluations(client, api_url, sid)
            code = fetch_code(client, api_url, sid)

            save_submission(out_dir, rank, entry, detail, evaluations, code, log_path)
            saved += 1
            log(f"  rank {rank}: UID {uid}, MFU {mfu}, id={sid}", log_path)

    log(f"Done: {saved} saved, {skipped} unchanged, to {out_dir}", log_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch top Crusades submissions (stats + source code) and persist locally."
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Crusades API base URL")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N, metavar="N", help="Number of top submissions (default: 10)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: richardzhang_work/top_submissions)")
    parser.add_argument("--log-file", type=Path, default=None, help="Log file path")
    parser.add_argument("--service", action="store_true", help="Run as service: fetch periodically until stopped")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SEC, metavar="SEC", help="Fetch interval in seconds for --service (default: 300)")
    args = parser.parse_args()

    # Env overrides
    if os.environ.get("FETCH_SUBS_API_URL"):
        args.api_url = os.environ["FETCH_SUBS_API_URL"]
    if os.environ.get("FETCH_SUBS_TOP_N"):
        args.top = int(os.environ["FETCH_SUBS_TOP_N"])
    if os.environ.get("FETCH_SUBS_INTERVAL"):
        args.interval = int(os.environ["FETCH_SUBS_INTERVAL"])
    if os.environ.get("FETCH_SUBS_LOG_FILE"):
        args.log_file = Path(os.environ["FETCH_SUBS_LOG_FILE"])

    sdir = script_dir()
    out_dir = args.out_dir or sdir / "top_submissions"
    log_path = args.log_file or sdir / "fetch-top-submissions.log"

    if args.service:
        shutdown = False

        def handler(signum: int, frame: object) -> None:
            nonlocal shutdown
            shutdown = True

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

        log(f"Service started; fetching top {args.top} every {args.interval}s.", log_path)
        while not shutdown:
            try:
                run_fetch(args.api_url, args.top, out_dir, log_path)
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

    return run_fetch(args.api_url, args.top, out_dir, log_path)


if __name__ == "__main__":
    sys.exit(main())
