#!/usr/bin/env python3
"""
Fetch top N Crusades submissions (stats + source code) and persist to disk.
Optionally fetch recent submissions and update your submissions.json with results.

Uses the Crusades API to get leaderboard data, submission details, and source code.
Saves each submission as a directory under richardzhang_work/top_submissions/.

Usage:
  python richardzhang_work/fetch_top_submissions.py                    # one-shot, top 10 (stats only)
  python richardzhang_work/fetch_top_submissions.py --recent           # also fetch recent + update my results
  python richardzhang_work/fetch_top_submissions.py --top 5            # one-shot, top 5
  python richardzhang_work/fetch_top_submissions.py --service          # fetch every 5 min
  python richardzhang_work/fetch_top_submissions.py --service --interval 120 --recent

Env overrides (for systemd, no code change):
  FETCH_SUBS_API_URL     — API base URL  (default: http://69.19.137.219:8080)
  FETCH_SUBS_TOP_N       — number of top submissions (default: 10)
  FETCH_SUBS_INTERVAL    — interval in seconds for --service (default: 300)
  FETCH_SUBS_LOG_FILE    — log file path
  WALLETS                — JSON list of wallets (for matching my UIDs in recent submissions)
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
DEFAULT_TOP_N = 5
DEFAULT_INTERVAL_SEC = 300  # 5 minutes


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


def fetch_recent_submissions(client: httpx.Client, base_url: str) -> list[dict[str, Any]]:
    data = api_get(client, base_url, "/api/stats/recent")
    return data if isinstance(data, list) else []


# ── My submissions tracking ──────────────────────────────────

def get_my_uids(wallets_json: str | None) -> set[int]:
    """Parse WALLETS env and return set of my UIDs."""
    if not wallets_json:
        return set()
    try:
        wallets = json.loads(wallets_json)
        return {w["uid"] for w in wallets if "uid" in w}
    except (json.JSONDecodeError, TypeError):
        return set()


def update_my_submissions(
    client: httpx.Client,
    api_url: str,
    recent: list[dict[str, Any]],
    my_uids: set[int],
    submissions_path: Path,
    log_path: Path | None,
) -> None:
    """Match recent submissions against my UIDs and update submissions.json with results.

    Matching strategy: for each recent submission from one of my UIDs, fetch its
    full detail to get code_hash (= gist URL), then match against gist_url in
    submissions.json.
    """
    if not my_uids or not recent:
        return

    # Load existing submissions.json
    submissions: list[dict] = []
    if submissions_path.exists():
        try:
            submissions = json.loads(submissions_path.read_text())
        except (json.JSONDecodeError, OSError):
            submissions = []

    if not submissions:
        return

    # Build lookup: gist_url -> index in submissions list
    gist_lookup: dict[str, int] = {}
    for i, s in enumerate(submissions):
        url = s.get("gist_url", "")
        if url:
            gist_lookup[url] = i

    updated = 0
    for entry in recent:
        uid = entry.get("miner_uid")
        if uid not in my_uids:
            continue
        sid = entry.get("submission_id", "")
        status = entry.get("status", "")
        mfu = entry.get("final_score")

        # Try to match by submission_id first (if already recorded)
        matched_idx = None
        for i, s in enumerate(submissions):
            if s.get("submission_id") == sid:
                matched_idx = i
                break

        # If we already have this submission and it's finished, skip (don't fetch again)
        if matched_idx is not None:
            existing = submissions[matched_idx]
            if existing.get("status", "") == "finished":
                log(f"  My submission (UID {uid}): {sid} already finished — skip", log_path)
                continue

        # If not matched by id, fetch full detail to get code_hash (= gist URL) and match
        if matched_idx is None:
            detail = fetch_submission(client, api_url, sid)
            code_hash = detail.get("code_hash", "") or ""
            for url, idx in gist_lookup.items():
                if url and (url in code_hash or code_hash in url):
                    matched_idx = idx
                    break

        if matched_idx is not None:
            existing = submissions[matched_idx]
            existing_status = existing.get("status", "")
            if existing_status == "finished":
                log(f"  My submission (UID {uid}): {sid} already finished — skip", log_path)
                continue
            status_changed = existing_status != status
            submissions[matched_idx]["submission_id"] = sid
            submissions[matched_idx]["status"] = status
            submissions[matched_idx]["mfu"] = mfu
            if status_changed:
                submissions[matched_idx]["last_checked"] = datetime.now().isoformat()
            updated += 1
            log(f"  My submission (UID {uid}): {sid} status={status} MFU={mfu}", log_path)
        else:
            log(f"  My submission (UID {uid}): {sid} status={status} — no match in submissions.json", log_path)

    if updated > 0:
        submissions_path.write_text(json.dumps(submissions, indent=2))
        log(f"  Updated {updated} of my submission(s) in {submissions_path.name}", log_path)


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
    fetch_recent: bool = False,
    my_uids: set[int] | None = None,
    submissions_path: Path | None = None,
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

        # Fetch recent submissions and update my results
        if fetch_recent and my_uids and submissions_path:
            log("Fetching recent submissions...", log_path)
            recent = fetch_recent_submissions(client, api_url)
            if recent:
                log(f"  {len(recent)} recent submission(s) returned.", log_path)
                update_my_submissions(client, api_url, recent, my_uids, submissions_path, log_path)
            else:
                log("  No recent submissions returned.", log_path)

    return 0


def main() -> int:
    load_dotenv(script_dir() / ".env")

    parser = argparse.ArgumentParser(
        description="Fetch top Crusades submissions (stats + source code) and persist locally."
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Crusades API base URL")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N, metavar="N", help="Number of top submissions (default: 10)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: richardzhang_work/top_submissions)")
    parser.add_argument("--log-file", type=Path, default=None, help="Log file path")
    parser.add_argument("--recent", action="store_true", help="Also fetch recent submissions and update my results in submissions.json")
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

    # Parse my UIDs from WALLETS env
    my_uids = get_my_uids(os.environ.get("WALLETS"))
    submissions_path = sdir / "improved" / "submissions.json"

    fetch_kwargs = dict(
        api_url=args.api_url,
        top_n=args.top,
        out_dir=out_dir,
        log_path=log_path,
        fetch_recent=args.recent,
        my_uids=my_uids,
        submissions_path=submissions_path,
    )

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
                run_fetch(**fetch_kwargs)
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

    return run_fetch(**fetch_kwargs)


if __name__ == "__main__":
    sys.exit(main())
