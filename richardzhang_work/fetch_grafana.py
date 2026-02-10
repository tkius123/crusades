#!/usr/bin/env python3
"""
Fetch Grafana (Loki) logs for a Crusades validator submission.

Determines the time range from the submission_id using only
  richardzhang_work/top_submissions/*/stats.json (evaluation window or created_at/updated_at).
Then queries Grafana Loki for logs in that range and filters to lines containing submission_id.

Usage:
  python richardzhang_work/fetch_grafana.py <submission_id>
  python richardzhang_work/fetch_grafana.py v3_commit_7501050_79 --save   # -> richardzhang_work/grafana_logs/v3_commit_7501050_79.log
  python richardzhang_work/fetch_grafana.py v3_commit_7501050_79 -o out.txt

If the submission is not in DB or top_submissions, falls back to --hours (default 168).

Environment:
  GRAFANA_URL         Base URL (default: https://grafana.tplr.ai)
  GRAFANA_API_KEY     API key for auth (optional)
  LOKI_QUERY_URL      If set, query Loki directly. By default uses Grafana API with 1-min chunks.
  LOKI_UID            Optional stream selector uid (e.g. 1). With LOKI_HOST narrows to one validator.
  LOKI_HOST           Optional stream selector host (e.g. basilica-98336dfa).
  CRUSADES_API_URL    If set and submission not in top_submissions, time range is fetched from the
                      Crusades API (GET /api/submissions/{id}, /api/submissions/{id}/evaluations).
                      Use the validator-backed API base URL (e.g. https://crusades-api.example.com).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Requires httpx: pip install httpx", file=sys.stderr)
    sys.exit(1)

# Defaults matching the Crusades validator dashboard
DEFAULT_GRAFANA_URL = "https://grafana.tplr.ai"
DEFAULT_ORG_ID = "1"
DEFAULT_LOKI_UID = "loki"
# Default time range when we cannot resolve from submission (fallback only)
DEFAULT_HOURS = 168
# Buffer when using evaluation window (min/max evaluation created_at) — tight range for log fetch
EVAL_WINDOW_BUFFER_SEC = 30
# Buffer when using submission created_at/updated_at only (no evaluations available)
RANGE_BUFFER_START_SEC = 5 * 60   # 5 min before
RANGE_BUFFER_END_SEC = 15 * 60    # 15 min after
# Chunk size for paginated fetch: narrow windows so each request gets that window's logs (not global tail)
CHUNK_MS = 1 * 60 * 1000  # 1 minute per chunk


def _project_root() -> Path:
    """Project root (crusades repo)."""
    return Path(__file__).resolve().parent.parent


def _ms_to_iso(ms: int) -> str:
    """Format millisecond timestamp as ISO with decimals (e.g. 2026-02-09 05:22:49.996)."""
    if ms <= 0:
        return ""
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{(ms % 1000):03d}"


def _format_range_utc(from_ms: int, to_ms: int) -> str:
    """Format time range as UTC for display."""
    return f"{_ms_to_iso(from_ms)} - {_ms_to_iso(to_ms)} UTC"


def _parse_iso_to_ms(s: str) -> int:
    """Parse ISO-ish datetime string (UTC) to milliseconds since epoch."""
    s = s.strip().replace("Z", "").rstrip("Z")
    if not s:
        raise ValueError("empty datetime")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(s[:26], fmt)  # cap to avoid extra digits in fractional part
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)


def resolve_time_range_from_submission(
    submission_id: str,
    top_submissions_dir: Path | None = None,
) -> tuple[int, int] | None:
    """Determine (from_ms, to_ms) for this submission from local data.

    Uses only richardzhang_work/top_submissions/*/stats.json (not the local SQLite DB,
    which may not be the validator's). Prefers evaluation window (evaluations[].created_at).
    Returns None if no matching stats.json is found.
    """
    root = _project_root()
    if top_submissions_dir is None:
        top_submissions_dir = root / "richardzhang_work" / "top_submissions"

    if not top_submissions_dir.exists():
        return None

    for d in top_submissions_dir.iterdir():
        if not d.is_dir() or submission_id not in d.name:
            continue
        stats_file = d / "stats.json"
        if not stats_file.exists():
            break
        try:
            with open(stats_file) as f:
                data = json.load(f)
            evals = data.get("evaluations") or []
            if evals:
                times = [e.get("created_at") for e in evals if e.get("created_at")]
                if times:
                    from_ms = _parse_iso_to_ms(min(times)) - EVAL_WINDOW_BUFFER_SEC * 1000
                    to_ms = _parse_iso_to_ms(max(times)) + EVAL_WINDOW_BUFFER_SEC * 1000
                    return from_ms, to_ms
            detail = data.get("submission_detail") or data.get("leaderboard_entry") or data
            created = detail.get("created_at")
            updated = detail.get("updated_at") or created
            if created:
                from_ms = _parse_iso_to_ms(created) - RANGE_BUFFER_START_SEC * 1000
                to_ms = _parse_iso_to_ms(updated) + RANGE_BUFFER_END_SEC * 1000 if updated else from_ms + 3600 * 1000 * 2
                return from_ms, to_ms
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        break

    return None


def resolve_time_range_from_api(submission_id: str, api_base_url: str) -> tuple[int, int] | None:
    """Determine (from_ms, to_ms) from the Crusades API (validator-backed).

    Calls GET /api/submissions/{id} and GET /api/submissions/{id}/evaluations.
    Prefers evaluation window (evaluations[].created_at); falls back to submission created_at/updated_at.
    Returns None on network error or if the API does not return usable timestamps.
    """
    base = api_base_url.rstrip("/")
    if not base:
        return None
    try:
        with httpx.Client(timeout=15.0) as client:
            sub = client.get(f"{base}/api/submissions/{submission_id}")
            sub.raise_for_status()
            sub_data = sub.json() or {}
            evals_resp = client.get(f"{base}/api/submissions/{submission_id}/evaluations")
            evals_resp.raise_for_status()
            evals = evals_resp.json() if isinstance(evals_resp.json(), list) else []
    except (httpx.HTTPError, json.JSONDecodeError):
        return None
    # Prefer evaluation window
    times = [e.get("created_at") for e in evals if e and e.get("created_at")]
    if times:
        # created_at may be ISO string or numeric; normalize to ms
        parsed = []
        for t in times:
            if isinstance(t, (int, float)):
                parsed.append(int(t) if t > 1e12 else int(t * 1000))
            else:
                try:
                    parsed.append(_parse_iso_to_ms(str(t)))
                except (ValueError, TypeError):
                    pass
        if parsed:
            from_ms = min(parsed) - EVAL_WINDOW_BUFFER_SEC * 1000
            to_ms = max(parsed) + EVAL_WINDOW_BUFFER_SEC * 1000
            return from_ms, to_ms
    # Fallback: submission created_at / updated_at
    created = sub_data.get("created_at")
    updated = sub_data.get("updated_at") or created
    if not created:
        return None
    if isinstance(created, (int, float)):
        from_ms = int(created) if created > 1e12 else int(created * 1000) - RANGE_BUFFER_START_SEC * 1000
    else:
        try:
            from_ms = _parse_iso_to_ms(str(created)) - RANGE_BUFFER_START_SEC * 1000
        except (ValueError, TypeError):
            return None
    if isinstance(updated, (int, float)):
        to_ms = int(updated) if updated > 1e12 else int(updated * 1000) + RANGE_BUFFER_END_SEC * 1000
    else:
        try:
            to_ms = _parse_iso_to_ms(str(updated)) + RANGE_BUFFER_END_SEC * 1000
        except (ValueError, TypeError):
            to_ms = from_ms + 2 * 3600 * 1000
    return from_ms, to_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Grafana/Loki logs for a submission by submission_id.",
        epilog="Example: %(prog)s v3_commit_79639_1 --hours 48",
    )
    parser.add_argument(
        "submission_id",
        nargs="?",
        help="Submission ID (e.g. v3_commit_79639_1)",
    )
    parser.add_argument(
        "--submission-id",
        dest="submission_id_opt",
        help="Submission ID (alternative to positional)",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=DEFAULT_HOURS,
        help=f"Time range: last N hours (default: {DEFAULT_HOURS})",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=5000,
        help="Max log lines to request from Loki (default: 5000; server limit).",
    )
    parser.add_argument(
        "--from-ms",
        type=int,
        default=None,
        help="Start time in milliseconds since epoch (overrides --hours)",
    )
    parser.add_argument(
        "--to-ms",
        type=int,
        default=None,
        help="End time in milliseconds since epoch (default: now)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write log lines to this file (overrides --save path if set)",
    )
    parser.add_argument(
        "--save",
        "-s",
        action="store_true",
        help="Save log to richardzhang_work/grafana_logs/<submission_id>.log",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw JSON response (for debugging)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose (print request and summary)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print response schema and sample values to stderr (to debug parsing)",
    )
    return parser.parse_args()


def get_submission_id(args: argparse.Namespace) -> str | None:
    sid = args.submission_id or args.submission_id_opt
    return sid.strip() if sid else None


def time_range_ms(hours: float) -> tuple[int, int]:
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - int(hours * 3600 * 1000)
    return start_ms, now_ms


def build_query_payload(
    submission_id: str,
    from_ms: int,
    to_ms: int,
    loki_uid: str = DEFAULT_LOKI_UID,
    max_lines: int = 5000,
    stream_uid: str | None = None,
    stream_host: str | None = None,
) -> dict:
    # LogQL: stream selector only (no line filter), like Grafana UI. Filter by submission_id in Python.
    parts = ["service=\"crusades-validator\""]
    if stream_uid:
        parts.append(f"uid=\"{stream_uid}\"")
    if stream_host:
        parts.append(f"host=\"{stream_host}\"")
    expr = "{" + ", ".join(parts) + "}"
    query: dict = {
        "refId": "A",
        "datasource": {"type": "loki", "uid": loki_uid},
        "expr": expr,
        "queryType": "range",
        "maxLines": max_lines,
        "intervalMs": 500,
    }
    return {
        "queries": [query],
        "from": str(from_ms),
        "to": str(to_ms),
    }


def _is_numeric_column(col: list) -> bool:
    """True if every value in col looks like a number (timestamp)."""
    if not col:
        return True
    for v in col[:10]:  # sample
        s = str(v).strip()
        if s and not (s.isdigit() or (s.startswith("-") and s[1:].isdigit())):
            return False
    return True


# Loki stream ID / labels look like: 1770579606715553536_ae729757 (nanos_hash)
_STREAM_ID_RE = re.compile(r"^\d+_[a-f0-9]+$", re.IGNORECASE)


def _looks_like_log_line(col: list) -> bool:
    """True if column values look like log message text, not stream IDs or timestamps."""
    if not col:
        return False
    sample = [str(v).strip() for v in col[:20] if v is not None]
    if not sample:
        return False
    # Log lines usually have spaces, or are long, or contain common log tokens
    log_like = 0
    for s in sample:
        if not s:
            continue
        if _STREAM_ID_RE.match(s) or s.isdigit():
            return False
        if " " in s or "|" in s or len(s) > 40 or "UTC" in s or "INFO" in s or "WARN" in s or "ERROR" in s:
            log_like += 1
    return log_like > 0


def extract_log_lines(response_data: dict) -> list[tuple[int, str]]:
    """Parse Grafana Loki query response into (timestamp_ms, log_line) list.

    Picks the time column and the string column that contains actual log text
    (not stream IDs like 1770579606715553536_ae729757).
    """
    out: list[tuple[int, str]] = []
    results = response_data.get("results") or {}
    for ref_result in results.values():
        frames = ref_result.get("frames") or []
        for frame in frames:
            schema = frame.get("schema", {})
            fields = schema.get("fields") or []
            data = frame.get("data", {})
            values = data.get("values") or []
            if len(fields) != len(values) or len(fields) < 2:
                continue
            ts_idx: int | None = None
            line_idx: int | None = None
            string_indices: list[int] = []
            for idx, f in enumerate(fields):
                ftype = (f.get("type") or "").lower()
                fname = (f.get("name") or "").lower()
                if ftype == "time" or "time" in fname:
                    ts_idx = idx
                if ftype == "string":
                    string_indices.append(idx)
                    if fname in ("line", "body", "value", "message", "content"):
                        line_idx = idx
            if ts_idx is None:
                for idx, f in enumerate(fields):
                    if (f.get("type") or "").lower() in ("time", "number"):
                        ts_idx = idx
                        break
                if ts_idx is None:
                    ts_idx = 0
            if line_idx is None and string_indices:
                line_idx = string_indices[0]
            # Prefer the string column that actually contains log text (not stream IDs)
            if string_indices:
                for idx in string_indices:
                    if _looks_like_log_line(values[idx]):
                        line_idx = idx
                        break
                if line_idx is None:
                    # None look like log lines; use the column with longest values (likely the log)
                    def avg_len(idx: int) -> float:
                        col = values[idx]
                        if not col:
                            return 0.0
                        return sum(len(str(v)) for v in col) / len(col)
                    line_idx = max(string_indices, key=avg_len)
            elif line_idx is None:
                line_idx = 1 if ts_idx != 1 else 0
            ts_col = values[ts_idx]
            log_col = values[line_idx]
            if _is_numeric_column(log_col) and not _is_numeric_column(ts_col):
                ts_col, log_col = log_col, ts_col
            for i in range(min(len(ts_col), len(log_col))):
                ts_val = ts_col[i]
                line_val = log_col[i]
                if isinstance(ts_val, (int, float)):
                    ts_ms = int(ts_val / 1_000_000) if ts_val > 1e15 else int(ts_val)
                else:
                    ts_ms = 0
                line_str = str(line_val).strip()
                if line_str and not _STREAM_ID_RE.match(line_str):
                    out.append((ts_ms, line_str))
    out.sort(key=lambda x: x[0])
    return out


def _debug_print_response(data: dict) -> None:
    """Print response schema and first row of each frame to stderr."""
    results = data.get("results") or {}
    for ref_id, ref_result in results.items():
        for fi, frame in enumerate(ref_result.get("frames") or []):
            schema = frame.get("schema", {})
            fields = schema.get("fields") or []
            values = frame.get("data", {}).get("values") or []
            print(f"[debug] ref={ref_id} frame={fi} fields={len(fields)}", file=sys.stderr)
            for idx, f in enumerate(fields):
                name = f.get("name", "?")
                ftype = f.get("type", "?")
                col = values[idx] if idx < len(values) else []
                sample = str(col[0])[:80] if col else "(empty)"
                print(f"  [{idx}] name={name!r} type={ftype} sample={sample!r}", file=sys.stderr)


def _parse_loki_streams_response(data: dict) -> list[tuple[int, str]]:
    """Parse Loki native query_range response (resultType=streams)."""
    out: list[tuple[int, str]] = []
    try:
        result_type = (data.get("data") or {}).get("resultType", "")
        if result_type != "streams":
            return out
        for stream in (data.get("data") or {}).get("result") or []:
            for ts_ns_str, line in stream.get("values") or []:
                try:
                    ts_ns = int(ts_ns_str)
                    ts_ms = ts_ns // 1_000_000
                    line_str = str(line).strip()
                    if line_str:
                        out.append((ts_ms, line_str))
                except (ValueError, TypeError):
                    continue
    except (KeyError, TypeError):
        pass
    out.sort(key=lambda x: x[0])
    return out


def fetch_logs_loki_direct(
    loki_url: str,
    submission_id: str,
    from_ms: int,
    to_ms: int,
    verbose: bool,
    max_lines: int = 5000,
) -> list[tuple[int, str]]:
    """Query Loki query_range API directly with direction=forward (ascending)."""
    escaped = submission_id.replace("\\", "\\\\").replace('"', '\\"')
    logql = f'{{service="crusades-validator"}} |= "{escaped}"'
    start_ns = from_ms * 1_000_000
    end_ns = to_ms * 1_000_000
    params = {
        "query": logql,
        "start": str(start_ns),
        "end": str(end_ns),
        "direction": "forward",
        "limit": str(max_lines),
    }
    if verbose:
        print(f"GET {loki_url} (Loki direct, direction=forward)", file=sys.stderr)

    with httpx.Client(timeout=60.0) as client:
        resp = client.get(loki_url, params=params)
    resp.raise_for_status()
    data = resp.json()
    return _parse_loki_streams_response(data)


def fetch_logs(
    base_url: str,
    submission_id: str,
    from_ms: int,
    to_ms: int,
    api_key: str | None,
    org_id: str,
    verbose: bool,
    max_lines: int = 5000,
    stream_uid: str | None = None,
    stream_host: str | None = None,
) -> tuple[dict, list[tuple[int, str]]]:
    url = f"{base_url.rstrip('/')}/api/ds/query"
    params = {"ds_type": "loki", "requestId": "fetch_grafana_1"}
    payload = build_query_payload(
        submission_id, from_ms, to_ms, max_lines=max_lines,
        stream_uid=stream_uid, stream_host=stream_host,
    )

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if org_id:
        headers["X-Grafana-Org-Id"] = str(org_id)

    if verbose:
        print(f"POST {url}", file=sys.stderr)
        print(f"Query: {payload['queries'][0]['expr']}", file=sys.stderr)
        print(f"From: {from_ms} To: {to_ms}", file=sys.stderr)

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, params=params, json=payload, headers=headers)

    resp.raise_for_status()
    data = resp.json()
    return data, extract_log_lines(data)


def fetch_logs_chunked(
    base_url: str,
    submission_id: str,
    from_ms: int,
    to_ms: int,
    api_key: str | None,
    org_id: str,
    verbose: bool,
    max_lines: int = 5000,
    stream_uid: str | None = None,
    stream_host: str | None = None,
) -> list[tuple[int, str]]:
    """Fetch logs in time-range chunks and merge."""
    span_ms = to_ms - from_ms
    if span_ms <= CHUNK_MS:
        _, lines = fetch_logs(
            base_url, submission_id, from_ms, to_ms, api_key, org_id, verbose, max_lines,
            stream_uid=stream_uid, stream_host=stream_host,
        )
        return lines
    seen: set[tuple[int, str]] = set()
    merged: list[tuple[int, str]] = []
    chunk_start = from_ms
    while chunk_start < to_ms:
        chunk_end = min(chunk_start + CHUNK_MS, to_ms)
        _, chunk_lines = fetch_logs(
            base_url, submission_id, chunk_start, chunk_end, api_key, org_id, verbose, max_lines,
            stream_uid=stream_uid, stream_host=stream_host,
        )
        print(f"  Chunk {_ms_to_iso(chunk_start)} - {_ms_to_iso(chunk_end)}: {len(chunk_lines)} lines", file=sys.stderr)
        for item in chunk_lines:
            if item not in seen:
                seen.add(item)
                merged.append(item)
        chunk_start = chunk_end
    merged.sort(key=lambda x: x[0])
    return merged


def main() -> int:
    args = parse_args()
    submission_id = get_submission_id(args)
    if not submission_id:
        print("Error: submission_id required (positional or --submission-id)", file=sys.stderr)
        return 1

    base_url = os.environ.get("GRAFANA_URL", DEFAULT_GRAFANA_URL)
    api_key = os.environ.get("GRAFANA_API_KEY") or os.environ.get("GRAFANA_TOKEN")
    org_id = os.environ.get("GRAFANA_ORG_ID", DEFAULT_ORG_ID)
    stream_uid = os.environ.get("LOKI_UID", "").strip() or None
    stream_host = os.environ.get("LOKI_HOST", "").strip() or None

    if args.to_ms is not None and args.from_ms is not None:
        from_ms, to_ms = args.from_ms, args.to_ms
        if args.verbose:
            print("Using time range from --from-ms/--to-ms", file=sys.stderr)
    elif args.from_ms is not None:
        to_ms = args.to_ms or int(time.time() * 1000)
        from_ms = args.from_ms
        if args.verbose:
            print("Using time range from --from-ms", file=sys.stderr)
    else:
        resolved = resolve_time_range_from_submission(submission_id)
        if resolved:
            from_ms, to_ms = resolved
            print("Time range from submission data (evaluation window when available)", file=sys.stderr)
            if args.verbose:
                print(f"  from_ms={from_ms} to_ms={to_ms}", file=sys.stderr)
        else:
            api_url = os.environ.get("CRUSADES_API_URL", "").strip()
            if api_url:
                resolved = resolve_time_range_from_api(submission_id, api_url)
                if resolved:
                    from_ms, to_ms = resolved
                    print("Time range from Crusades API (evaluation window when available)", file=sys.stderr)
                    if args.verbose:
                        print(f"  from_ms={from_ms} to_ms={to_ms}", file=sys.stderr)
                else:
                    from_ms, to_ms = time_range_ms(args.hours)
                    print(f"API did not return a range; using last {args.hours}h", file=sys.stderr)
            else:
                from_ms, to_ms = time_range_ms(args.hours)
                print(f"Submission not in top_submissions and CRUSADES_API_URL unset; using last {args.hours}h", file=sys.stderr)

    print(f"Time range (UTC): {_format_range_utc(from_ms, to_ms)}", file=sys.stderr)

    # Use Grafana API with 1-min chunks by default. Set LOKI_QUERY_URL to use direct Loki instead.
    loki_query_url = os.environ.get("LOKI_QUERY_URL", "").strip()

    try:
        lines = []
        if loki_query_url:
            # Direct Loki query_range with direction=forward (gets full range, not just tail)
            try:
                print("Using direct Loki query_range (direction=forward)", file=sys.stderr)
                lines = fetch_logs_loki_direct(
                    loki_query_url, submission_id, from_ms, to_ms, args.verbose, args.max_lines
                )
            except httpx.HTTPStatusError as e:
                if 400 <= e.response.status_code < 600:
                    print(
                        f"Loki direct returned {e.response.status_code}, falling back to Grafana...",
                        file=sys.stderr,
                    )
                    loki_query_url = ""
                else:
                    raise
        if not loki_query_url:
            if args.raw:
                data, _ = fetch_logs(
                    base_url, submission_id, from_ms, to_ms,
                    api_key, org_id, args.verbose, args.max_lines,
                    stream_uid=stream_uid, stream_host=stream_host,
                )
                print(json.dumps(data, indent=2))
                return 0
            if args.debug:
                data, _ = fetch_logs(
                    base_url, submission_id, from_ms, to_ms,
                    api_key, org_id, args.verbose, args.max_lines,
                    stream_uid=stream_uid, stream_host=stream_host,
                )
                _debug_print_response(data)
            lines = fetch_logs_chunked(
                base_url=base_url,
                submission_id=submission_id,
                from_ms=from_ms,
                to_ms=to_ms,
                api_key=api_key,
                org_id=org_id,
                verbose=args.verbose,
                max_lines=args.max_lines,
                stream_uid=stream_uid,
                stream_host=stream_host,
            )
    except httpx.HTTPStatusError as e:
        print(f"HTTP error: {e.response.status_code} {e.response.text[:500]}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    # Query uses stream selector only; filter to lines mentioning this submission
    before = len(lines)
    lines = [(ts, line) for ts, line in lines if submission_id in line]
    if args.verbose or before != len(lines):
        print(f"Filtered to {len(lines)} lines containing {submission_id!r} (from {before} total)", file=sys.stderr)

    if args.verbose:
        print(f"Found {len(lines)} log line(s) for submission_id={submission_id}", file=sys.stderr)

    if not lines:
        print(
            f"No logs found for submission_id={submission_id} in the given time range.",
            file=sys.stderr,
        )
        print(
            "Try a longer range, e.g. --hours 720 (30 days), or --raw to inspect the API response.",
            file=sys.stderr,
        )
        return 0

    # Format with timestamp prefix (e.g. 2026-02-09 05:22:49.996\tmessage)
    formatted = [_ms_to_iso(ts_ms) + "\t" + line for ts_ms, line in lines]
    text = "\n".join(formatted) + "\n" if formatted else ""

    output_path = args.output
    if output_path is None and args.save:
        output_path = _project_root() / "richardzhang_work" / "grafana_logs" / f"{submission_id}.log"
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path is not None:
        output_path.write_text(text, encoding="utf-8")
        print(f"Wrote {len(lines)} lines to {output_path}", file=sys.stderr)
    else:
        print(text, end="")

    return 0


if __name__ == "__main__":
    sys.exit(main())
