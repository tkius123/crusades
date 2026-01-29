#!/usr/bin/env python3
"""View submission details and code from the database.

Usage:
    # List all submissions
    uv run scripts/view_submission.py

    # View specific submission
    uv run scripts/view_submission.py commit_9303_1

    # Save code to file
    uv run scripts/view_submission.py commit_9303_1 --save
"""

import argparse
import sqlite3
import sys
from pathlib import Path


def list_submissions(db_path: str):
    """List all submissions in the database."""
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("""
            SELECT submission_id, miner_uid, status, final_score, created_at,
                   LENGTH(code_content) as code_len
            FROM submissions
            ORDER BY created_at DESC
        """)
        rows = cur.fetchall()

        if not rows:
            print("No submissions found.")
            return

        print("=" * 80)
        print(
            f"{'SUBMISSION':<20} {'UID':<5} {'STATUS':<10} {'TPS':<12} {'CODE':<10} {'SUBMITTED'}"
        )
        print("=" * 80)

        for r in rows:
            tps = f"{r['final_score']:.2f}" if r["final_score"] else "N/A"
            code = f"{r['code_len']} bytes" if r["code_len"] else "N/A"
            created = r["created_at"][:19] if r["created_at"] else "N/A"
            print(
                f"{r['submission_id']:<20} {r['miner_uid']:<5} {r['status']:<10} {tps:<12} {code:<10} {created}"
            )
    finally:
        conn.close()


def view_submission(db_path: str, submission_id: str, save: bool = False):
    """View a specific submission's details and code."""
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute(
            """
            SELECT submission_id, miner_uid, miner_hotkey, status, final_score,
                   code_hash, created_at, code_content
            FROM submissions
            WHERE submission_id = ?
        """,
            (submission_id,),
        )
        row = cur.fetchone()

        if not row:
            print(f"Submission '{submission_id}' not found.")
            print("\nAvailable submissions:")
            list_submissions(db_path)
            return

        print("=" * 70)
        print(f"SUBMISSION: {row['submission_id']}")
        print("=" * 70)
        print(f"UID:        {row['miner_uid']}")
        print(f"Hotkey:     {row['miner_hotkey']}")
        print(f"Status:     {row['status']}")
        if row["final_score"]:
            print(f"Score:      {row['final_score']:.2f} TPS")
        print(f"Code URL:   {row['code_hash']}")
        print(f"Submitted:  {row['created_at']}")
        print()

        if row["code_content"]:
            if save:
                # Save to file
                filename = f"{submission_id}_train.py"
                with open(filename, "w") as f:
                    f.write(row["code_content"])
                print(f"✓ Code saved to: {filename}")
            else:
                print("=" * 70)
                print("CODE CONTENT:")
                print("=" * 70)
                print(row["code_content"])
                print()
                print("(Use --save to save code to a file)")
        else:
            print("(Code not yet stored - evaluation in progress)")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="View submission details and code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all submissions
  uv run scripts/view_submission.py

  # View specific submission
  uv run scripts/view_submission.py commit_9303_1

  # Save code to file
  uv run scripts/view_submission.py commit_9303_1 --save
        """,
    )
    parser.add_argument("submission_id", nargs="?", help="Submission ID to view")
    parser.add_argument("--save", action="store_true", help="Save code to file")
    parser.add_argument("--db", default="crusades.db", help="Database path")

    args = parser.parse_args()

    # Find database
    db_path = args.db
    if not Path(db_path).exists():
        # Try from project root
        project_root = Path(__file__).parent.parent
        db_path = project_root / "crusades.db"
        if not db_path.exists():
            print(f"Database not found: {args.db}")
            sys.exit(1)
        db_path = str(db_path)

    if args.submission_id:
        view_submission(db_path, args.submission_id, args.save)
    else:
        list_submissions(db_path)


if __name__ == "__main__":
    main()
