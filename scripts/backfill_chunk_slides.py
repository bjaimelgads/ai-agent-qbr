#!/usr/bin/env python3
"""Backfill chunk start/end slide numbers from content markers."""

from __future__ import annotations

import argparse
import re
import sqlite3


def _infer_slide_number(content: str) -> int | None:
    match = re.search(r"<!-- PAGE (\d+) -->", content)
    if not match:
        return None
    return int(match.group(1))


def backfill(db_path: str) -> int:
    updated = 0
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "select id, content from chunks where start_slide is null or end_slide is null"
        ).fetchall()
        for row in rows:
            slide_num = _infer_slide_number(row["content"] or "")
            if slide_num is None:
                continue
            conn.execute(
                "update chunks set start_slide=?, end_slide=? where id=?",
                (slide_num, slide_num, row["id"]),
            )
            updated += 1
        conn.commit()
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill chunk slide ranges.")
    parser.add_argument(
        "--db",
        default="qbr_intelligence.db",
        help="Path to SQLite database",
    )
    args = parser.parse_args()
    updated = backfill(args.db)
    print(f"Updated {updated} chunk(s).")


if __name__ == "__main__":
    main()
