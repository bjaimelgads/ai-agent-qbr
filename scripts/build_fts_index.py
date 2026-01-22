#!/usr/bin/env python
"""Create and rebuild the chunks FTS5 index for BM25 text search."""

from __future__ import annotations

import argparse
import sqlite3
from urllib.parse import urlparse


def _resolve_sqlite_path(database_url: str) -> str:
    parsed = urlparse(database_url)
    if parsed.scheme not in {"sqlite", "sqlite+aiosqlite"}:
        raise ValueError("DATABASE_URL must be sqlite for FTS5 indexing")
    if not parsed.path:
        raise ValueError("DATABASE_URL is missing a database path")
    path = parsed.path
    if path.startswith("//"):
        path = path[1:]
    return path


def _ensure_fts(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
        USING fts5(content, content='chunks', content_rowid='id');

        CREATE TRIGGER IF NOT EXISTS chunks_fts_ai
        AFTER INSERT ON chunks BEGIN
          INSERT INTO chunks_fts(rowid, content)
          VALUES (new.id, new.content);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_fts_ad
        AFTER DELETE ON chunks BEGIN
          INSERT INTO chunks_fts(chunks_fts, rowid, content)
          VALUES('delete', old.id, old.content);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_fts_au
        AFTER UPDATE ON chunks BEGIN
          INSERT INTO chunks_fts(chunks_fts, rowid, content)
          VALUES('delete', old.id, old.content);
          INSERT INTO chunks_fts(rowid, content)
          VALUES (new.id, new.content);
        END;
        """
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=None,
        help="SQLite DATABASE_URL (defaults to env DATABASE_URL)",
    )
    args = parser.parse_args()

    database_url = args.database_url
    if not database_url:
        from os import getenv

        database_url = getenv("DATABASE_URL", "sqlite:///qbr_intelligence.db")
    db_path = _resolve_sqlite_path(database_url)

    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error as exc:
        raise SystemExit(f"Failed to open database: {exc}")

    try:
        _ensure_fts(conn)
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        conn.commit()
    except sqlite3.OperationalError as exc:
        raise SystemExit(f"FTS5 setup failed: {exc}")
    finally:
        conn.close()

    print(f"FTS5 index built for: {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
