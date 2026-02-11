#!/usr/bin/env python3
"""Backfill slide.title values from PPTX slide titles."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sqlite3
import sys


def _to_sync_database_url(database_url: str) -> str:
    return database_url.replace("+aiosqlite", "").replace("+asyncpg", "")


def _sqlite_path_from_db_arg(db_arg: str) -> Path:
    if db_arg.startswith("sqlite:///"):
        return Path(db_arg.replace("sqlite:///", "", 1))
    return Path(db_arg)


def _extract_google_presentation_id(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(
        r"https?://docs\.google\.com/presentation/d/([a-zA-Z0-9_-]+)",
        value,
    )
    return match.group(1) if match else None


def _parse_doc_ids(value: str | None) -> list[int] | None:
    if not value:
        return None
    out: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if token.isdigit():
            out.append(int(token))
    return out or None


def _resolve_local_pptx(file_path: str | None, filename: str | None, search_dirs: list[Path]) -> Path | None:
    candidates: list[Path] = []
    if file_path:
        candidates.append(Path(file_path))
    if filename:
        candidates.append(Path(filename))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    if filename:
        for base in search_dirs:
            joined = base / filename
            if joined.exists() and joined.is_file():
                return joined
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill slide titles in slides.title.")
    default_db = _to_sync_database_url(os.getenv("DATABASE_URL", "sqlite:///qbr_intelligence.db"))
    parser.add_argument("--db", default=default_db, help=f"Database URL or sqlite path (default: {default_db})")
    parser.add_argument("--doc-ids", help="Comma-separated list of document IDs")
    parser.add_argument("--pptx", help="PPTX path for a single document backfill")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite non-empty slide titles")
    parser.add_argument("--dry-run", action="store_true", help="Show counts without writing")
    parser.add_argument("--limit", type=int, help="Max documents to process")
    parser.add_argument(
        "--search-dir",
        action="append",
        default=[],
        help="Extra directory to look for local PPTX files (can repeat)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from qbr_intelligence.pipeline.slide_title_backfill import backfill_slide_titles_for_document

    db_path = _sqlite_path_from_db_arg(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    doc_ids = _parse_doc_ids(args.doc_ids)
    search_dirs = [Path.cwd(), repo_root / "qbr_extraction" / "decks"]
    search_dirs.extend(Path(p) for p in args.search_dir)

    processed = 0
    total_updated = 0
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        sql = "SELECT id, filename, file_path FROM documents ORDER BY id ASC"
        params: list[object] = []
        if doc_ids:
            placeholders = ",".join("?" for _ in doc_ids)
            sql = f"SELECT id, filename, file_path FROM documents WHERE id IN ({placeholders}) ORDER BY id ASC"
            params.extend(doc_ids)
        if args.limit is not None:
            sql += " LIMIT ?"
            params.append(args.limit)
        rows = conn.execute(sql, params).fetchall()

    for row in rows:
        doc_id = int(row["id"])
        filename = row["filename"]
        file_path = row["file_path"]

        pptx_path = Path(args.pptx) if args.pptx else _resolve_local_pptx(file_path, filename, search_dirs)
        stats = backfill_slide_titles_for_document(
            database_url=str(args.db),
            document_id=doc_id,
            pptx_path=pptx_path,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        processed += 1
        total_updated += stats.updated
        source = "pptx" if stats.source_pptx_used else "raw_text"
        print(
            f"[doc {doc_id}] updated={stats.updated} skipped_existing={stats.skipped_existing} "
            f"skipped_empty={stats.skipped_empty} skipped_same={stats.skipped_same} source={source}"
        )

    mode = "Dry run" if args.dry_run else "Done"
    print(f"{mode}: processed={processed} updated={total_updated}")


if __name__ == "__main__":
    main()
