#!/usr/bin/env python3
"""Backfill Google Slides object IDs for existing slides."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import re
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

def _extract_presentation_id(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(
        r"https?://docs\.google\.com/presentation/d/([a-zA-Z0-9_-]+)",
        value,
    )
    return match.group(1) if match else None


def _ensure_slide_column(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info('slides');").fetchall()}
    if "google_slide_id" in cols:
        return
    conn.execute("ALTER TABLE slides ADD COLUMN google_slide_id TEXT")
    conn.commit()


def _parse_doc_ids(value: str | None) -> list[int] | None:
    if not value:
        return None
    ids: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            ids.append(int(token))
    return ids or None


def backfill(db_path: str, doc_ids: list[int] | None, limit: int | None, dry_run: bool) -> int:
    load_dotenv()
    module_path = Path(__file__).resolve().parents[1] / "src" / "qbr_intelligence" / "infrastructure" / "google_slides.py"
    spec = importlib.util.spec_from_file_location("google_slides", module_path)
    if spec is None or spec.loader is None:
        print("Unable to load google_slides helper module.")
        return 0
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}")
        print("Install deps: pip install google-api-python-client google-auth google-auth-oauthlib")
        return 0
    client_cls = getattr(module, "GoogleSlidesClient", None)
    if client_cls is None:
        print("GoogleSlidesClient not found in helper module.")
        return 0
    client = client_cls.from_env(base_dir=Path.cwd())
    if client is None:
        print("Google Slides credentials missing. Set CLIENT_ID/CLIENT_SECRET and token envs.")
        return 0

    updated = 0
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        _ensure_slide_column(conn)

        query = "SELECT id, file_path FROM documents WHERE file_path IS NOT NULL"
        params: list[object] = []
        if doc_ids:
            placeholders = ",".join("?" for _ in doc_ids)
            query += f" AND id IN ({placeholders})"
            params.extend(doc_ids)
        query += " ORDER BY id"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        rows = conn.execute(query, params).fetchall()
        for row in rows:
            doc_id = int(row["id"])
            presentation_id = _extract_presentation_id(row["file_path"])
            if not presentation_id:
                continue
            try:
                slide_ids = client.list_slide_ids(presentation_id)
            except Exception as exc:
                print(f"[doc {doc_id}] Failed to fetch slides: {exc}")
                continue
            if not slide_ids:
                continue
            for idx, slide_id in enumerate(slide_ids, start=1):
                if dry_run:
                    updated += 1
                    continue
                conn.execute(
                    "UPDATE slides SET google_slide_id=? WHERE document_id=? AND slide_number=?",
                    (slide_id, doc_id, idx),
                )
                updated += 1
        if not dry_run:
            conn.commit()
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill google_slide_id for slides from Google Slides documents."
    )
    parser.add_argument(
        "--db",
        default="qbr_intelligence.db",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--doc-ids",
        help="Comma-separated list of document IDs to backfill",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of documents to process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute updates without writing to the database",
    )
    args = parser.parse_args()
    updated = backfill(
        db_path=args.db,
        doc_ids=_parse_doc_ids(args.doc_ids),
        limit=args.limit,
        dry_run=args.dry_run,
    )
    print(f"Updated {updated} slide(s).")


if __name__ == "__main__":
    main()
