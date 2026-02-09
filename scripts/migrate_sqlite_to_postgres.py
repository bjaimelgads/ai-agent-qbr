#!/usr/bin/env python3
"""Migrate qbr_intelligence SQLite DB to Postgres/Lakebase."""

from __future__ import annotations

import argparse
import os
import re

from dotenv import load_dotenv

from qbr_intelligence.db.migrate import migrate_sqlite_to_postgres


def _expand_token(url: str, token: str) -> str:
    return re.sub(r"\$\{MIGRATION_TOKEN\}", token, url)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate SQLite DB to Postgres")
    parser.add_argument(
        "--env-file",
        default=".env.migration",
        help="Env file with MIGRATION_SOURCE_URL and MIGRATION_TARGET_URL",
    )
    parser.add_argument("--source", help="SQLite URL (e.g., sqlite:///qbr_intelligence.db)")
    parser.add_argument(
        "--target",
        help="Postgres URL (e.g., postgresql+psycopg://user:pass@host:5432/db)",
    )
    parser.add_argument("--batch-size", type=int, default=1000, help="Rows per batch")
    parser.add_argument("--no-create-tables", action="store_true", help="Skip table creation")
    parser.add_argument("--truncate", action="store_true", help="Truncate target tables before copy")
    parser.add_argument("--no-reset-sequences", action="store_true", help="Skip sequence reset")
    parser.add_argument("--create-fts-index", action="store_true", help="Create Postgres FTS index on chunks")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    load_dotenv(args.env_file, override=False)
    sqlite_url = args.source or os.getenv("MIGRATION_SOURCE_URL")
    postgres_url = args.target or os.getenv("MIGRATION_TARGET_URL")
    migration_token = os.getenv("MIGRATION_TOKEN")
    if postgres_url and migration_token:
        postgres_url = _expand_token(postgres_url, migration_token)
    if not sqlite_url or not postgres_url:
        raise SystemExit(
            "Missing source/target. Provide --source/--target or set "
            "MIGRATION_SOURCE_URL and MIGRATION_TARGET_URL in .env.migration."
        )

    result = migrate_sqlite_to_postgres(
        sqlite_url=sqlite_url,
        postgres_url=postgres_url,
        batch_size=args.batch_size,
        create_tables=not args.no_create_tables,
        truncate=args.truncate,
        reset_sequences=not args.no_reset_sequences,
        create_fts_indexes=args.create_fts_index,
    )
    print("Migration complete.")
    print(f"Total rows copied: {result.total_rows}")
    for table, count in result.table_counts.items():
        print(f"  {table}: {count}")


if __name__ == "__main__":
    main()
