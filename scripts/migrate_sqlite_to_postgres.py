#!/usr/bin/env python3
"""Migrate qbr_intelligence SQLite DB to Postgres/Lakebase."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from urllib.parse import quote

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    load_dotenv = None

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("QBR_INTELLIGENCE_LIGHT_IMPORT", "1")

from qbr_intelligence.db.migrate import migrate_sqlite_to_postgres


def _expand_token(url: str, token: str) -> str:
    return re.sub(r"\$\{MIGRATION_TOKEN\}", token, url)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate SQLite DB to Postgres")
    parser.add_argument(
        "--env-file",
        default=".env",
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


def _build_target_url_from_parts() -> str | None:
    host = os.getenv("MIGRATION_TARGET_HOST", "").strip()
    user = os.getenv("MIGRATION_TARGET_USER", "").strip()
    dbname = os.getenv("MIGRATION_TARGET_DBNAME", "").strip()
    token = (
        os.getenv("MIGRATION_TARGET_TOKEN", "").strip()
        or os.getenv("MIGRATION_TOKEN", "").strip()
    )
    port = os.getenv("MIGRATION_TARGET_PORT", "5432").strip() or "5432"
    sslmode = os.getenv("MIGRATION_TARGET_SSLMODE", "require").strip() or "require"
    driver = os.getenv("MIGRATION_TARGET_DRIVER", "postgresql+psycopg").strip() or "postgresql+psycopg"

    if not host or not user or not dbname or not token:
        return None

    encoded_user = quote(user, safe="")
    encoded_token = quote(token, safe="")
    return f"{driver}://{encoded_user}:{encoded_token}@{host}:{port}/{dbname}?sslmode={sslmode}"


def _load_env_fallback(path: str, *, override: bool = False) -> None:
    if not path or not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip()
            value = value.strip()
            if value and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if not key:
                continue
            if not override and key in os.environ:
                continue
            os.environ[key] = value


def _load_env(path: str, *, override: bool = False) -> None:
    if load_dotenv is not None:
        load_dotenv(path, override=override)
        return
    _load_env_fallback(path, override=override)


def main() -> None:
    args = _parse_args()
    # Always load default .env first, then optional override file if different.
    _load_env(".env", override=False)
    _load_env(args.env_file, override=False)
    sqlite_url = args.source or os.getenv("MIGRATION_SOURCE_URL")
    postgres_url = args.target or os.getenv("MIGRATION_TARGET_URL")
    if not postgres_url:
        postgres_url = _build_target_url_from_parts()
    migration_token = os.getenv("MIGRATION_TOKEN")
    if postgres_url and migration_token:
        postgres_url = _expand_token(postgres_url, migration_token)
    if not sqlite_url or not postgres_url:
        raise SystemExit(
            "Missing source/target. Provide --source/--target or set "
            "MIGRATION_SOURCE_URL with MIGRATION_TARGET_URL (or MIGRATION_TARGET_HOST/USER/DBNAME/TOKEN) in env."
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
