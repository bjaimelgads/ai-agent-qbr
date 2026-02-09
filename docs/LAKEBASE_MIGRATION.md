# Lakebase Migration Guide

This repo can now target Lakebase (Databricks-managed Postgres) instead of shipping a local SQLite DB.

## 1) Configure the app
Update `deploy/dev/dev-qbr.app.yaml` with your Lakebase connection:

- `DATABASE_URL=postgresql+asyncpg://token:@<lakebase-host>:5432/<db_name>`
- `STORAGE_BACKEND=postgres`
- `TEXT_SEARCH_BACKEND=postgres_fts` (or `auto`)

## 2) Prepare Lakebase schema
Create `.env.migration` (see `.env.migration.example`) with:

```
MIGRATION_SOURCE_URL=sqlite:///qbr_intelligence.db
MIGRATION_TARGET_URL=postgresql+psycopg://token:__TOKEN__@<lakebase-host>:5432/<db_name>?sslmode=require
```

Then run the migration script locally against your SQLite DB and Lakebase Postgres:

```bash
uv run python scripts/migrate_sqlite_to_postgres.py \
  --truncate \
  --create-fts-index
```

Notes:
- `--truncate` clears target tables first.
- `--create-fts-index` creates a GIN index for `chunks.content`.

## 3) Deploy without bundling the DB
`deploy/dev/dev-qbr.deploy.sh` now defaults to skipping DB/FAISS bundling.
Set these flags to control behavior:

- `DEPLOY_DB_BUNDLE=true` to bundle `qbr_intelligence.db` (default is false)
- `DEPLOY_FAISS_BUNDLE=true` to bundle FAISS index files (default is false)

Example:

```bash
DEPLOY_DB_BUNDLE=false DEPLOY_FAISS_BUNDLE=false ./deploy/dev/dev-qbr.deploy.sh
```

## 4) Optional E2E test (requires Postgres)
Set a test Postgres URL and run:

```bash
export QBR_PG_TEST_URL=postgresql+psycopg://token:@<lakebase-host>:5432/<db_name>
uv run pytest tests/test_migrate_lakebase_e2e.py
```
