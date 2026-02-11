# Lakebase Migration Guide (QBR)

This document describes the full migration from local SQLite to Lakebase (Databricks Postgres) for the QBR agent, including Databricks-side setup, permissions, migration execution, and production validation.

## Goals
- Move runtime storage from SQLite to Lakebase Postgres.
- Keep metric query and retrieval functional after migration.
- Enable `pgvector` search backend and Postgres full-text search.
- Make deployment repeatable with clear operational checks.

## Architecture Changes Included
- Runtime storage backend switched to Postgres (`STORAGE_BACKEND=postgres`).
- Metric QA path runs against the same Lakebase-aware DB gateway as app startup.
- Retrieval backend supports `pgvector` (`VECTOR_BACKEND=pgvector`).
- Migration tooling copies SQLite data into Lakebase and prepares FTS indexes.
- Schema alignment includes `metrics.llm_context_label` so metric queries do not fail at view creation time.

## Databricks-Side Prerequisites

### 1) Lakebase instance readiness
Confirm all of the following before app deploy:
- Lakebase instance exists and is not paused.
- Read/write endpoint is available for the instance.
- Database name exists (for this project: `databricks_postgres`).
- Network path allows the caller (IP ACL / private link / workspace app networking).

If this is not ready, common runtime error is:
- `External authorization failed` with details about paused instance, ACLs, readable secondaries, or private link.

### 2) Identity to use
QBR app runtime is designed to connect with Databricks app/service-principal identity in Databricks Apps.  
For local migration scripts, a valid Databricks user token/PAT can be used.

### 3) Workspace-level permissions
Ensure the identity used by runtime/migration can:
- Access the workspace.
- Call Databricks Database APIs used for temporary DB credentials.
- Access the target Lakebase instance.

### 4) Database privileges
Inside Postgres/Lakebase, grant the identity privileges required by QBR:
- Database connect.
- Schema usage (`public`).
- Table read/write as needed by runtime and migrations.
- Sequence usage/update for inserted rows.

Example SQL pattern (adapt principal/role name to your environment):
```sql
GRANT CONNECT ON DATABASE databricks_postgres TO "<principal>";
GRANT USAGE ON SCHEMA public TO "<principal>";
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO "<principal>";
GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA public TO "<principal>";
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO "<principal>";
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO "<principal>";
```

### 5) Extensions
If using vector retrieval, enable pgvector:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## QBR Configuration (App Runtime)

Reference file: `deploy/dev/dev-qbr.app.yaml`

Required:
- `QBR_LAKEBASE_ENABLED=true`
- `QBR_DATABRICKS_APP_AUTH=true`
- `QBR_LAKEBASE_DATABRICKS_HOST=<workspace-host>`
- `QBR_LAKEBASE_DB_INSTANCE=<instance-name>`
- `QBR_LAKEBASE_DB_NAME=<db-name>`
- `QBR_LAKEBASE_DB_PORT=5432`
- `QBR_LAKEBASE_DB_USERNAME=<identity-username-or-client-id>`
- `STORAGE_BACKEND=postgres`
- `TEXT_SEARCH_BACKEND=postgres_fts`

Recommended:
- `VECTOR_BACKEND=pgvector`
- `METRIC_QA_DEBUG_LOG=true`
- `PLANNER_DEBUG_EVENTS=true`

Note:
- Runtime uses a Lakebase placeholder-style URL and injects short-lived DB password dynamically through Databricks API.
- SSL is enabled for asyncpg Postgres connections in Lakebase mode.

## Local Migration (SQLite -> Lakebase)

Migration script:
- `scripts/migrate_sqlite_to_postgres.py`

Source:
- `sqlite:///qbr_intelligence.db`

Target can be provided either as full URL or assembled from parts via env vars.

### 1) Prepare environment
Use one consistent set of migration variables (avoid duplicates):
```bash
MIGRATION_SOURCE_URL=sqlite:///qbr_intelligence.db
MIGRATION_TARGET_HOST=<lakebase-host>
MIGRATION_TARGET_PORT=5432
MIGRATION_TARGET_DBNAME=databricks_postgres
MIGRATION_TARGET_USER=<user-or-principal>
MIGRATION_TARGET_SSLMODE=require
MIGRATION_TARGET_DRIVER=postgresql+psycopg
MIGRATION_TOKEN=<valid-databricks-db-token>
MIGRATION_TARGET_TOKEN=${MIGRATION_TOKEN}
```

### 2) Install dependencies
```bash
uv sync --all-extras
```

### 3) Run migration
```bash
uv run scripts/migrate_sqlite_to_postgres.py --truncate --create-fts-index
```

What this does:
- Creates tables from SQLAlchemy metadata.
- Applies compatibility DDL (column width adjustments and missing-column guard for `metrics.llm_context_label`).
- Truncates target tables when `--truncate` is set.
- Copies all rows in batches.
- Resets sequences.
- Creates Postgres FTS index on `chunks.content` when `--create-fts-index` is set.

## Critical Schema Alignment

`metrics.llm_context_label` must exist in two places:
- Database table (Postgres): `metrics.llm_context_label`
- ORM model: `src/qbr_intelligence/db/models.py`

If model and DB drift, migration can silently copy the table without this field and metric query view creation will fail later.

## Verification Checklist

### 1) Connectivity check
From logs, confirm startup includes:
- `Database connectivity check: ok (...)`

### 2) Row-level checks
```sql
SELECT COUNT(*) FROM metrics;
SELECT COUNT(*) FROM metrics WHERE llm_context_label IS NOT NULL;
SELECT COUNT(*) FROM chunks;
SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL;
```

### 3) Extension checks
```sql
SELECT extname, extversion FROM pg_extension WHERE extname='vector';
```

### 4) Retrieval backend checks
App startup should show:
- `VECTOR_BACKEND=pgvector`
- FAISS auto-build skipped when pgvector is active.

### 5) Metric query checks
Send a KPI/metric prompt and confirm logs show tool path:
- `resolve_metric_intent`
- `query_metrics`

If debug logs are enabled, confirm metric QA plan and row counts are logged.

## Common Failures and Fixes

### `Could not parse SQLAlchemy URL`
Cause:
- Invalid `QBR_DB_URL` or unresolved placeholder in URL.
Fix:
- Use a valid URL format or rely on Lakebase env vars with placeholder URL intentionally ignored.

### `Lakebase settings missing ...`
Cause:
- Missing required envs (`QBR_LAKEBASE_DB_INSTANCE`, `QBR_LAKEBASE_DB_NAME`, `QBR_LAKEBASE_DB_USERNAME`, and auth settings).
Fix:
- Set all required values in deployment config.

### `validate: more than one authorization method configured: oauth and pat`
Cause:
- Databricks SDK received both OAuth and PAT at once.
Fix:
- Use one auth method per runtime mode.
- In Databricks Apps, prefer app auth mode and avoid conflicting token envs.

### `Ensure that connection is using SSL`
Cause:
- Non-SSL connection attempt to Postgres endpoint.
Fix:
- Ensure `sslmode=require` for psycopg local migration.
- Ensure Lakebase runtime path injects `ssl=True` for asyncpg.

### `External authorization failed`
Cause:
- Databricks/Lakebase-side permissions or network restrictions.
Fix:
- Verify instance state, ACL/private link, identity permissions, and DB grants.

### `column m.llm_context_label does not exist`
Cause:
- Target schema missing expected column.
Fix:
- Apply migration/schema fix and rerun migration with truncate.

### `current transaction is aborted`
Cause:
- Prior SQL failure in same transaction (often schema/view mismatch).
Fix:
- Resolve first SQL error, rerun clean migration, and recreate view.

## Deployment Notes

Deploy script: `deploy/dev/dev-qbr.deploy.sh`

Default behavior should avoid bundling local DB/FAISS artifacts when Lakebase is the source of truth:
- `DEPLOY_DB_BUNDLE=false`
- `DEPLOY_FAISS_BUNDLE=false`

Example:
```bash
DEPLOY_DB_BUNDLE=false DEPLOY_FAISS_BUNDLE=false ./deploy/dev/dev-qbr.deploy.sh
```

## Operational Runbook (Quick)
1. Verify Databricks prerequisites (instance + permissions + network).
2. Verify app config in `deploy/dev/dev-qbr.app.yaml`.
3. Run SQLite -> Lakebase migration.
4. Validate row counts and required columns/extensions.
5. Deploy app.
6. Verify startup connectivity log and first metric query tool path.

## Files Involved
- `deploy/dev/dev-qbr.app.yaml`
- `deploy/dev/dev-qbr.deploy.sh`
- `scripts/migrate_sqlite_to_postgres.py`
- `src/qbr_intelligence/db/migrate.py`
- `src/qbr_intelligence/db/models.py`
- `src/qbr_intelligence/metric_qa/dao.py`
- `src/qbr_intelligence/metric_qa/engine.py`
- `src/qbr_agent/infrastructure/sqlalchemy_gateway.py`
- `src/qbr_agent/infrastructure/pgvector_index.py`
