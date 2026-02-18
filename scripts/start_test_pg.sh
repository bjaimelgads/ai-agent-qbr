#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-qbr-pg-test}"
DB_NAME="${DB_NAME:-qbr_test}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-postgres}"
PORT="${PORT:-5432}"

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  docker start "${CONTAINER_NAME}" >/dev/null
else
  docker run --name "${CONTAINER_NAME}" \
    -e POSTGRES_PASSWORD="${DB_PASSWORD}" \
    -e POSTGRES_DB="${DB_NAME}" \
    -p "${PORT}:5432" \
    -d postgres:16 >/dev/null
fi

cat <<EOF
Postgres test DB is running.

Export this to run the e2e migration test:
  export QBR_PG_TEST_URL=postgresql+psycopg://${DB_USER}:${DB_PASSWORD}@localhost:${PORT}/${DB_NAME}

Then run:
  uv run pytest tests/test_migrate_lakebase_e2e.py
EOF
