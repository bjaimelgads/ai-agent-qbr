# Local Run (Real LLM, AG-UI)

This is the exact local flow to run the query path with:
- `OUTPUT_PROTOCOL=agui`
- `USE_STUB_LLM=false`

## 1) Prerequisites

- You are in the repo root.
- `.env` exists (copy from `.env.example` if needed).
- Your Databricks credentials are set in `.env`:
  - `DATABRICKS_HOST`
  - `DATABRICKS_API_BASE`
  - `DATABRICKS_API_KEY` (or OAuth vars used by your setup)

## 2) Important DB setting (required)

Use `QBR_DB_URL` (not `DATABASE_URL`) and set an absolute SQLite path.

Example for this repo:

```bash
export QBR_DB_URL=sqlite+aiosqlite:////home/brunojaime/Documents/Projects/mcp-projects/LGAds/ai-agent-qbr/qbr_intelligence.db
```

Why: if you use `sqlite+aiosqlite:///qbr_intelligence.db`, app startup can resolve to `/qbr_intelligence.db` and fail with a permission error.

## 3) Start the server (terminal 1)

```bash
OUTPUT_PROTOCOL=agui USE_STUB_LLM=false \
QBR_DB_URL=sqlite+aiosqlite:////home/brunojaime/Documents/Projects/mcp-projects/LGAds/ai-agent-qbr/qbr_intelligence.db \
uv run uvicorn ai_agent_qbr.api.app:app --host 0.0.0.0 --port 8000
```

Wait for:
- `Application startup complete.`
- `Uvicorn running on http://0.0.0.0:8000`

## 4) Run a query (terminal 2)

```bash
uv run python scripts/ws_cli.py \
  --protocol agui \
  --base-url ws://localhost:8000 \
  --message "Provide me the CPA for Disney in the US for the last H2 and H1 2024" \
  --timeout 180
```

## 5) Expected answer for that query

- H1 FY24 (Oct 2023 - Mar 2024): `$53.05`
- H2 FY24 (Apr 2024 - Sep 2024): `$45.44`

## Quick troubleshooting

- `PermissionError: '/qbr_intelligence.db'`
  - Fix `QBR_DB_URL` to an absolute path with 4 slashes after scheme (`sqlite+aiosqlite:////...`).
- WebSocket timeout
  - Increase `--timeout` (for real LLM, `180` or more can be needed).
