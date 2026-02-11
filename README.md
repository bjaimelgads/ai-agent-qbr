# ai-agent-qbr

QBR retrieval agent that answers questions using knowledge extracted from past PowerPoints. It preserves the existing WebSocket transport and adds AG-UI streaming support while cleanly separating domain, application, and infrastructure layers.

## Overview
- WebSocket server with legacy and AG-UI output formats.
- QBR knowledge stored in SQLite (`qbr_intelligence.db`) with chunk embeddings.
- Clean architecture ports for repository, vector search, and embeddings.

## Quick Start

```bash
# Install dependencies
python -m pip install -e .

# Run the WebSocket server
python app.py
# or
uvicorn ai_agent_qbr.api.app:app --reload
```

## Configuration

Copy `.env.example` to `.env` and update values.

Key settings:
- `OUTPUT_PROTOCOL=legacy|agui`
- `DATABASE_URL=sqlite+aiosqlite:///qbr_intelligence.db`
- `STORAGE_BACKEND=sqlite`
- `VECTOR_BACKEND=sqlite_embeddings`
- `EMBEDDINGS_BACKEND=sentence_transformers|hash`
- `EMBEDDINGS_MODEL=all-MiniLM-L6-v2`
- `RETRIEVAL_TOP_K=5`
- `RERANK_BACKEND=cross_encoder|none`
- `RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`
- `RERANK_TOP_N=20`
- `RETRIEVAL_MMR_LAMBDA=0.5`
- `RETRIEVAL_MAX_CHUNKS_PER_DOC=3`

## Local WebSocket Chat (Legacy CLI)

1) Create your env file:
```
cp .env.example .env
```

2) Update `.env` for local legacy WebSocket runs:
```
OUTPUT_PROTOCOL=websocket
PLANNER_STREAM_FINAL_RESPONSE=true
LOG_LEVEL=DEBUG
PLANNER_DEBUG_EVENTS=true
QBR_INTELLIGENCE_LIGHT_IMPORT=1
WS_RECEIVE_TIMEOUT_SECONDS=180
WS_KEEPALIVE_SECONDS=20
```
Make sure your Databricks settings are set (when `USE_STUB_LLM=false`):
```
DATABRICKS_HOST=...
DATABRICKS_API_KEY=...   # or DATABRICKS_TOKEN
# optional service principal auth
DATABRICKS_CLIENT_ID=...
DATABRICKS_CLIENT_SECRET=...
```

3) Start the server:
```
uvicorn ai_agent_qbr.api.app:app --reload
```

4) Run the legacy CLI:
```
python scripts/ws_cli.py --protocol legacy --base-url ws://localhost:8000 --repl --timeout 120
```

Single message example:
```
python scripts/ws_cli.py --protocol legacy --base-url ws://localhost:8000 --message "What is the value for CPA in H2 FY25?" --timeout 120
```

## Architecture

See:
- `docs/discovery.md` for current system findings
- `docs/architecture.md` for the target clean architecture

## How Retrieval Works
1. Query is embedded via `EmbeddingsProvider`.
2. `VectorIndex` ranks stored chunk embeddings (SQLite or FAISS).
3. `KnowledgeRepository` returns chunk payloads and text matches.
4. Retrieved context is injected into the planner (`qbr_context`).

## Retrieval Improvements (Phase 1)
- Optional cross-encoder reranking for top-N candidates.
- MMR diversification to reduce redundancy.
- Per-document caps to avoid over-indexing a single QBR.

Docs:
- `docs/retrieval.md`
- `docs/retrieval_phase1.md`

Key settings:
- `RERANK_BACKEND=cross_encoder|none`
- `RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`
- `RERANK_TOP_N=20`
- `RETRIEVAL_MMR_LAMBDA=0.5`
- `RETRIEVAL_MAX_CHUNKS_PER_DOC=3`

## Data Source (Using Existing QBR Data)
The agent reads QBR content from a SQLite database. You can point it at an existing
database without running extraction.

Minimum expected tables:
- `documents`
- `chunks`

To use the prebuilt DB shipped in this repo:
```
DATABASE_URL=sqlite+aiosqlite:///qbr_intelligence.db
```

Notes:
- If `chunks.embedding` is missing, vector retrieval returns no results.
- Text search still works when embeddings are missing.

## Run Tests

```bash
# Install dev deps
python -m pip install -e .[dev]

# Run tests
pytest
```

## WebSocket Contract Validation (Legacy Protocol)
Contract details: `docs/websocket-protocol.md`

Legacy WebSocket e2e test:
```bash
RUN_E2E=1 .venv/bin/python -m pytest tests/test_websocket_legacy_e2e.py
```

End-to-end runner (writes artifacts in `data/`):
```bash
.venv/bin/python scripts/run_legacy_websocket_e2e.py
```

## MLflow Tracing (Optional)
Enable per-interaction tracing with nested runs for retrieval and planner output.

Env:
- `MLFLOW_ENABLED=true`
- `MLFLOW_TRACKING_URI=...`
- `MLFLOW_EXPERIMENT=...`
- `MLFLOW_TRACING_ENABLED=true`

Docs:
- `docs/mlflow_tracing.md`

## Playground (Dev)

```bash
./scripts/penguiflow_dev.sh
```

### Playground UI

```bash
python -m ai_agent_qbr.playground_ui
```

## FAISS Retrieval
Build the FAISS index and run with `VECTOR_BACKEND=faiss`:

```bash
python -m pip install -e .[retrieval]
python scripts/build_faiss_index.py
VECTOR_BACKEND=faiss python app.py
```

To build on startup:
```
VECTOR_BACKEND=faiss
FAISS_AUTO_BUILD=true
FAISS_REBUILD_ON_STARTUP=false
```

## Add a New Storage Backend
1. Implement `KnowledgeRepository` and `VectorIndex` in `src/qbr_agent/infrastructure/`.
2. Register them in `qbr_agent.infrastructure.factory.build_infrastructure`.
3. Add env selectors (e.g., `STORAGE_BACKEND=postgres`).
4. Add integration tests for the new backend.

## Extraction Pipeline
Legacy extraction code and assets live in `qbr_extraction/`.

## Docs
- `docs/architecture.md`
- `docs/discovery.md`
- `docs/runbook.md`
