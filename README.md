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
- `OUTPUT_PROTOCOL=websocket|agui`
- `DATABASE_URL=sqlite+aiosqlite:///qbr_intelligence.db`
- `STORAGE_BACKEND=sqlite`
- `VECTOR_BACKEND=sqlite_embeddings`
- `EMBEDDINGS_BACKEND=sentence_transformers|hash`
- `EMBEDDINGS_MODEL=all-MiniLM-L6-v2`
- `RETRIEVAL_TOP_K=5`

## Architecture

See:
- `docs/discovery.md` for current system findings
- `docs/architecture.md` for the target clean architecture

## How Retrieval Works
1. Query is embedded via `EmbeddingsProvider`.
2. `VectorIndex` ranks stored chunk embeddings (SQLite or FAISS).
3. `KnowledgeRepository` returns chunk payloads and text matches.
4. Retrieved context is injected into the planner (`qbr_context`).

## Data Source (Using Existing QBR Data)
The agent reads QBR content from a SQLite database. You can point it at an existing
database without running extraction.

Minimum expected tables:
- `documents`
- `chunks`

To use the prebuilt DB shipped in this repo:
```
DATABASE_URL=sqlite+aiosqlite:///qbr_extraction/qbr_intelligence.db
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
