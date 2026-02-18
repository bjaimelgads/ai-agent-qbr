# Architecture Discovery

## Scope
This document captures how the existing base agent streams output (AG-UI + legacy WebSocket), how QBR knowledge is stored and queried today, and where the entrypoints/configuration/DI seams live so we can refactor toward clean architecture without coupling to SQLite.

## Base Agent (ai-agent-base)

### What it is
A FastAPI WebSocket server that hosts a PenguiFlow-based agent. It supports two output protocols over the same WebSocket connection:
- `OUTPUT_PROTOCOL=legacy` for legacy message DTOs.
- `OUTPUT_PROTOCOL=agui` for AG-UI event streams.

### Streaming flow and AG-UI structure
Key files:
- `src/app.py` wires the FastAPI app, session services, and output strategy.
- `src/transport/websocket/service.py` manages the WebSocket lifecycle, keepalive pings, and dispatches inbound messages.
- `src/transport/websocket/strategies/agui.py` maps inbound payloads to `RunAgentInput` and streams AG-UI events.
- `src/transport/agui/adapter.py` translates PenguiFlow planner events into AG-UI events.

AG-UI event shape (from `docs/ag_ui.md`):
- `RUN_STARTED`, `STEP_STARTED`, `TEXT_MESSAGE_START`, `TEXT_MESSAGE_CONTENT`, `TEXT_MESSAGE_END`, `RUN_FINISHED`.
- Planner events (tool calls, artifacts, state updates) are emitted as `CUSTOM` events.

Adapter behavior:
- `llm_stream_chunk` events with channel `answer` become `TEXT_MESSAGE_*` deltas.
- `stream_chunk` events are mapped to `CUSTOM` `thinking` events.
- `artifact_chunk`, `artifact_stored`, and `resource_updated` are forwarded as `CUSTOM` events with URLs.

### Entrypoints
- WebSocket server: `src/app.py` + `src/server.py` (uvicorn).
- WebSocket endpoint: `/ws/chat/{session_id}`.

### Configuration surface
- `src/config.py` loads environment vars. Key ones:
  - `OUTPUT_PROTOCOL` (`legacy` or `agui`).
  - `PLANNER_STREAM_FINAL_RESPONSE` for native streaming.
  - LLM provider configuration (`LLM_MODEL`, Databricks settings).
  - Memory store knobs for short-term memory and platform-backed memory.

### Dependencies
- PenguiFlow planner + telemetry.
- AG-UI protocol adapter.
- Memory store (in-memory or platform-backed).

### Testing approach
- Unit tests: mapping functions in `transport/agui/adapter.py` and `transport/websocket/dto.py`.
- Integration tests: start the FastAPI app and assert WebSocket stream events for both output protocols.

## QBR Project (ai-agent-qbr)

### Knowledge store and extraction pipeline
- Data is stored in SQLite (`qbr_intelligence.db`) using SQLAlchemy models in `src/qbr_intelligence/db/models.py`.
- Embeddings are stored as JSON arrays in `chunks.embedding` with `chunks.embedding_model` metadata.
- Extraction pipeline uses Kreuzberg embeddings and optional SentenceTransformers fallback:
  - `src/qbr_intelligence/pipeline/processor.py`
  - `src/qbr_intelligence/pipeline/embeddings.py`
  - `src/qbr_intelligence/pipeline/post_embeddings.py`

### Query path today
- `src/qbr_intelligence/query/interface.py` exposes `QBRQueryInterface`.
- `search_content()` performs a simple text search (`Chunk.content ILIKE`), not vector search.
- Tool helpers live in `src/qbr_intelligence/query/tools.py` (document summaries, metrics, entities, etc.).

### Vector database status
- No dedicated vector database is currently wired. Embeddings are stored in SQLite and not queried for similarity. The current search path is text-only.

### Entrypoints
- WebSocket server: `app.py` (uvicorn) -> `src/ai_agent_qbr/api/app.py`.
- WebSocket endpoint: `/ws/chat/{session_id}`.
- CLI: `src/ai_agent_qbr/__main__.py` (runs orchestrator directly).
- Extraction pipeline: `qbr_extraction/qbr_pipeline/pipelines/extraction/run_extraction_pipeline.py` and `qbr_extraction/qbr_pipeline/pipelines/extraction/scripts/extract_kreuzberg_standalone.py`.

### Configuration surface
- `src/ai_agent_qbr/config.py` loads environment vars for:
  - LLM model and Databricks config.
  - WebSocket keepalive timeouts.
  - Short-term memory in PenguiFlow.
- `.env.example` includes extraction settings such as `DATABASE_URL` and Kreuzberg embedding flags.
- Current WebSocket output is legacy (no AG-UI adapter in this repo yet).

### Dependencies
- PenguiFlow planner for orchestration.
- SQLAlchemy async engine for QBR data.
- Kreuzberg + SentenceTransformers for embeddings during extraction.

### Testing approach
- Unit tests exist in `qbr_extraction/tests` for embeddings settings.
- No WebSocket streaming or retrieval tests currently cover the QBR agent runtime.

## Reference: pengui_iceberg
The `pengui_iceberg` repo contains mature retrieval patterns useful for the QBR agent:
- Vector index interface and implementations:
  - `src/pengui_iceberg/retrieval/faiss_index.py` (FAISS)
  - `src/pengui_iceberg/retrieval/pgvector_index.py` (pgvector)
- Prompt embedding utilities and hybrid search flow:
  - `src/pengui_iceberg/memory/prompt_embeddings.py`
  - `src/pengui_iceberg/memory/auto_retrieve_flow.py`

These can guide port definitions (ports, adapters) without coupling the domain to a specific backend.

## Gaps Identified
- The runtime agent in this repo does not use QBR knowledge (no retrieval step).
- Embeddings are stored but not used for vector similarity search.
- AG-UI streaming support exists in ai-agent-base but is not present in ai-agent-qbr.
- Current logic is coupled to direct SQLAlchemy access without a clean architecture boundary.

## Current Retrieval (Updated)
- Hybrid retrieval now runs for each query: vector similarity + text match over chunks.
- FAISS can be used as the vector backend (`VECTOR_BACKEND=faiss`) after building an index.
- Retrieved context is injected into planner prompts as `qbr_context`.
