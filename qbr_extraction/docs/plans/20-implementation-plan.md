# Implementation Plan (ReactPlanner + WebSockets + SQLite RAG)

## Phase 0: Project Scaffolding (PenguiFlow First-Class)

- Create a local agent package in this repo (following PenguiFlow React template).
- Add a `planner.py` with ReactPlanner config and tool catalog registration.
- Add a minimal orchestrator to run planner in-process.
- Wire a WebSocket server (FastAPI) that calls the orchestrator.

Deliverables:
- `src/qbr_agent/` with `planner.py`, `orchestrator.py`, `tools/`.
- FastAPI entrypoint with `/ws/chat/{session_id}`.

## Phase 1: RAG Retrieval Baseline

- Build a `rag_retrieve` tool:
  - Uses SQLAlchemy async session.
  - Loads chunk embeddings and ranks via cosine similarity.
  - Returns top-k chunks + metadata.
- Build `list_documents` and `get_document` tools from `qbr_intelligence.query`.
- Build `answer_final` tool to construct responses with citations.

Deliverables:
- Tool catalog (`ModelRegistry` + `Node` list).
- Pydantic models for tool input/output.
- Unit tests for retrieval + ranking.

## Phase 2: WebSocket DTO Protocol

- Define input payload schema:
  - `message`
  - `metadata`: `document_id`, `client_name`, `period`, etc.
- Define output DTOs:
  - `ready`
  - `thinking`
  - `progress`
  - `final`
  - `error`
- Add a `parse_update()` step that maps planner events to DTOs.

Deliverables:
- DTO module + parser
- WebSocket message loop with keepalive pings
- Event callback wired from ReactPlanner to WebSocket streaming

## Phase 3: Memory + Session State

- Keep per-session state (recent turns, current document, filters).
- Optional: use PenguiFlow short-term memory hooks or custom session memory.
- Plan for pause/resume if we add HITL.

Deliverables:
- `SessionState` dataclass
- `SessionStateService` for persistence (in-memory or SQLite table)
- Option to persist planner state in SQLite (ReactPlanner state store)

## Phase 4: Vector Index Upgrade (Optional)

- Add FAISS index build + persistence.
- Add background index refresh.
- Switch `rag_retrieve` to use FAISS when available.

Deliverables:
- `VectorIndexService`
- Index build CLI
- Metrics for retrieval latency and hit rate

## Phase 5: Hardening + Ops

- Health endpoints (readiness / liveness).
- Observability: log planner events, query latency, chunk counts.
- Error handling: recoverable failures, graceful disconnects.

Deliverables:
- Health router
- Structured logs
- Runbook markdown
