# Context Findings (PenguiFlow + WebSocket Reference)

## PenguiFlow (ReactPlanner) Core Observations

- The project already depends on `penguiflow[planner]` in `pyproject.toml`, so ReactPlanner is the expected first-class runtime.
- ReactPlanner is JSON-only ReAct with pause/resume, short-term memory summarization, planner events, and tool catalog integration.
- Planner runs are configured through a node catalog + registry:
  - Nodes are `Node(async_fn, name=...)`.
  - Models registered with `ModelRegistry`.
  - Decorated tools via `@tool` from `penguiflow.catalog`.
- There is a built-in template scaffold for a ReactPlanner agent (planner, tools, orchestrator, telemetry).
- The planner supports an event callback for streaming status and artifacts, suitable for wiring into WebSockets.

Key source references:
- `pyproject.toml` (penguiflow dependency)
- `penguiflow/planner/react.py` (planner loop, pause/resume, event emission)
- `penguiflow/templates/new/react/*` (reference scaffolding)
- `penguiflow/agui_adapter/penguiflow.py` (AG-UI adapter; useful pattern for streaming events)

## ai-agent-reporting WebSocket Patterns

The reference implementation provides a production-grade WebSocket loop that we can adapt:

- WebSocket endpoint: `/ws/chat/{session_id}` with origin validation and session-based logging.
- Lifecycle:
  1. `websocket.accept()`
  2. user auth and session hydration
  3. resource manager setup
  4. send `OutputReadyDTO`
  5. message loop with ping/pong keepalive and timeouts
  6. stream updates and final responses
- Message payload from client: JSON with `message` and `metadata`.
- Health/robustness: retries, exponential backoff on resource acquisition, recoverable error handling.
- Updates are converted to DTOs via `parse_update` and sent to frontend in JSON.

Key source references:
- `ai-agent-reporting/app.py`
- `ai-agent-reporting/src/pengui_chart_agent/api/routers/websocket.py`
- `ai-agent-reporting/src/pengui_chart_agent/services/websocket_service.py`
- `ai-agent-reporting/docs/README_LOCAL_DEVELOPMENT.md`

## QBR Data (RAG Inputs)

Existing QBR storage is in SQLite with chunk text and embeddings:
- Chunks stored in `qbr_intelligence.db` table `chunks`.
- Each chunk has `content`, optional `embedding`, and `embedding_model`.
- Current query interface includes a simple text search that can be replaced with vector search.

Key source references:
- `qbr_intelligence/db/models.py` (Chunk schema)
- `qbr_intelligence/query/interface.py` (search_content and document listing)
- `qbr_intelligence/pipeline/post_embeddings.py` (SentenceTransformers fallback)
