# Architecture Options (PenguiFlow First-Class)

## Goal Recap

- Build a PenguiFlow-first agent that answers user questions via WebSocket.
- RAG over `qbr_intelligence.db` (SQLite) as the primary knowledge source.
- Use ReactPlanner with a node catalog for planning, retrieval, reasoning, and response.
- WebSocket integration should follow the production-ready patterns from `ai-agent-reporting`.

## Option A: Minimal In-Process RAG (SQLite + Cosine)

### Summary
Compute embeddings (already stored), load vectors in-process, perform cosine similarity for top-k, then respond.

### Pros
- Minimal infrastructure
- Fast local iteration
- All logic in one process

### Cons
- No vector index acceleration (scales poorly)
- Memory-heavy if DB grows large

### Fit
Best for early phase and local development.

## Option B: Lightweight Vector Index (FAISS On Disk)

### Summary
Build a FAISS index from `chunks.embedding` and persist to disk; use it for retrieval.

### Pros
- Fast retrieval
- Still local-friendly

### Cons
- Requires index rebuild or incremental updates
- Adds binary dependency and lifecycle

### Fit
Good for medium scale and still single-host.

## Option C: External Vector Store (pgvector / Qdrant)

### Summary
Move embeddings to an external vector store and query by vector search.

### Pros
- Scales cleanly
- Supports metadata filtering

### Cons
- Infrastructure dependency
- More ops for early phase

### Fit
Best once the workflow is stable.

## WebSocket Integration Options

### Option 1: Direct FastAPI WebSocket (ai-agent-reporting pattern)
- Mirror the `websocket_service` style:
  - Session hydration + ready message
  - Ping/pong keepalive
  - Streaming updates as DTOs
- Works well with ReactPlanner event callbacks.

### Option 2: AG-UI Adapter + WebSocket Wrapper
- Use `penguiflow.agui_adapter.PenguiFlowAdapter` to emit standard events.
- Wrap those AG-UI events into your WebSocket payloads.

### Fit
Option 1 first; Option 2 if we need AG-UI compatibility.

## Recommended Path (Phase 1)

- Use Option A for retrieval (in-process cosine).
- Implement a WebSocket endpoint + connection manager aligned with ai-agent-reporting.
- Define a PenguiFlow tool catalog for:
  - `rag_retrieve`
  - `rag_summarize`
  - `answer_final`
  - `list_documents`
  - `filter_by_doc`
