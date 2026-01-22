#!/usr/bin/env markdown
# QBR Agent Improvements and System Overview

This document explains how the QBR agent works end-to-end and summarizes the
retrieval, reranking, and observability improvements currently implemented.
It is intended as the primary technical reference for how requests are handled,
how evidence is retrieved, and how answers are produced.

## 1) System Overview

### Purpose
The agent answers LG Ads QBR questions using a local knowledge base built from
historical PowerPoint decks. It uses retrieval‑augmented generation (RAG) with
hybrid retrieval and optional reranking to assemble grounded responses.

### High-Level Flow
1. The client sends a WebSocket `message` payload to `/ws/chat/{session_id}`.
2. The `AiAgentQbrOrchestrator` retrieves relevant QBR chunks and builds context.
3. The ReactPlanner (LLM planner) runs tools to produce the final response.
4. The WebSocket strategy streams progress and the final message back to the client.

## 2) Architecture and Components

### Layers
- **Domain**: Entities like `Document`, `Chunk`, and `RetrievalResult` live in
  `src/qbr_agent/domain`.
- **Application**: Use cases such as `HybridSearchKnowledge` and `AnswerQuestion`
  live in `src/qbr_agent/application`.
- **Infrastructure**: SQLite/FAISS adapters, embedding providers, and rerankers
  live in `src/qbr_agent/infrastructure`.
- **Interface**: WebSocket transport and planner wiring live in `src/ai_agent_qbr`.

### Key Services
- **AiAgentQbrOrchestrator**: Orchestrates retrieval + planner and formats outputs.
  File: `src/ai_agent_qbr/orchestrator.py`.
- **WebSocket Transport**: Handles the legacy contract and status events.
  File: `src/ai_agent_qbr/transport/websocket`.
- **ReactPlanner**: Tool-driven planner from PenguiFlow.
  File: `src/ai_agent_qbr/planner.py`.

## 3) Retrieval Pipeline (Phase 1)

### 3.1 Hybrid Retrieval
Two sources contribute scores:
- **Vector search**: cosine similarity of embeddings (SQLite or FAISS).
- **Text search**: FTS5 BM25 ranking over chunk content (LIKE fallback).

Scores are normalized and combined:
```
hybrid = (text_norm * RETRIEVAL_TEXT_WEIGHT) + (vector_norm * RETRIEVAL_VECTOR_WEIGHT)
```

### 3.2 Reranking
Top‑N hybrid candidates are optionally reranked with a cross‑encoder:
- Default model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Reranker lives in `src/qbr_agent/infrastructure/reranker.py`

### 3.3 MMR Diversification
MMR reduces redundancy among candidate chunks:
```
score = lambda * sim(query, candidate) - (1 - lambda) * max(sim(candidate, selected))
```

### 3.4 Per-Document Caps
The final top‑k output enforces a maximum per‑document count to prevent a single
QBR deck from dominating the context.

### 3.5 Retrieval Debug Output
Hybrid/QA use cases retain a `last_debug` payload with:
- Vector matches
- Text matches
- Hybrid combined scores
- Rerank scores
- MMR selections
- Final chunk IDs

This is logged to MLflow (see Observability section).

## 4) Planner and Tooling

### Tools
- `search_documents`: Calls `HybridSearchKnowledge` and returns chunk snippets
  for the planner. File: `src/ai_agent_qbr/tools/search.py`.
- `analyze_results`: Summarizes results into a final answer.
  File: `src/ai_agent_qbr/tools/analyze.py`.
- `agent_capabilities`: Returns capability overview for user help.
  File: `src/ai_agent_qbr/tools/agent_capabilities.py`.

### Planner
ReactPlanner uses these tools and the `qbr_context` injected by the orchestrator.
The planner emits telemetry events that the WebSocket strategy maps into progress
messages for the client.

## 5) WebSocket Contract (Legacy)

The legacy WebSocket contract is documented at `docs/websocket-protocol.md`.
Key message types:
- `ready`: session opened
- `thinking`: progress updates
- `user_message`: echo of user input
- `final`: final response

The WebSocket strategy preserves compatibility with existing ADS clients while
supporting richer telemetry messages.

## 6) Observability and MLflow Tracing

### Runs (Artifacts)
Each interaction can log:
- `qbr_context.txt`
- `citations.json`
- `retrieval_debug.json`
- `final_answer.txt`

### Traces (Span View)
When tracing is enabled, MLflow captures:
- `interaction` root span
- `retrieval` child span (inputs/outputs include context and debug)
- `planner` child span (inputs/outputs include final answer)

Docs: `docs/mlflow_tracing.md`

## 7) End-to-End Validation

### Legacy WebSocket E2E Test
Use the test to validate the contract and reranker invocation:
```
RUN_E2E=1 .venv/bin/python -m pytest tests/test_websocket_legacy_e2e.py
```

### Runner Script
The runner script seeds a sample DB and captures the full WebSocket exchange:
```
.venv/bin/python scripts/run_legacy_websocket_e2e.py
```

Artifacts:
- `data/legacy_websocket_e2e.jsonl`
- `data/legacy_websocket_e2e_summary.txt`

## 8) Key Config Knobs

Retrieval:
- `RETRIEVAL_TOP_K`
- `RETRIEVAL_TEXT_WEIGHT`
- `RETRIEVAL_VECTOR_WEIGHT`
- `RETRIEVAL_MMR_LAMBDA`
- `RETRIEVAL_MAX_CHUNKS_PER_DOC`
- `RETRIEVAL_CANDIDATE_MULTIPLIER`
- `TEXT_SEARCH_BACKEND` (fts5/auto/like)

Rerank:
- `RERANK_BACKEND=cross_encoder|none`
- `RERANK_MODEL`
- `RERANK_TOP_N`
- `RERANK_MAX_LENGTH`

Observability:
- `MLFLOW_ENABLED`
- `MLFLOW_TRACING_ENABLED`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_EXPERIMENT`

## 9) Source Map

- Retrieval use cases: `src/qbr_agent/application/use_cases.py`
- Reranker: `src/qbr_agent/infrastructure/reranker.py`
- Orchestrator: `src/ai_agent_qbr/orchestrator.py`
- WebSocket protocol: `docs/websocket-protocol.md`
- MLflow tracing: `docs/mlflow_tracing.md`
