# Phase 1 Retrieval Enhancements

This document describes the Phase 1 improvements added to QBR retrieval: reranking, MMR
diversification, and per-document caps. These changes improve answer quality by promoting
the most relevant chunks while reducing redundancy from a single document.

## What Changed

- Added a reranker port and infrastructure implementation (cross-encoder + noop fallback).
- Extended hybrid retrieval to:
  - Rerank the top-N candidates.
  - Diversify with MMR when embeddings are available.
  - Enforce per-document caps for the final top-k.
- Added configuration flags for reranking and diversity controls.
- Wired the orchestrator so the planner tools and answer context use the new pipeline.
- Added tests to confirm reranking order and per-document caps.

## Configuration

```
# Hybrid scoring controls
RETRIEVAL_TEXT_WEIGHT=0.6
RETRIEVAL_VECTOR_WEIGHT=0.4
RETRIEVAL_CANDIDATE_MULTIPLIER=4

# Reranking
RERANK_BACKEND=cross_encoder
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_TOP_N=20
# RERANK_MAX_LENGTH=

# Diversity and caps
RETRIEVAL_MMR_LAMBDA=0.5
RETRIEVAL_MAX_CHUNKS_PER_DOC=3
```

Set `RERANK_BACKEND=none` to disable reranking.

## Flow Overview

1. Embed query.
2. Retrieve vector matches and text matches.
3. Combine scores with hybrid normalization.
4. Rerank top-N candidates (if enabled).
5. Apply MMR to reduce redundancy (if embeddings present).
6. Enforce per-document cap and return final top-k.

## Key Files

- `src/qbr_agent/application/ports.py` (reranker port)
- `src/qbr_agent/infrastructure/reranker.py` (cross-encoder + noop)
- `src/qbr_agent/application/use_cases.py` (rerank/MMR/doc-cap pipeline)
- `src/qbr_agent/infrastructure/factory.py` (reranker wiring)
- `src/ai_agent_qbr/orchestrator.py` (use case wiring)
- `docs/retrieval/retrieval.md` and `.env.example` (configuration docs)
