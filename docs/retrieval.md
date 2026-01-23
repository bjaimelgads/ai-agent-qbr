# Retrieval

## Hybrid Retrieval (QBR)
The agent uses a hybrid retrieval strategy inspired by `pengui_iceberg`:
- Embed the query.
- Retrieve vector matches (ANN).
- Retrieve text matches (FTS5 BM25 over chunk content).
- Normalize scores and combine (`text_weight`, `vector_weight`).
- Rerank top candidates with a cross-encoder (optional).
- Diversify with MMR and apply per-document caps.
- Return the top-k chunks for prompting.

## Data Source (No Extraction Required)
Retrieval reads from a SQLite database that already contains QBR documents and chunks.
If you have a prebuilt DB, you can point the agent at it and skip extraction entirely.

Minimum tables required for retrieval:
- `documents`
- `chunks`

Example (repo-shipped DB):
```
DATABASE_URL=sqlite+aiosqlite:///qbr_extraction/qbr_intelligence.db
```

If `chunks.embedding` is empty, vector matches are skipped and only text matches contribute
to the hybrid score.

Implementation:
- Use case: `src/qbr_agent/application/use_cases.py` (`HybridSearchKnowledge`).
- Text search: `src/qbr_agent/infrastructure/sqlalchemy_repository.py` (FTS5 BM25 with LIKE fallback).
- Vector search: `src/qbr_agent/infrastructure/vector_index.py` or FAISS.

## FAISS Vector Index

### Build the index
```bash
# Install retrieval extras
python -m pip install -e .[retrieval]

# Build FAISS index from stored chunk embeddings
python scripts/build_faiss_index.py
```

Optional environment variables:
- `FAISS_DIR=./data/faiss`
- `FAISS_NORMALIZE=true`
- `FAISS_INDEX_TYPE=Flat`
- `FAISS_METRIC=ip`
- `FAISS_EMBEDDING_MODEL_FILTER=kreuzberg:fast`

### Run with FAISS
Set the vector backend in `.env`:
```
VECTOR_BACKEND=faiss
FAISS_AUTO_BUILD=true
```

If `index.faiss` or `index_ids.json` are missing, the runtime will log a warning and return no vector results.
Set `FAISS_REBUILD_ON_STARTUP=true` to force rebuilding on every boot.

## Text Search (FTS5 + BM25)

SQLite FTS5 provides BM25 ranking for chunk text. Scores are converted to
`1 / (1 + bm25)` so higher scores rank higher in hybrid normalization.

To build the FTS table and triggers:

```bash
python scripts/build_fts_index.py
```

Configuration:

```
TEXT_SEARCH_BACKEND=fts5
```

Supported values:
- `fts5` (default)
- `auto` (use FTS5 if available, otherwise LIKE)
- `like` (legacy fallback)

## Hybrid Scoring
The combined score is computed as:
```
hybrid = (text_norm * RETRIEVAL_TEXT_WEIGHT) + (vector_norm * RETRIEVAL_VECTOR_WEIGHT)
```

Tune these weights via environment variables.

## Reranking and Diversity

Optional reranking via sentence-transformers cross-encoder:

```
RERANK_BACKEND=cross_encoder
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_TOP_N=20
RERANK_MAX_LENGTH=
```

After reranking, MMR and per-document caps shape the final top-k:

```
RETRIEVAL_MMR_LAMBDA=0.5
RETRIEVAL_MAX_CHUNKS_PER_DOC=3
```

Set `RERANK_BACKEND=none` to disable reranking.

## Cross-Deck Comparison Retrieval

When a question requires comparing multiple decks, the planner can call a
dedicated intent tool and pass the signal into retrieval.

Tool:
- `detect_comparison_intent` (`src/ai_agent_qbr/tools/comparison_intent.py`)
  - Returns `comparison_intent=true` when the question indicates a cross-deck
    comparison (e.g., "compare", "vs", "between decks").

Behavior:
1. Stage 1 runs the normal hybrid search across all chunks to identify top
   candidate documents.
2. Stage 2 runs per-document retrieval for the top documents in parallel and
   merges the results.

Parallelism:
- Stage 2 executes per-document retrieval concurrently via `asyncio.gather` in
  `src/ai_agent_qbr/tools/search.py`.

Configuration (env):
- `COMPARISON_TOP_DOCS` (default `3`): number of documents to compare.
- `COMPARISON_PER_DOC_K` (default `3`): chunks per document in stage 2.
- `COMPARISON_STAGE1_TOP_K` (optional): override stage-1 top-k selection.

Planner integration:
- The planner calls `detect_comparison_intent` and, when true, passes
  `comparison_intent=true` into `search_documents` (see
  `src/ai_agent_qbr/tools/search.py`).
