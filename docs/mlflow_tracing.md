# MLflow Tracing

The orchestrator can emit MLflow runs for each user interaction when enabled.
This provides a full trace of retrieval and planner output, including rerank/MMR
debug data when available.

## Enable

Set environment variables:

```
MLFLOW_ENABLED=true
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT=ai-agent-qbr
MLFLOW_TRACING_ENABLED=true
```

## Run Structure

- Parent run: `interaction-{trace_id}`
  - Tags: `tenant_id`, `user_id`, `session_id`, `trace_id`, `output_protocol`
  - Params: retrieval weights, rerank settings, top-k configuration
  - Artifacts: `query.txt`

- Nested run: `retrieval`
  - Artifacts: `qbr_context.txt`, `citations.json`, `retrieval_debug.json`

- Nested run: `planner`
  - Artifacts: `final_answer.txt`, `planner_payload.json`

## Retrieval Debug Fields

The `retrieval_debug.json` artifact includes:
- Vector match scores
- Text match scores (BM25 via FTS5, when enabled)
- Hybrid combined scores
- Rerank scores (if enabled)
- MMR-selected chunk IDs
- Final chunk IDs returned to the planner

## Trace View

When tracing is enabled, MLflow will record spans for:
- `interaction` (root span)
- `retrieval` (retriever span)
- `planner` (LLM span)

Each span includes inputs/outputs such as context, citations, and debug scores.
