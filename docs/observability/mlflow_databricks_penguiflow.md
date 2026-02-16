# MLflow on Databricks for PenguiFlow Agents

This guide describes a reusable pattern to instrument any PenguiFlow-based agent with MLflow runs and traces stored in Databricks.

## 1. What You Need

- A Python agent using PenguiFlow planner execution.
- `mlflow` installed in the runtime environment.
- Databricks workspace access from the runtime where the agent runs.
- A target MLflow experiment path (for example: `/Shared/your-agent-experiment`).

Recommended dependency layout:

```toml
[project]
dependencies = [
  "penguiflow[planner]>=2.x",
]

[project.optional-dependencies]
observability = [
  "mlflow>=2.13.0",
]
```

Important: if you deploy using exported requirements, include the `observability` extra so `mlflow` is actually present at runtime.

## 2. Required Environment Variables

Use these variables in every environment where tracing is expected:

```bash
MLFLOW_ENABLED=true
MLFLOW_TRACING_ENABLED=true
MLFLOW_TRACKING_URI=databricks
MLFLOW_EXPERIMENT=/Shared/your-agent-experiment
```

Notes:
- `MLFLOW_TRACKING_URI=databricks` is required to write to Databricks tracking.
- If `MLFLOW_TRACKING_URI` is omitted, MLflow may default to local storage and traces will not appear in Databricks Experiments UI.

## 3. Minimal MLflow Bootstrap

Create one small observability module that:

1. Imports MLflow only when enabled.
2. Sets tracking URI and experiment once.
3. Wraps interaction execution in a parent run.
4. Creates spans for key phases (retrieval/planner/tooling).
5. Logs structured artifacts (`*.json`) and text artifacts (`*.txt`).

Example skeleton:

```python
import mlflow


def configure_mlflow(tracking_uri: str | None, experiment: str | None) -> None:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if experiment:
        mlflow.set_experiment(experiment)


def run_interaction(trace_id: str, tags: dict[str, str]):
    with mlflow.start_run(run_name=f"interaction-{trace_id}", tags=tags):
        with mlflow.start_span(name="interaction", span_type="AGENT") as span:
            span.set_inputs({"trace_id": trace_id})
            # retrieval, planning, tool calls...
```

## 4. PenguiFlow Integration Points

Instrument at orchestration level, not inside every tool implementation.

Recommended hooks:
- Wrap `planner.run(...)` with MLflow run/span context.
- Use PenguiFlow planner event callback to capture tool lifecycle events.

Tool I/O capture pattern:
- Listen for `tool_call_start` -> record tool name + input arguments.
- Listen for `tool_call_result` -> record tool output.
- Listen for `tool_call_end` -> finalize status.
- Save collected records as one artifact, for example `tool_calls.json`.

This keeps tools decoupled from MLflow and avoids instrumentation duplication.

## 5. Recommended Run/Span Structure

Use a predictable hierarchy so every agent is easy to debug:

- Parent run: one per user interaction
  - Tags: `tenant_id`, `user_id`, `session_id`, `trace_id`, `agent_name`, `env`
  - Params: runtime model settings, retrieval settings, planner mode flags
  - Artifacts: `query.txt`, optional normalized context

- Nested run: `retrieval`
  - Artifacts: citations, scoring/debug payloads, selected chunks

- Nested run: `planner`
  - Artifacts: final answer payload, planner payload, `tool_calls.json`

- Spans (MLflow tracing)
  - `interaction` (root)
  - `retrieval`
  - `planner`
  - Optional: per-step/per-tool spans if you need finer latency attribution

## 6. Databricks Authentication Expectations

Your runtime must be authorized to write tracking data to the workspace.

Typical setups:
- Databricks-managed runtime/app identity: workspace auth is available by default.
- External runtime: provide Databricks host + token/OAuth credentials according to your platform standard.

If authentication is broken, you will usually see MLflow warnings/errors in application logs.

## 7. Deployment Checklist

Before deploy:
- `mlflow` is present in final runtime dependencies.
- `MLFLOW_TRACKING_URI=databricks` is set.
- `MLFLOW_EXPERIMENT` points to an accessible path.
- `MLFLOW_ENABLED` and `MLFLOW_TRACING_ENABLED` are `true`.

After deploy:
- Send at least one real request through the agent.
- Verify in logs that MLflow is enabled and configured for Databricks URI.
- Open Databricks Experiments UI and confirm new runs/traces under the configured experiment.

## 8. Validation Queries

Use one deterministic test prompt and verify these artifacts exist:

- `query.txt`
- planner payload artifact
- `tool_calls.json`
- retrieval artifacts (if retrieval path was used)

Also verify trace spans include expected inputs/outputs.

## 9. Troubleshooting

### No runs/traces in Databricks UI

Common causes:
- Tracking URI not set to Databricks (`MLFLOW_TRACKING_URI` missing or wrong).
- Experiment path typo or no permission to write there.
- No post-deploy traffic yet.
- Runtime missing `mlflow` package.

### Runs exist but no tool input/output

Common causes:
- Planner event callback is not wired.
- Event names not mapped (`tool_call_start`, `tool_call_result`, `tool_call_end`).
- Tool I/O collector not persisted as artifact.

### Spans missing but runs exist

Common causes:
- Tracing not enabled (`MLFLOW_TRACING_ENABLED=false`).
- `mlflow.tracing.enable()` not called in bootstrap.

## 10. Operational Best Practices

- Keep PII/secrets out of artifacts; redact before logging.
- Log stable JSON schemas for artifacts so downstream analysis remains robust.
- Add a version tag (for example `agent_version` or git SHA) to every run.
- Keep per-interaction payload sizes bounded to avoid noisy and expensive logs.
- Prefer orchestration-level instrumentation over tool-by-tool manual logging.
