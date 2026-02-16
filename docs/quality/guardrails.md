# Guardrails Implementation

This project uses PenguiFlow guardrails to route and block requests before the planner responds.

## Overview

Guardrails are applied at the ReactPlanner layer. The SLM Router endpoint classifies each user request into routes. Off-topic or jailbreak routes are blocked with a STOP decision. Heuristic jailbreak patterns and PenguiFlow injection rules are also applied.

## Core Files

- `src/ai_agent_qbr/infrastructure/guardrails.py`
  - Rule definitions
  - Databricks endpoint normalization
  - Workspace token resolution for router calls
  - Guardrail gateway wiring
- `src/ai_agent_qbr/infrastructure/databricks.py`
  - WorkspaceClient-based token extraction
- `src/ai_agent_qbr/planner.py`
  - `build_guardrail_gateway(config)` passed to `ReactPlanner`
- `src/ai_agent_qbr/config.py`
  - Guardrail environment variables and validation

## Active Rules

- `HeuristicJailbreakRule`
  - Regex-based detection for common jailbreak attempts.
- `InjectionPatternRule`
  - Built-in PenguiFlow prompt-injection rule.
- `SLMRouterRule`
  - Calls the Databricks SLM router endpoint to classify requests.
  - Blocks `off_topic` and `prompt_injection_jailbreak` routes.
- `JailbreakSentinelRule` (optional)
  - Runs only when `GUARDRAILS_JAILBREAK_ENDPOINT` is configured.

## Endpoint Handling

The router endpoint is normalized in code:

- `/ml/endpoints/.../metrics` is converted to `/serving-endpoints/.../invocations`

This matches the Databricks serving endpoint invocation format.

## Token Resolution

Order of precedence:

1. `GUARDRAILS_ROUTER_TOKEN`
2. `DATABRICKS_TOKEN`
3. WorkspaceClient token (service principal in local mode, runtime token in Databricks)

WorkspaceClient extraction uses `workspace_client.config.authenticate()` and handles OAuth M2M.

## Configuration

### Environment variables

```
GUARDRAILS_ROUTER_ENDPOINT=https://dbc-3b4bc42a-bf11.cloud.databricks.com/ml/endpoints/slm-router/metrics?o=185296477739056
GUARDRAILS_ROUTER_TOKEN=
GUARDRAILS_MODE=enforce
GUARDRAILS_ENABLED=true
```

### `deploy/dev/dev-qbr.app.yaml`

The guardrail endpoint can be set directly in the deployment config.

## Quick Verification

1. Send an off-topic message (e.g., "tell me a joke").
2. Confirm the router logs show `route=off_topic` and `decision=off_topic`.

If you need additional routes or policies, update `ROUTE_CONFIG` and the STOP conditions in `SLMRouterRule`.
