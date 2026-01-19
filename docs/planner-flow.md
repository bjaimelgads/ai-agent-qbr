# Planner Flow

ReactPlanner is the core runtime. The orchestrator constructs the planner with:

- A tool catalog (search + analyze stubs).
- A ModelRegistry mapping tool input/output schemas.
- A scripted LLM stub that returns deterministic planner steps.

## Sequence

1. WebSocket receives a `message` payload.
2. `AiAgentQbrOrchestrator.execute` runs the ReactPlanner with telemetry.
3. ReactPlanner emits planner events during the run.
4. The event callback maps events to `thinking` DTOs.
5. The orchestrator normalizes the final planner output for the API.

## Notes

- Phase 0 does not include RAG or database access.
- Tool catalog and ModelRegistry are intentionally minimal but wired for future phases.
