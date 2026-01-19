"""Session registry for orchestrator reuse."""

from __future__ import annotations

from collections.abc import Callable

from ai_agent_qbr.orchestrator import AiAgentQbrOrchestrator
from ai_agent_qbr.telemetry import AgentTelemetry


class SessionRegistry:
    def __init__(
        self,
        *,
        orchestrator_factory: Callable[[AgentTelemetry], AiAgentQbrOrchestrator],
    ) -> None:
        self._orchestrator_factory = orchestrator_factory
        self._orchestrators: dict[str, AiAgentQbrOrchestrator] = {}
        self._telemetry: dict[str, AgentTelemetry] = {}

    def get_or_create(
        self,
        session_id: str,
        *,
        telemetry_factory: Callable[[], AgentTelemetry],
    ) -> AiAgentQbrOrchestrator:
        orchestrator = self._orchestrators.get(session_id)
        if orchestrator is not None:
            return orchestrator

        telemetry = telemetry_factory()
        orchestrator = self._orchestrator_factory(telemetry)
        self._orchestrators[session_id] = orchestrator
        self._telemetry[session_id] = telemetry
        return orchestrator

    def get_telemetry(self, session_id: str) -> AgentTelemetry | None:
        return self._telemetry.get(session_id)

    async def close_session(self, session_id: str) -> None:
        orchestrator = self._orchestrators.pop(session_id, None)
        if orchestrator is None:
            return
        self._telemetry.pop(session_id, None)
        orchestrator.clear_session(session_id=session_id)
        await orchestrator.stop()
