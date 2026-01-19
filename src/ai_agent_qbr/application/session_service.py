"""Application use cases for sessions and messaging."""

from __future__ import annotations

from collections.abc import Callable
import logging

from ai_agent_qbr.application.session_registry import SessionRegistry
from ai_agent_qbr.domain.models import SessionContext
from ai_agent_qbr.orchestrator import AgentResponse
from ai_agent_qbr.telemetry import AgentTelemetry

_LOGGER = logging.getLogger("uvicorn.error")


class StartSessionUseCase:
    def __init__(self, registry: SessionRegistry) -> None:
        self._registry = registry

    async def execute(
        self,
        *,
        session: SessionContext,
        telemetry_factory: Callable[[], AgentTelemetry],
    ) -> None:
        _LOGGER.info(
            "Starting session use case for %s (tenant=%s, user=%s)",
            session.session_id,
            session.tenant_id,
            session.user_id,
        )
        orchestrator = self._registry.get_or_create(
            session.session_id,
            telemetry_factory=telemetry_factory,
        )
        await orchestrator.start_session(
            tenant_id=session.tenant_id,
            user_id=session.user_id,
            session_id=session.session_id,
        )


class SendMessageUseCase:
    def __init__(self, registry: SessionRegistry) -> None:
        self._registry = registry

    async def execute(
        self,
        *,
        session: SessionContext,
        message: str,
        telemetry_factory: Callable[[], AgentTelemetry],
    ) -> AgentResponse:
        orchestrator = self._registry.get_or_create(
            session.session_id,
            telemetry_factory=telemetry_factory,
        )
        return await orchestrator.execute(
            query=message,
            tenant_id=session.tenant_id,
            user_id=session.user_id,
            session_id=session.session_id,
        )


class CloseSessionUseCase:
    def __init__(self, registry: SessionRegistry) -> None:
        self._registry = registry

    async def execute(self, session_id: str) -> None:
        await self._registry.close_session(session_id)
