"""Application use cases for sessions and messaging."""

from __future__ import annotations

from collections.abc import Callable
import logging

from ai_agent_qbr.application.session_registry import SessionRegistry
from ai_agent_qbr.domain.models import SessionContext
from ai_agent_qbr.orchestrator import AgentResponse
from ai_agent_qbr.services.message_service import MessageService, get_message_service
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
    def __init__(
        self,
        registry: SessionRegistry,
        *,
        message_service: MessageService | None = None,
    ) -> None:
        self._registry = registry
        self._message_service = message_service or get_message_service()

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
        await self._persist_user_message(session, message)
        response = await orchestrator.execute(
            query=message,
            tenant_id=session.tenant_id,
            user_id=session.user_id,
            session_id=session.session_id,
        )
        await self._persist_assistant_message(session, response)
        return response

    async def _persist_user_message(self, session: SessionContext, message: str) -> None:
        try:
            await self._message_service.persist_message(
                session_id=session.session_id,
                message_type="user",
                content=message,
                user_email=_coerce_email(session.user_id),
                metadata={
                    "tenant_id": session.tenant_id,
                    "user_id": session.user_id,
                },
            )
        except Exception as exc:
            _LOGGER.warning("Failed to persist user message: %s", exc)

    async def _persist_assistant_message(
        self,
        session: SessionContext,
        response: AgentResponse,
    ) -> None:
        if not response.answer:
            return
        metadata = {"tenant_id": session.tenant_id, "user_id": session.user_id}
        if response.metadata:
            metadata.update(response.metadata)
        metadata["trace_id"] = response.trace_id
        try:
            await self._message_service.persist_message(
                session_id=session.session_id,
                message_type="assistant",
                content=response.answer,
                user_email=_coerce_email(session.user_id),
                artifacts=response.artifacts,
                metadata=metadata,
            )
        except Exception as exc:
            _LOGGER.warning("Failed to persist assistant message: %s", exc)


class CloseSessionUseCase:
    def __init__(self, registry: SessionRegistry) -> None:
        self._registry = registry

    async def execute(self, session_id: str) -> None:
        await self._registry.close_session(session_id)


def _coerce_email(value: str) -> str | None:
    if not value:
        return None
    return value if "@" in value else None
