"""WebSocket protocol strategy for legacy ADS chat payloads."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ai_agent_qbr.application.session_registry import SessionRegistry
from ai_agent_qbr.application.session_service import SendMessageUseCase
from ai_agent_qbr.domain.models import SessionContext
from ai_agent_qbr.transport.websocket.dto import parse_planner_event
from ai_agent_qbr.transport.websocket.schemas import (
    ChatRequest,
    OutputError,
    OutputFinal,
    OutputReady,
    OutputThinking,
    OutputUserMessage,
)

from .base import SendJson, WebsocketOutputStrategy


class LegacyWebsocketOutputStrategy(WebsocketOutputStrategy):
    """Maintain the existing ADS WebSocket response schema."""

    def __init__(
        self,
        *,
        sender: SendJson,
        send_message: SendMessageUseCase,
        session_registry: SessionRegistry,
        logger: logging.Logger | None = None,
    ) -> None:
        self._send = sender
        self._send_message = send_message
        self._session_registry = session_registry
        self._logger = logger or logging.getLogger(__name__)

    async def on_connect(self, session_id: str) -> None:
        await self._send_payload(session_id, OutputReady(session_id=session_id))

    async def on_message(self, session_id: str, message: Any) -> None:
        try:
            request = ChatRequest.model_validate(message)
        except Exception as exc:
            await self._send(session_id, OutputError(message=str(exc)))
            return

        await self._send_payload(session_id, OutputThinking(message="Planning response"))
        await self._send_payload(
            session_id,
            OutputUserMessage(data={"content": request.message, "session_id": session_id}),
        )

        telemetry = self._session_registry.get_telemetry(session_id)
        if telemetry is None:
            await self._send(session_id, OutputError(message="Missing session telemetry"))
            return

        def status_callback(msg: str | None, step: int | None) -> None:
            self._logger.debug("Status update: %s", msg)
            self._send_safe(session_id, OutputThinking(message=msg, step=step))

        def event_callback(event: Any) -> None:
            update = parse_planner_event(event)
            if update is None:
                return
            self._send_safe(session_id, update)

        try:
            with telemetry.subscribe(status_callback=status_callback, event_callback=event_callback):
                response = await self._send_message.execute(
                    session=SessionContext(
                        tenant_id=(request.metadata.tenant_id if request.metadata else "public"),
                        user_id=(request.metadata.user_id if request.metadata else "user"),
                        session_id=session_id,
                    ),
                    message=request.message,
                    telemetry_factory=lambda: telemetry,
                )
        except Exception as exc:
            await self._send_payload(session_id, OutputError(message=str(exc)))
            return

        await self._send_payload(session_id, OutputFinal(data={"content": response.answer or ""}))

    async def on_disconnect(self, session_id: str) -> None:
        self._logger.info("WebSocket disconnected: %s", session_id)

    def _send_safe(self, session_id: str, payload: Any) -> None:
        try:
            asyncio.create_task(self._send_payload(session_id, payload))
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.warning("Failed to send telemetry update: %s", exc)

    async def _send_payload(self, session_id: str, payload: Any) -> None:
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()
        await self._send(session_id, payload)
