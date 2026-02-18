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
    OutputPartial,
    OutputReady,
    OutputThinking,
    OutputUserMessage,
    PlannerEventPayload,
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
        self._partial_buffers: dict[str, str] = {}

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
            if self._handle_stream_chunk(session_id, event):
                return
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
            self._logger.exception("WebSocket run failed")
            self._partial_buffers.pop(session_id, None)
            await self._send_payload(session_id, OutputError(message=str(exc)))
            return

        self._partial_buffers.pop(session_id, None)
        final_data = {"content": response.answer or ""}
        if response.artifacts:
            final_data["artifacts"] = response.artifacts
        await self._send_payload(session_id, OutputFinal(data=final_data))

    async def on_disconnect(self, session_id: str) -> None:
        self._partial_buffers.pop(session_id, None)
        self._logger.info("WebSocket disconnected: %s", session_id)

    def _send_safe(self, session_id: str, payload: Any) -> None:
        try:
            asyncio.create_task(self._send_payload(session_id, payload))
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.warning("Failed to send telemetry update: %s", exc)

    def _handle_stream_chunk(self, session_id: str, event: Any) -> bool:
        event_type, payload = self._extract_event(event)
        if not event_type or event_type.lower() != "llm_stream_chunk":
            return False
        if not isinstance(payload, dict):
            payload = {}
        channel = payload.get("channel")
        if channel != "answer":
            return False
        text = payload.get("text")
        if not text:
            return True
        buffer = self._partial_buffers.get(session_id, "")
        buffer += str(text)
        self._partial_buffers[session_id] = buffer
        self._send_safe(session_id, OutputPartial(data={"content": buffer}))
        return True

    def _extract_event(self, event: Any) -> tuple[str | None, dict[str, Any] | None]:
        if isinstance(event, PlannerEventPayload):
            return event.type, event.payload
        if isinstance(event, dict):
            event_type = event.get("type") or event.get("event_type")
            payload = event.get("payload") or event.get("extra")
            return event_type, payload if isinstance(payload, dict) else None
        event_type = getattr(event, "event_type", None)
        payload = None
        if hasattr(event, "extra"):
            payload = getattr(event, "extra", None)
        elif hasattr(event, "to_payload"):
            payload = event.to_payload()
        return event_type, payload if isinstance(payload, dict) else None

    async def _send_payload(self, session_id: str, payload: Any) -> None:
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()
        await self._send(session_id, payload)
