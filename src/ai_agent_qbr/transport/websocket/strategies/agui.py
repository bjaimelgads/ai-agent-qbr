"""WebSocket strategy that emits AG-UI protocol events."""

from __future__ import annotations

import asyncio
import logging
import secrets
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from ag_ui.core import RunAgentInput
from pydantic import ValidationError

from ai_agent_qbr.application.session_registry import SessionRegistry
from ai_agent_qbr.application.session_service import SendMessageUseCase
from ai_agent_qbr.domain.models import SessionContext
from ai_agent_qbr.transport.agui.adapter import AGUIWebsocketAdapter, build_run_input, pick_query

from .base import SendJson, WebsocketOutputStrategy


class AguiWebsocketOutputStrategy(WebsocketOutputStrategy):
    """Emit AG-UI events over the existing WebSocket connection."""

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
        self._reasoning_ids: dict[str, str | None] = {}

    async def on_connect(self, session_id: str) -> None:
        self._reasoning_ids[session_id] = None
        self._logger.info("AG-UI websocket connected: %s", session_id)

    async def on_message(self, session_id: str, message: Any) -> None:
        run_input = self._parse_input(session_id, message)
        if run_input is None:
            await self._send_error(session_id, "Invalid AG-UI input")
            return

        query = pick_query(run_input.messages)
        if not query:
            await self._send_error(session_id, "Missing user message")
            return

        telemetry = self._session_registry.get_telemetry(session_id)
        if telemetry is None:
            await self._send_error(session_id, "Missing session telemetry")
            return

        adapter = AGUIWebsocketAdapter()
        adapter.start_run(run_input)

        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()
        run_error: Exception | None = None
        run_result: Any | None = None

        def event_callback(event: Any) -> None:
            for mapped in adapter.convert_planner_event(event):
                queue.put_nowait(mapped)

        def status_callback(msg: str | None, step: int | None) -> None:
            payload = {"message": msg, "step": step}
            queue.put_nowait(adapter.custom("status", payload))

        async def run_agent() -> None:
            nonlocal run_error, run_result
            try:
                run_result = await self._send_message.execute(
                    session=SessionContext(
                        tenant_id=self._extract_tenant_id(message),
                        user_id=self._extract_user_id(message),
                        session_id=session_id,
                    ),
                    message=query,
                    telemetry_factory=lambda: telemetry,
                )
            except Exception as exc:
                self._logger.exception("AG-UI run task failed")
                run_error = exc
            finally:
                queue.put_nowait(sentinel)

        asyncio.create_task(run_agent())

        async def stream_events():
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                yield item

            if run_error is not None:
                raise run_error

            if run_result is not None and getattr(run_result, "answer", None) and not adapter.streamed_answer:
                for event in adapter.emit_text_block(run_result.answer):
                    yield event
            if run_result is not None and getattr(run_result, "artifacts", None):
                self._logger.info(
                    "AG-UI artifact emit session_id=%s artifact_type=%s keys=%s",
                    session_id,
                    run_result.artifacts.get("type") if isinstance(run_result.artifacts, dict) else None,
                    sorted(run_result.artifacts.keys()) if isinstance(run_result.artifacts, dict) else None,
                )
                if isinstance(run_result.artifacts, dict) and run_result.artifacts.get("type") == "datagrid":
                    yield adapter.custom(
                        "artifact_chunk",
                        {
                            "stream_id": "artifact",
                            "seq": 0,
                            "done": True,
                            "artifact_type": "ui_component",
                            "chunk": {
                                "component": "datagrid",
                                "title": run_result.artifacts.get("title"),
                                "props": run_result.artifacts,
                            },
                            "meta": {"source": "metric_grid"},
                        },
                    )
                yield adapter.custom("artifact", {"artifact": run_result.artifacts})
            elif run_result is not None:
                self._logger.info(
                    "AG-UI artifact missing session_id=%s",
                    session_id,
                )

        with telemetry.subscribe(status_callback=status_callback, event_callback=event_callback):
            try:
                async for event in adapter.with_run_lifecycle(run_input, stream_events()):
                    await self._send_event(session_id, event)
            except Exception as exc:
                self._logger.exception("AG-UI run failed")
            finally:
                adapter.end_run()

    async def on_disconnect(self, session_id: str) -> None:
        self._reasoning_ids.pop(session_id, None)
        self._logger.info("AG-UI websocket disconnected: %s", session_id)

    def _parse_input(self, session_id: str, message: Any) -> RunAgentInput | None:
        if isinstance(message, RunAgentInput):
            return message
        try:
            return RunAgentInput.model_validate(message)
        except ValidationError:
            pass
        if isinstance(message, dict):
            normalized = _normalize_message_payload(message)
            if normalized is not None:
                try:
                    return RunAgentInput.model_validate(normalized)
                except ValidationError:
                    pass
        if isinstance(message, dict):
            user_message = message.get("message")
            if not isinstance(user_message, str):
                user_message = message.get("text")
            if not isinstance(user_message, str):
                return None
            return build_run_input(
                session_id=session_id,
                message=user_message,
                run_id=secrets.token_hex(8),
            )
        return None

    async def _send_event(self, session_id: str, event: Any) -> None:
        if hasattr(event, "model_dump"):
            payload = event.model_dump(by_alias=True, exclude_none=True)
        else:
            payload = event
        for outbound in self._normalize_payloads(session_id, payload):
            if isinstance(outbound, dict) and "timestamp" not in outbound:
                outbound["timestamp"] = _now_iso_timestamp()
            await self._send(session_id, outbound)

    async def _send_error(self, session_id: str, message: str) -> None:
        payload = {"type": "RUN_ERROR", "message": message, "timestamp": _now_iso_timestamp()}
        await self._send(session_id, payload)

    def _extract_tenant_id(self, message: Any) -> str:
        if isinstance(message, dict):
            meta = message.get("metadata")
            if isinstance(meta, dict) and isinstance(meta.get("tenant_id"), str):
                return meta["tenant_id"]
        return "public"

    def _extract_user_id(self, message: Any) -> str:
        if isinstance(message, dict):
            meta = message.get("metadata")
            if isinstance(meta, dict) and isinstance(meta.get("user_id"), str):
                return meta["user_id"]
        return "user"

    def _normalize_payloads(self, session_id: str, payload: Any) -> list[Any]:
        if not isinstance(payload, dict):
            return [payload]

        event_type = payload.get("type")
        if event_type == "RUN_STARTED":
            self._reasoning_ids[session_id] = None
            return [payload]

        if event_type == "RUN_FINISHED":
            return self._flush_reasoning(session_id) + [payload]

        if event_type == "CUSTOM" and payload.get("name") == "thinking":
            value = payload.get("value") or {}
            if not isinstance(value, dict):
                value = {}
            text = value.get("text") if isinstance(value.get("text"), str) else ""
            done = bool(value.get("done"))
            return self._reasoning_payloads(session_id, text=text, done=done)

        return [payload]

    def _reasoning_payloads(self, session_id: str, *, text: str, done: bool) -> list[dict[str, Any]]:
        message_id = self._reasoning_ids.get(session_id)
        if message_id is None and not text and not done:
            return []
        if message_id is None:
            message_id = str(uuid4())
            self._reasoning_ids[session_id] = message_id
            payloads: list[dict[str, Any]] = [
                {"type": "CUSTOM", "name": "REASONING_START", "messageId": message_id}
            ]
        else:
            payloads = []

        if text:
            payloads.append(
                {
                    "type": "CUSTOM",
                    "name": "REASONING_MESSAGE_CONTENT",
                    "messageId": message_id,
                    "delta": text,
                }
            )

        if done:
            payloads.append({"type": "CUSTOM", "name": "REASONING_END", "messageId": message_id})
            self._reasoning_ids[session_id] = None

        return payloads

    def _flush_reasoning(self, session_id: str) -> list[dict[str, Any]]:
        message_id = self._reasoning_ids.get(session_id)
        if not message_id:
            return []
        self._reasoning_ids[session_id] = None
        return [{"type": "CUSTOM", "name": "REASONING_END", "messageId": message_id}]


def _now_iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _normalize_message_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return None
    normalized_messages: list[Any] = []
    changed = False
    for item in messages:
        if not isinstance(item, dict):
            normalized_messages.append(item)
            continue
        normalized = dict(item)
        role = normalized.get("role")
        if role == "agent":
            normalized["role"] = "assistant"
            changed = True
        if "content" not in normalized:
            text = normalized.get("text")
            if isinstance(text, str):
                normalized["content"] = text
                changed = True
        normalized_messages.append(normalized)
    if not changed:
        return payload
    updated = dict(payload)
    updated["messages"] = normalized_messages
    return updated
