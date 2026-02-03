"""WebSocket strategy that emits AG-UI protocol events."""

from __future__ import annotations

import asyncio
import logging
import secrets
from typing import Any

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

    async def on_connect(self, session_id: str) -> None:
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
        self._logger.info("AG-UI websocket disconnected: %s", session_id)

    def _parse_input(self, session_id: str, message: Any) -> RunAgentInput | None:
        if isinstance(message, RunAgentInput):
            return message
        try:
            return RunAgentInput.model_validate(message)
        except ValidationError:
            pass
        if isinstance(message, dict) and "message" in message:
            user_message = message.get("message")
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
        await self._send(session_id, payload)

    async def _send_error(self, session_id: str, message: str) -> None:
        payload = {"type": "RUN_ERROR", "message": message}
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
