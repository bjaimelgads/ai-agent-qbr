"""WebSocket chat service following the ai-agent-reporting pattern."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

import logging

from ..api_models import (
    ChatRequest,
    OutputError,
    OutputFinal,
    OutputPing,
    OutputPong,
    OutputReady,
    OutputThinking,
    OutputUserMessage,
)
from ..config import Config
from ..dto import parse_planner_event
from ..orchestrator import AiAgentQbrOrchestrator
from ..telemetry import AgentTelemetry
from .connection_manager import ConnectionManager


class WebsocketChatService:
    def __init__(
        self,
        orchestrator_factory: Callable[[Config, AgentTelemetry], AiAgentQbrOrchestrator],
        connection_manager: ConnectionManager,
        config: Config,
        logger: logging.Logger | None = None,
    ) -> None:
        self._orchestrator_factory = orchestrator_factory
        self._manager = connection_manager
        self._config = config
        self._logger = logger or logging.getLogger(__name__)
        self._orchestrators: dict[str, AiAgentQbrOrchestrator] = {}

    async def handle_session(self, websocket: WebSocket, session_id: str) -> None:
        await self._manager.connect(session_id, websocket)
        await self._send(session_id, OutputReady(session_id=session_id))

        keepalive_task = asyncio.create_task(self._keepalive_loop(session_id))
        try:
            await self._receive_loop(session_id)
        except WebSocketDisconnect:
            self._manager.disconnect(session_id)
        finally:
            keepalive_task.cancel()
            await self._cleanup_session(session_id)

    async def _receive_loop(self, session_id: str) -> None:
        websocket = self._manager.active_connections[session_id]
        recv_task: asyncio.Task[Any] | None = None

        while True:
            try:
                if recv_task is None:
                    recv_task = asyncio.create_task(websocket.receive_json())
                done, _ = await asyncio.wait(
                    {recv_task},
                    timeout=self._config.receive_timeout_seconds,
                )
                if not done:
                    await self._send(session_id, OutputPing())
                    continue
                message = recv_task.result()
                recv_task = None
            except asyncio.TimeoutError:
                await self._send(session_id, OutputPing())
                continue

            if isinstance(message, dict) and message.get("type") == "pong":
                continue
            if isinstance(message, dict) and message.get("type") == "ping":
                await self._send(session_id, OutputPong())
                continue

            await self._handle_message(session_id, message)

    async def _handle_message(self, session_id: str, message: Any) -> None:
        try:
            request = ChatRequest.model_validate(message)
        except Exception as exc:
            await self._send(session_id, OutputError(message=str(exc)))
            return

        await self._send(session_id, OutputThinking(message="Planning response"))
        await self._send(
            session_id,
            OutputUserMessage(data={"content": request.message, "session_id": session_id}),
        )

        orchestrator = await self._get_orchestrator(session_id)

        try:
            metadata = request.metadata.model_dump() if request.metadata else {}
            response = await orchestrator.execute(
                request.message,
                tenant_id=metadata.get("tenant_id", "public"),
                user_id=metadata.get("user_id", "user"),
                session_id=session_id,
            )
        except Exception as exc:
            await self._send(session_id, OutputError(message=str(exc)))
            return

        await self._send(session_id, OutputFinal(data={"content": response.answer or ""}))

    async def _get_orchestrator(self, session_id: str) -> AiAgentQbrOrchestrator:
        orchestrator = self._orchestrators.get(session_id)
        if orchestrator is not None:
            return orchestrator

        event_callback = self._event_callback(session_id)
        telemetry = AgentTelemetry(
            flow_name="ai-agent-qbr",
            logger=self._logger,
            status_callback=None,
            event_callback=event_callback,
        )
        orchestrator = self._orchestrator_factory(self._config, telemetry)
        self._orchestrators[session_id] = orchestrator
        return orchestrator

    async def _cleanup_session(self, session_id: str) -> None:
        orchestrator = self._orchestrators.pop(session_id, None)
        if orchestrator is None:
            return
        if hasattr(orchestrator, "stop"):
            try:
                await orchestrator.stop()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                self._logger.warning("Failed to stop orchestrator: %s", exc)

    def _event_callback(self, session_id: str):
        def handler(event: Any) -> None:
            update = parse_planner_event(event)
            if update is None:
                return
            asyncio.create_task(self._send(session_id, update))

        return handler

    async def _keepalive_loop(self, session_id: str) -> None:
        while True:
            await asyncio.sleep(self._config.keepalive_interval_seconds)
            await self._send(session_id, OutputPing())

    async def _send(self, session_id: str, payload: Any) -> None:
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()
        await self._manager.send_json(session_id, payload)
