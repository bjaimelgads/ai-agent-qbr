"""Websocket chat service for the PenguiFlow agent template."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from collections.abc import Callable

from ai_agent_qbr.application.session_service import CloseSessionUseCase, StartSessionUseCase
from ai_agent_qbr.config import Config
from ai_agent_qbr.domain.models import SessionContext
from ai_agent_qbr.telemetry import AgentTelemetry
from ai_agent_qbr.transport.websocket.connection_manager import ConnectionManager
from ai_agent_qbr.transport.websocket.schemas import OutputPing, OutputPong
from ai_agent_qbr.transport.websocket.strategies import WebsocketOutputStrategy


class WebsocketChatService:
    def __init__(
        self,
        *,
        connection_manager: ConnectionManager,
        start_session: StartSessionUseCase,
        close_session: CloseSessionUseCase,
        config: Config,
        output_strategy: WebsocketOutputStrategy,
        telemetry_factory: Callable[[], AgentTelemetry],
        logger: logging.Logger | None = None,
    ) -> None:
        self._manager = connection_manager
        self._start_session = start_session
        self._close_session = close_session
        self._config = config
        self._output_strategy = output_strategy
        self._telemetry_factory = telemetry_factory
        self._logger = logger or logging.getLogger(__name__)

    async def handle_session(self, websocket: WebSocket, session_id: str) -> None:
        await self._manager.connect(session_id, websocket)
        self._logger.info("WebSocket connected; starting session for %s", session_id)
        await self._start_session.execute(
            session=SessionContext(
                tenant_id="public",
                user_id="user",
                session_id=session_id,
            ),
            telemetry_factory=self._telemetry_factory,
        )
        await self._output_strategy.on_connect(session_id)

        keepalive_task = asyncio.create_task(self._keepalive_loop(session_id))
        try:
            await self._receive_loop(session_id)
        except WebSocketDisconnect:
            self._manager.disconnect(session_id)
        finally:
            keepalive_task.cancel()
            await self._output_strategy.on_disconnect(session_id)
            await self._close_session.execute(session_id)

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

    async def _handle_message(
        self,
        session_id: str,
        message: Any,
    ) -> None:
        await self._output_strategy.on_message(session_id, message)

    async def _keepalive_loop(self, session_id: str) -> None:
        while True:
            await asyncio.sleep(self._config.keepalive_interval_seconds)
            await self._send(session_id, OutputPing())

    async def _send(self, session_id: str, payload: Any) -> None:
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()
        await self._manager.send_json(session_id, payload)
