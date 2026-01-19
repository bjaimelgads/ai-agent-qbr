"""FastAPI app for websocket transport."""

from __future__ import annotations

import logging

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from typing import Callable
from pathlib import Path

from ai_agent_qbr.application.session_registry import SessionRegistry
from ai_agent_qbr.application.session_service import CloseSessionUseCase, SendMessageUseCase, StartSessionUseCase
from ai_agent_qbr.config import Config
from ai_agent_qbr.infrastructure.memory_store import InMemoryMemoryStore
from ai_agent_qbr.orchestrator import AiAgentQbrOrchestrator
from ai_agent_qbr.telemetry import AgentTelemetry
from ai_agent_qbr.transport.websocket.connection_manager import ConnectionManager
from ai_agent_qbr.transport.websocket.service import WebsocketChatService
from ai_agent_qbr.transport.websocket.strategies import build_websocket_strategy
from qbr_agent.infrastructure.faiss_builder import FaissBuildConfig, build_faiss_index

logger = logging.getLogger(__name__)


def create_app(
    *,
    config: Config | None = None,
    orchestrator_factory: Callable[[AgentTelemetry], AiAgentQbrOrchestrator] | None = None,
    memory_store: object | None = None,
) -> FastAPI:
    load_dotenv()
    config = config or Config.from_env()
    config.validate()

    if memory_store is None:
        memory_store = InMemoryMemoryStore(
            max_turns=config.short_term_memory_full_zone_turns,
            retrieval_turns=config.short_term_memory_full_zone_turns,
        )

    if orchestrator_factory is None:
        def orchestrator_factory(telemetry: AgentTelemetry) -> AiAgentQbrOrchestrator:
            return AiAgentQbrOrchestrator(
                config,
                memory_store=memory_store,
                telemetry=telemetry,
            )

    session_registry = SessionRegistry(orchestrator_factory=orchestrator_factory)
    manager = ConnectionManager()

    close_session = CloseSessionUseCase(session_registry)
    send_message = SendMessageUseCase(session_registry)
    start_session = StartSessionUseCase(session_registry)

    def telemetry_factory() -> AgentTelemetry:
        return AgentTelemetry(flow_name="ai-agent-qbr", logger=logger)

    output_strategy = build_websocket_strategy(
        config=config,
        manager=manager,
        send_message_use_case=send_message,
        session_registry=session_registry,
        logger=logger,
    )

    chat_service = WebsocketChatService(
        connection_manager=manager,
        start_session=start_session,
        close_session=close_session,
        config=config,
        output_strategy=output_strategy,
        telemetry_factory=telemetry_factory,
    )

    app = FastAPI(title="ai-agent-qbr")
    app.state.config = config
    app.state.session_registry = session_registry
    app.state.chat_service = chat_service

    @app.on_event("startup")
    async def _startup() -> None:
        if config.vector_backend != "faiss":
            return
        if not config.faiss_auto_build:
            return
        build_config = FaissBuildConfig(
            database_url=config.database_url,
            base_dir=Path(config.faiss_dir),
            normalize=config.faiss_normalize,
            index_type=config.faiss_index_type,
            metric=config.faiss_metric,
            embedding_model_filter=None,
            force=config.faiss_rebuild_on_startup,
        )
        try:
            built = await build_faiss_index(build_config)
        except Exception as exc:
            logger.warning("FAISS index build failed: %s", exc)
            return
        if built:
            logger.info("FAISS index built at %s", build_config.base_dir)

    @app.websocket("/ws/chat/{session_id}")
    async def chat_ws(websocket: WebSocket, session_id: str) -> None:
        await chat_service.handle_session(websocket, session_id)

    return app


app = create_app()
