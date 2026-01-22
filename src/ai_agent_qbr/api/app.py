"""FastAPI app for websocket transport."""

from __future__ import annotations

import logging
import shutil
from urllib.parse import urlparse

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

logger = logging.getLogger("uvicorn.error")


def _maybe_seed_sqlite_db(database_url: str) -> None:
    if not database_url.startswith("sqlite"):
        return
    parsed = urlparse(database_url.replace("sqlite+aiosqlite", "sqlite"))
    if not parsed.path:
        return
    db_path = Path(parsed.path)
    if db_path.exists():
        logger.info("SQLite database already exists at %s", db_path)
        return
    source_root = Path(__file__).resolve().parents[3]
    seed_candidates = [
        source_root / "qbr_extraction" / "qbr_intelligence.db",
        source_root / "qbr_intelligence.db",
        source_root / "data" / "qbr_intelligence.db",
    ]
    seed_path = next((path for path in seed_candidates if path.exists()), None)
    if seed_path is None:
        logger.warning("Seed database not found. Checked: %s", seed_candidates)
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(seed_path, db_path)
    logger.info("Seeded SQLite database at %s", db_path)


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
        _maybe_seed_sqlite_db(config.database_url)
        logger.info(
            "MLflow tracing: enabled=%s trace=%s uri=%s experiment=%s",
            config.mlflow_enabled,
            config.mlflow_tracing_enabled,
            config.mlflow_tracking_uri or "(default)",
            config.mlflow_experiment or "(default)",
        )
        logger.info(
            "FAISS startup config: vector_backend=%s auto_build=%s rebuild=%s dir=%s",
            config.vector_backend,
            config.faiss_auto_build,
            config.faiss_rebuild_on_startup,
            config.faiss_dir,
        )
        logger.info("Database URL: %s", config.database_url)
        if config.vector_backend != "faiss":
            logger.info("FAISS auto-build skipped: VECTOR_BACKEND=%s", config.vector_backend)
            return
        if not config.faiss_auto_build:
            logger.info("FAISS auto-build disabled (FAISS_AUTO_BUILD=false).")
            return
        logger.info(
            "FAISS auto-build enabled (dir=%s, db=%s, rebuild=%s)",
            config.faiss_dir,
            config.database_url,
            config.faiss_rebuild_on_startup,
        )
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
