"""CLI playground for local testing."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from dotenv import load_dotenv

from ai_agent_qbr.application.session_registry import SessionRegistry
from ai_agent_qbr.application.session_service import SendMessageUseCase, StartSessionUseCase
from ai_agent_qbr.config import Config
from ai_agent_qbr.domain.models import SessionContext
from ai_agent_qbr.infrastructure.memory_store import InMemoryMemoryStore
from ai_agent_qbr.orchestrator import AiAgentQbrOrchestrator
from ai_agent_qbr.telemetry import AgentTelemetry

logger = logging.getLogger(__name__)


def main() -> None:
    load_dotenv()
    asyncio.run(run_playground())


async def run_playground() -> None:
    config = Config.from_env()

    memory_store = InMemoryMemoryStore(
        max_turns=config.short_term_memory_full_zone_turns,
        retrieval_turns=config.short_term_memory_full_zone_turns,
    )

    def orchestrator_factory(telemetry: AgentTelemetry) -> AiAgentQbrOrchestrator:
        return AiAgentQbrOrchestrator(
            config,
            memory_store=memory_store,
            telemetry=telemetry,
        )

    session_registry = SessionRegistry(orchestrator_factory=orchestrator_factory)
    send_message = SendMessageUseCase(session_registry)
    start_session = StartSessionUseCase(session_registry)

    session_id = input("Session id [local-session]: ").strip() or "local-session"
    tenant_id = input("Tenant id [public]: ").strip() or "public"
    user_id = input("User id [user]: ").strip() or "user"

    def telemetry_factory() -> AgentTelemetry:
        return AgentTelemetry(flow_name="ai-agent-qbr", logger=logger)

    await start_session.execute(
        session=SessionContext(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
        ),
        telemetry_factory=telemetry_factory,
    )

    print("Type 'exit' to quit.")

    while True:
        prompt = input("You: ").strip()
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break
        response = await send_message.execute(
            session=SessionContext(
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=session_id,
            ),
            message=prompt,
            telemetry_factory=telemetry_factory,
        )
        print(f"Agent: {response.answer}")
