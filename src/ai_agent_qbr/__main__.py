"""Entry point for ai-agent-qbr."""

from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv

# Load .env file before importing config
load_dotenv()

from .config import Config
from .orchestrator import AiAgentQbrOrchestrator

logging.basicConfig(level=logging.INFO)


async def _run_demo() -> None:
    config = Config.from_env()
    orchestrator = AiAgentQbrOrchestrator(config)
    response = await orchestrator.execute(
        query="Summarise PenguiFlow features",
        tenant_id="demo-tenant",
        user_id="demo-user",
        session_id="demo-session",
    )
    print(f"Agent response: {response.answer}")
    await orchestrator.stop()


def main() -> None:
    asyncio.run(_run_demo())


if __name__ == "__main__":
    main()