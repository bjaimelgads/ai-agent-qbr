"""ai-agent-qbr ReactPlanner agent."""

from .config import Config
from .orchestrator import AgentResponse, AiAgentQbrOrchestrator

# Aliases for PenguiFlow playground discovery (align with ai-agent-base).
Orchestrator = AiAgentQbrOrchestrator
orchestrator = AiAgentQbrOrchestrator

__all__ = [
    "AgentResponse",
    "AiAgentQbrOrchestrator",
    "Config",
    "Orchestrator",
    "orchestrator",
]
