"""Tool registry for ai-agent-qbr."""

from __future__ import annotations

from penguiflow.node import Node
from penguiflow.registry import ModelRegistry

from .agent_capabilities import agent_capabilities
from .analyze import analyze_results
from .search import search_documents
from ..models import (
    AgentCapabilitiesArgs,
    AgentCapabilitiesResult,
    FinalAnswer,
    Query,
    SearchResults,
)

__all__ = [
    "agent_capabilities",
    "analyze_results",
    "search_documents",
    "AgentCapabilitiesArgs",
    "AgentCapabilitiesResult",
    "FinalAnswer",
    "build_catalog_bundle",
]


def build_catalog_bundle() -> tuple[list[Node], ModelRegistry]:
    """Create the planner catalog and registry."""
    registry = ModelRegistry()
    registry.register("agent_capabilities", AgentCapabilitiesArgs, AgentCapabilitiesResult)
    registry.register("search_documents", Query, SearchResults)
    registry.register("analyze_results", SearchResults, FinalAnswer)

    nodes = [
        Node(agent_capabilities, name="agent_capabilities"),
        Node(search_documents, name="search_documents"),
        Node(analyze_results, name="analyze_results"),
    ]
    return nodes, registry
