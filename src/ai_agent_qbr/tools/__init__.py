"""Tool registry for ai-agent-qbr."""

from __future__ import annotations

from penguiflow.node import Node
from penguiflow.registry import ModelRegistry

from .agent_capabilities import agent_capabilities
from .analyze import analyze_results
from .comparison_intent import detect_comparison_intent
from .query_metrics import query_metrics
from .region_verifier import verify_region_filter
from .search import search_documents
from ..models import (
    AgentCapabilitiesArgs,
    AgentCapabilitiesResult,
    ComparisonIntentArgs,
    ComparisonIntentResult,
    FinalAnswer,
    MetricQueryArgs,
    Query,
    RegionFilterVerificationArgs,
    RegionFilterVerificationResult,
    SearchResults,
)
from qbr_intelligence.schemas.metric_qa import MetricAnswer

__all__ = [
    "agent_capabilities",
    "analyze_results",
    "detect_comparison_intent",
    "query_metrics",
    "search_documents",
    "verify_region_filter",
    "AgentCapabilitiesArgs",
    "AgentCapabilitiesResult",
    "ComparisonIntentArgs",
    "ComparisonIntentResult",
    "FinalAnswer",
    "MetricQueryArgs",
    "MetricAnswer",
    "RegionFilterVerificationArgs",
    "RegionFilterVerificationResult",
    "build_catalog_bundle",
]


def build_catalog_bundle() -> tuple[list[Node], ModelRegistry]:
    """Create the planner catalog and registry."""
    registry = ModelRegistry()
    registry.register("agent_capabilities", AgentCapabilitiesArgs, AgentCapabilitiesResult)
    registry.register("detect_comparison_intent", ComparisonIntentArgs, ComparisonIntentResult)
    registry.register(
        "verify_region_filter",
        RegionFilterVerificationArgs,
        RegionFilterVerificationResult,
    )
    registry.register("query_metrics", MetricQueryArgs, MetricAnswer)
    registry.register("search_documents", Query, SearchResults)
    registry.register("analyze_results", SearchResults, FinalAnswer)

    nodes = [
        Node(agent_capabilities, name="agent_capabilities"),
        Node(detect_comparison_intent, name="detect_comparison_intent"),
        Node(verify_region_filter, name="verify_region_filter"),
        Node(query_metrics, name="query_metrics"),
        Node(search_documents, name="search_documents"),
        Node(analyze_results, name="analyze_results"),
    ]
    return nodes, registry
