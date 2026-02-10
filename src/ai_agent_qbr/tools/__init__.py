"""Tool registry for ai-agent-qbr."""

from __future__ import annotations

from penguiflow.node import Node
from penguiflow.registry import ModelRegistry

from .agent_capabilities import agent_capabilities
from .analyze import analyze_results
from .comparison_intent import detect_comparison_intent
from .refine_metric_intent import refine_metric_intent
from .query_metrics import query_metrics
from .region_verifier import verify_region_filter
from .resolve_metric_intent import resolve_metric_intent
from .search import search_documents
from ..models import (
    AgentCapabilitiesArgs,
    AgentCapabilitiesResult,
    CandidateSet,
    ComparisonIntentArgs,
    ComparisonIntentResult,
    FinalAnswer,
    FieldConfidence,
    MetricQueryArgs,
    Query,
    RefineMetricIntentArgs,
    RefineMetricIntentResult,
    RegionFilterVerificationArgs,
    RegionFilterVerificationResult,
    ResolveMetricIntentArgs,
    ResolveMetricIntentResult,
    ResolvedMetricIntent,
    SearchResults,
)
from qbr_intelligence.schemas.metric_qa import MetricAnswer

__all__ = [
    "agent_capabilities",
    "analyze_results",
    "detect_comparison_intent",
    "refine_metric_intent",
    "query_metrics",
    "resolve_metric_intent",
    "search_documents",
    "verify_region_filter",
    "AgentCapabilitiesArgs",
    "AgentCapabilitiesResult",
    "CandidateSet",
    "ComparisonIntentArgs",
    "ComparisonIntentResult",
    "FinalAnswer",
    "FieldConfidence",
    "MetricQueryArgs",
    "MetricAnswer",
    "RefineMetricIntentArgs",
    "RefineMetricIntentResult",
    "RegionFilterVerificationArgs",
    "RegionFilterVerificationResult",
    "ResolveMetricIntentArgs",
    "ResolveMetricIntentResult",
    "ResolvedMetricIntent",
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
    registry.register("resolve_metric_intent", ResolveMetricIntentArgs, ResolveMetricIntentResult)
    registry.register("refine_metric_intent", RefineMetricIntentArgs, RefineMetricIntentResult)
    registry.register("search_documents", Query, SearchResults)
    registry.register("search_qbr", Query, SearchResults)
    registry.register("analyze_results", SearchResults, FinalAnswer)

    nodes = [
        Node(agent_capabilities, name="agent_capabilities"),
        Node(detect_comparison_intent, name="detect_comparison_intent"),
        Node(verify_region_filter, name="verify_region_filter"),
        Node(query_metrics, name="query_metrics"),
        Node(resolve_metric_intent, name="resolve_metric_intent"),
        Node(refine_metric_intent, name="refine_metric_intent"),
        Node(search_documents, name="search_documents"),
        Node(search_documents, name="search_qbr"),
        Node(analyze_results, name="analyze_results"),
    ]
    return nodes, registry
