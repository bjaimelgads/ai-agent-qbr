"""Structured metric query tool."""

from __future__ import annotations

from penguiflow.catalog import tool
from penguiflow.planner import ToolContext

from ai_agent_qbr.models import MetricQueryArgs
from qbr_intelligence.metric_qa import MetricQueryEngine
from qbr_intelligence.schemas.metric_qa import MetricAnswer

import logging

_LOGGER = logging.getLogger(__name__)

@tool(desc="Query structured metric facts with deterministic filters", side_effects="read", tags=["planner"])
async def query_metrics(args: MetricQueryArgs, ctx: ToolContext) -> MetricAnswer:
    status_publisher = ctx.tool_context.get("status_publisher")
    if callable(status_publisher):
        status_publisher("Querying structured metric facts.", "Metric QA")

    engine = ctx.tool_context.get("metric_query_engine")
    if not isinstance(engine, MetricQueryEngine):
        _LOGGER.warning(
            "Metric query engine missing in tool_context (keys=%s)",
            sorted(ctx.tool_context.keys()) if isinstance(ctx.tool_context, dict) else "unknown",
        )
        return MetricAnswer(
            summary_text="Metric query engine is not available.",
            citations=[],
            confidence=0.0,
            assumptions=["Metric engine missing."],
            followups=["Please try again later."],
        )

    _LOGGER.info("Metric query: %s", args.question)
    result = await engine.query(args.question, debug=args.debug)
    interaction_metadata = ctx.tool_context.get("interaction_metadata")
    if isinstance(interaction_metadata, dict):
        interaction_metadata["metric_intent"] = result.intent.model_dump()
    return result.answer
