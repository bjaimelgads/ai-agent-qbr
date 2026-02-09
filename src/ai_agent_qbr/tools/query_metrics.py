"""Structured metric query tool."""

from __future__ import annotations

from penguiflow.catalog import tool
from penguiflow.planner import ToolContext

from ai_agent_qbr.models import MetricQueryArgs
from ai_agent_qbr.tools.question_normalization import normalize_question_arg
from ai_agent_qbr.tools.status import ToolStatusEmitter
from qbr_intelligence.metric_qa import MetricQueryEngine
from qbr_intelligence.schemas.metric_qa import MetricAnswer

import logging
import time

_LOGGER = logging.getLogger("uvicorn.error")

@tool(
    desc=(
        "Query structured metric facts with deterministic filters. "
        "Call this after resolve_metric_intent (and refine_metric_intent if needed)."
    ),
    side_effects="read",
    tags=["planner"],
)
async def query_metrics(args: MetricQueryArgs, ctx: ToolContext) -> MetricAnswer:
    status = ToolStatusEmitter(ctx, tool_name="query_metrics")
    await status.step("Looking up metric results.", step_name="Finding metrics")
    canonical_query = normalize_question_arg(args.question, ctx.tool_context)
    _LOGGER.info("METRIC_QA_QUERY raw=%s canonical=%s", args.question, canonical_query)

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

    _LOGGER.info("Metric query start: %s", canonical_query)
    start = time.perf_counter()
    trace_id = None
    if isinstance(ctx.tool_context, dict):
        trace_id = ctx.tool_context.get("trace_id")
    result = await engine.query(canonical_query, debug=args.debug, trace_id=trace_id)
    _LOGGER.info("METRIC_QA_RAW_ANSWER %s", result.answer.model_dump(mode="json"))
    _LOGGER.info("Metric query done: %.2fs rows=%s", time.perf_counter() - start, len(result.answer.data or []))
    interaction_metadata = ctx.tool_context.get("interaction_metadata")
    if isinstance(interaction_metadata, dict):
        # Keep metadata JSON-serializable for planner memory/llm_context round-trips.
        interaction_metadata["metric_intent"] = result.intent.model_dump(mode="json")
        interaction_metadata["metric_answer"] = result.answer.model_dump(mode="json")
        interaction_metadata["metric_answer_row_count"] = len(result.answer.data or [])
    return result.answer
