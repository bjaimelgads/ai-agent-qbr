"""Deterministic metric intent resolution tool."""

from __future__ import annotations

import logging

from penguiflow.catalog import tool
from penguiflow.planner import ToolContext

from ai_agent_qbr.models import (
    FieldConfidence,
    PeriodSpec,
    ResolveMetricIntentArgs,
    ResolveMetricIntentResult,
    ResolvedMetricIntent,
)
from ai_agent_qbr.tools.status import ToolStatusEmitter
from qbr_intelligence.metric_qa import MetricQueryEngine
from qbr_intelligence.metric_qa.intent import DeterministicIntentExtractor


@tool(
    desc=(
        "Deterministically resolve metric intent (metric/client/region/period). "
        "Call this first for metric/KPI questions. If any fields are missing or ambiguous, "
        "follow with refine_metric_intent to validate candidate values before querying metrics."
    ),
    tags=["planner"],
)
async def resolve_metric_intent(
    args: ResolveMetricIntentArgs,
    ctx: ToolContext,
) -> ResolveMetricIntentResult:
    status = ToolStatusEmitter(ctx, tool_name="resolve_metric_intent")
    await status.step("Understanding your request.", step_name="Understanding request")

    engine = ctx.tool_context.get("metric_query_engine")
    if not isinstance(engine, MetricQueryEngine):
        return ResolveMetricIntentResult(
            intent=ResolvedMetricIntent(metric_ids=[]),
            confidence=FieldConfidence(),
        )

    await engine._load_catalogs()
    anchor_date = await engine._select_anchor_date(args.question)

    extractor = DeterministicIntentExtractor(engine._catalog or [], engine._clients or [])
    intent, debug = extractor.extract(args.question, anchor_date=anchor_date)

    period_spec = None
    if intent.period:
        period_spec = PeriodSpec(
            type=intent.period.type,
            value=intent.period.value,
            start=intent.period.start,
            end=intent.period.end,
        )

    confidence = FieldConfidence(
        metric=max(debug.metrics.values(), default=0.0),
        client=debug.client_confidence,
        region=debug.region_confidence,
        period=1.0 if intent.period else 0.0,
    )

    resolved = ResolvedMetricIntent(
        metric_ids=list(intent.metric_ids),
        client=intent.client,
        region=intent.region,
        period=period_spec,
        aggregation=intent.aggregation,
        grouping=intent.grouping,
        limit=intent.limit,
    )

    result = ResolveMetricIntentResult(intent=resolved, confidence=confidence)
    logging.getLogger(__name__).info("RESOLVE_INTENT_RESULT %s", result.model_dump())
    return result
