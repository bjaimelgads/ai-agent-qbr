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
from ai_agent_qbr.tools.question_normalization import normalize_question_arg
from ai_agent_qbr.tools.status import ToolStatusEmitter
from qbr_intelligence.metric_qa import MetricQueryEngine
from qbr_intelligence.metric_qa.intent import DeterministicIntentExtractor
from qbr_intelligence.metric_qa.resolvers import RegionResolver


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
    logger = logging.getLogger("uvicorn.error")
    canonical_query = normalize_question_arg(args.question, ctx.tool_context)
    logger.info(
        "RESOLVE_INTENT_REQUEST raw=%r canonical=%r",
        args.question,
        canonical_query,
    )

    engine = ctx.tool_context.get("metric_query_engine")
    if not isinstance(engine, MetricQueryEngine):
        return ResolveMetricIntentResult(
            intent=ResolvedMetricIntent(metric_ids=[]),
            confidence=FieldConfidence(),
        )

    await engine._load_catalogs()
    anchor_date = await engine._select_anchor_date(canonical_query)
    logger.info("RESOLVE_INTENT_ANCHOR_DATE %s", anchor_date)

    extractor = DeterministicIntentExtractor(engine._catalog or [], engine._clients or [])
    intent, debug = extractor.extract(canonical_query, anchor_date=anchor_date)
    logger.info(
        "RESOLVE_INTENT_RAW intent=%s debug=%s",
        getattr(intent, "model_dump", lambda: intent)(),
        getattr(debug, "model_dump", lambda: debug)(),
    )

    period_specs: list[PeriodSpec] = []
    raw_periods = intent.period or []
    if not isinstance(raw_periods, list):
        raw_periods = [raw_periods]
    for period in raw_periods:
        if hasattr(period, "start") and hasattr(period, "end"):
            period_specs.append(
                PeriodSpec(
                    type=getattr(period, "type", None),
                    value=getattr(period, "value", None),
                    start=getattr(period, "start", None),
                    end=getattr(period, "end", None),
                )
            )
            continue
        if isinstance(period, dict):
            period_specs.append(
                PeriodSpec(
                    type=period.get("type"),
                    value=period.get("value"),
                    start=period.get("start"),
                    end=period.get("end"),
                )
            )
            continue
        if isinstance(period, (list, tuple)) and len(period) >= 2:
            period_specs.append(
                PeriodSpec(
                    type=period[3] if len(period) > 3 else None,
                    value=period[2] if len(period) > 2 else None,
                    start=period[0],
                    end=period[1],
                )
            )

    confidence = FieldConfidence(
        metric=max(debug.metrics.values(), default=0.0),
        client=debug.client_confidence,
        region=debug.region_confidence,
        period=1.0 if intent.period else 0.0,
    )

    client_values = intent.client if isinstance(intent.client, list) else ([intent.client] if intent.client else [])
    region_values = intent.region if isinstance(intent.region, list) else ([intent.region] if intent.region else [])
    if not region_values:
        fallback = RegionResolver().resolve(canonical_query)
        if fallback.value:
            region_values = [fallback.value]
        elif fallback.candidates:
            region_values = list(fallback.candidates)

    resolved = ResolvedMetricIntent(
        metric_ids=list(intent.metric_ids),
        client=client_values,
        region=region_values,
        period=period_specs,
        aggregation=intent.aggregation,
        grouping=intent.grouping,
        limit=intent.limit,
    )

    result = ResolveMetricIntentResult(intent=resolved, confidence=confidence)
    interaction_metadata = ctx.tool_context.get("interaction_metadata")
    if isinstance(interaction_metadata, dict):
        interaction_metadata["resolved_metric_intent"] = result.intent.model_dump(mode="json")
    logging.getLogger(__name__).info("RESOLVE_INTENT_RESULT %s", result.model_dump())
    return result
