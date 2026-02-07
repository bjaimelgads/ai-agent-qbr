"""Refine metric intent with candidate validation."""

from __future__ import annotations

import logging

from penguiflow.catalog import tool
from penguiflow.planner import ToolContext

from ai_agent_qbr.models import (
    PeriodSpec,
    RefineMetricIntentArgs,
    RefineMetricIntentResult,
    ResolvedMetricIntent,
)
from ai_agent_qbr.tools.status import ToolStatusEmitter
from qbr_intelligence.metric_qa import MetricQueryEngine
from qbr_intelligence.metric_qa.resolvers import ClientResolver, MetricResolver, PeriodResolver, RegionResolver


def _like_match(values: list[str], candidates: list[str]) -> list[str]:
    if not values or not candidates:
        return []
    lowered_values = [value.lower() for value in values]
    matches: list[str] = []
    for candidate in candidates:
        token = candidate.strip().lower()
        if not token:
            continue
        for idx, value in enumerate(lowered_values):
            if token in value:
                matches.append(values[idx])
    return list(dict.fromkeys(matches))


@tool(
    desc=(
        "Validate planner-proposed intent candidates deterministically. "
        "Use this after resolve_metric_intent when any fields are missing or ambiguous. "
        "Pass candidate values; this tool confirms matches against catalogs/aliases and period parsing."
    ),
    tags=["planner"],
)
async def refine_metric_intent(
    args: RefineMetricIntentArgs,
    ctx: ToolContext,
) -> RefineMetricIntentResult:
    status = ToolStatusEmitter(ctx, tool_name="refine_metric_intent")
    await status.step("Confirming details.", step_name="Confirming details")

    engine = ctx.tool_context.get("metric_query_engine")
    if not isinstance(engine, MetricQueryEngine):
        return RefineMetricIntentResult(
            intent=args.intent,
            assumptions=["Metric query engine missing."],
            unresolved_fields=["metric", "client", "region", "period"],
        )

    await engine._load_catalogs()
    anchor_date = await engine._select_anchor_date(args.question)

    updated = ResolvedMetricIntent.model_validate(args.intent.model_dump())
    assumptions: list[str] = []
    unresolved: list[str] = []

    metric_resolver = MetricResolver(engine._catalog or [])
    client_resolver = ClientResolver(engine._clients or [])
    region_resolver = RegionResolver()
    period_resolver = PeriodResolver()

    if not updated.metric_ids:
        metric_hits: list[str] = []
        for candidate in args.candidates.metric_ids:
            resolved, _ = metric_resolver.resolve(candidate)
            if resolved:
                metric_hits.append(resolved[0])
        metric_hits = list(dict.fromkeys(metric_hits))
        if len(metric_hits) == 1:
            updated.metric_ids = [metric_hits[0]]
            assumptions.append("Metric inferred from candidate list")
        else:
            unresolved.append("metric")

    if not updated.client:
        client_hits: list[str] = []
        for candidate in args.candidates.clients:
            resolution = client_resolver.resolve(candidate)
            if resolution.value:
                client_hits.append(resolution.value)
        if not client_hits:
            client_hits = _like_match(engine._clients or [], args.candidates.clients)
        client_hits = list(dict.fromkeys(client_hits))
        if client_hits:
            updated.client = client_hits
            assumptions.append("Client inferred from candidate list")
        else:
            unresolved.append("client")

    if not updated.region:
        region_hits: list[str] = []
        for candidate in args.candidates.regions:
            resolution = region_resolver.resolve(candidate)
            if resolution.value:
                region_hits.append(resolution.value)
        region_hits = list(dict.fromkeys(region_hits))
        if region_hits:
            updated.region = region_hits
            assumptions.append("Region inferred from candidate list")
        else:
            unresolved.append("region")

    if not updated.period:
        period_hits: list[PeriodSpec] = []
        for candidate in args.candidates.periods:
            bounds, label, period_type = period_resolver.resolve(candidate, anchor_date=anchor_date)
            if bounds:
                period_hits.append(
                    PeriodSpec(
                        type=period_type,
                        value=label,
                        start=bounds.start,
                        end=bounds.end,
                    )
                )
        if period_hits:
            updated.period = period_hits
            assumptions.append("Period inferred from candidate list")
        else:
            unresolved.append("period")

    result = RefineMetricIntentResult(
        intent=updated,
        assumptions=assumptions,
        unresolved_fields=unresolved,
    )
    logging.getLogger(__name__).info("REFINE_INTENT_RESULT %s", result.model_dump())
    return result
