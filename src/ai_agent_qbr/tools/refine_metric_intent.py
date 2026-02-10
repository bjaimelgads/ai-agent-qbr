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
from ai_agent_qbr.tools.question_normalization import normalize_question_arg
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


def _query_token_match(query: str, candidates: list[str]) -> list[str]:
    lowered = (query or "").lower()
    hits: list[str] = []
    for candidate in candidates:
        token = candidate.strip().lower()
        if token and token in lowered:
            hits.append(candidate)
    return list(dict.fromkeys(hits))


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
    logger = logging.getLogger("uvicorn.error")
    canonical_query = normalize_question_arg(args.question, ctx.tool_context)
    logger.info(
        "REFINE_INTENT_REQUEST raw=%r canonical=%r current_intent=%s candidates=%s",
        args.question,
        canonical_query,
        args.intent.model_dump(),
        args.candidates.model_dump(),
    )

    engine = ctx.tool_context.get("metric_query_engine")
    if not isinstance(engine, MetricQueryEngine):
        return RefineMetricIntentResult(
            intent=args.intent,
            assumptions=["Metric query engine missing."],
            unresolved_fields=["metric", "client", "region", "period"],
        )

    await engine._load_catalogs()
    anchor_date = await engine._select_anchor_date(canonical_query)

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
        region_from_query = region_resolver.resolve(canonical_query)
        if region_from_query.value:
            updated.region = [region_from_query.value]
            assumptions.append("Region inferred from query")
        else:
            region_hits: list[str] = []
            candidate_hits = _query_token_match(canonical_query, args.candidates.regions)
            for candidate in candidate_hits:
                resolution = region_resolver.resolve(candidate)
                if resolution.value:
                    region_hits.append(resolution.value)
            region_hits = list(dict.fromkeys(region_hits))
            if len(region_hits) == 1:
                updated.region = region_hits
                assumptions.append("Region inferred from query-matched candidate")
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
    interaction_metadata = ctx.tool_context.get("interaction_metadata")
    if isinstance(interaction_metadata, dict):
        interaction_metadata["refined_metric_intent"] = result.intent.model_dump(mode="json")
    logger.info("REFINE_INTENT_RESULT %s", result.model_dump())
    return result
