"""Refine metric intent with candidate validation."""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Iterable

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
from qbr_intelligence.metrics.models import MetricCatalogEntry
from qbr_intelligence.metric_qa import MetricQueryEngine
from qbr_intelligence.metric_qa.resolvers import ClientResolver, MetricResolver, PeriodResolver, RegionResolver

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_REGION_CANONICAL = {"us": "US", "usa": "US", "u s": "US", "emea": "EMEA", "global": "GLOBAL"}


def _normalize_text(value: str) -> str:
    return " ".join(_TOKEN_RE.findall((value or "").lower()))


def _best_fuzzy_match(
    *,
    probes: Iterable[str],
    choices: dict[str, str],
    min_ratio: float = 0.86,
    min_gap: float = 0.04,
) -> str | None:
    ranked: list[tuple[float, str]] = []
    for probe in probes:
        probe_norm = _normalize_text(probe)
        if not probe_norm:
            continue
        for key, value in choices.items():
            ratio = SequenceMatcher(None, probe_norm, key).ratio()
            ranked.append((ratio, value))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    top_ratio, top_value = ranked[0]
    second_ratio = ranked[1][0] if len(ranked) > 1 else 0.0
    if top_ratio >= min_ratio and (top_ratio - second_ratio) >= min_gap:
        return top_value
    return None


def _fuzzy_metric_from_candidates(candidates: list[str], catalog: list[MetricCatalogEntry]) -> str | None:
    alias_to_metric: dict[str, str] = {}
    for entry in catalog:
        alias_to_metric[_normalize_text(entry.name)] = entry.metric_id
        alias_to_metric[_normalize_text(entry.metric_id.replace("_", " "))] = entry.metric_id
        for alias in entry.aliases:
            if alias.alias:
                alias_to_metric[_normalize_text(alias.alias)] = entry.metric_id
    return _best_fuzzy_match(probes=candidates, choices=alias_to_metric, min_ratio=0.82)


def _fuzzy_client_from_candidates(candidates: list[str], clients: list[str]) -> str | None:
    choices = {_normalize_text(client): client for client in clients if client}
    return _best_fuzzy_match(probes=candidates, choices=choices, min_ratio=0.74)


def _fuzzy_region_from_candidates(candidates: list[str]) -> str | None:
    return _best_fuzzy_match(probes=candidates, choices=_REGION_CANONICAL, min_ratio=0.78)


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
        if not metric_hits:
            fuzzy_metric = _fuzzy_metric_from_candidates(args.candidates.metric_ids, list(engine._catalog or []))
            if fuzzy_metric:
                metric_hits = [fuzzy_metric]
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
        if not client_hits:
            fuzzy_client = _fuzzy_client_from_candidates(args.candidates.clients, list(engine._clients or []))
            if fuzzy_client:
                client_hits = [fuzzy_client]
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
            if not region_hits:
                fuzzy_region = _fuzzy_region_from_candidates(args.candidates.regions)
                if fuzzy_region:
                    region_hits.append(fuzzy_region)
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
