"""Deterministic metric intent resolution tool."""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Iterable

from penguiflow.catalog import tool
from penguiflow.planner import ToolContext

from ai_agent_qbr.models import (
    CandidateSet,
    FieldConfidence,
    PeriodSpec,
    ResolveMetricIntentArgs,
    ResolveMetricIntentResult,
    ResolvedMetricIntent,
)
from ai_agent_qbr.tools.question_normalization import normalize_question_arg
from ai_agent_qbr.tools.status import ToolStatusEmitter
from qbr_intelligence.metrics.models import MetricCatalogEntry
from qbr_intelligence.metric_qa import MetricQueryEngine
from qbr_intelligence.metric_qa.intent import DeterministicIntentExtractor
from qbr_intelligence.metric_qa.resolvers import (
    ClientResolver,
    MetricResolver,
    PeriodResolver,
    RegionResolver,
)


_YEAR_RE = re.compile(r"\b(20\d{2}|\d{2})\b")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HALF_MULTI_RE = re.compile(r"\bH([12])\b[^.]*?\bH([12])\b[^.]*?\b(20\d{2}|\d{2})\b", re.IGNORECASE)
_QUARTER_MULTI_RE = re.compile(
    r"\bQ([1-4])\b[^.]*?\bQ([1-4])\b[^.]*?\b(20\d{2}|\d{2})\b",
    re.IGNORECASE,
)
_HALF_EXPLICIT_RE = re.compile(r"\bH([12])\s*(20\d{2}|\d{2})\b", re.IGNORECASE)
_QUARTER_EXPLICIT_RE = re.compile(r"\bQ([1-4])\s*(20\d{2}|\d{2})\b", re.IGNORECASE)
_COMPACT_PERIOD_RE = re.compile(r"\b([hq])\s*([1-4])\s*([0-9]{2,4})\b", re.IGNORECASE)
_REGION_CANONICAL = {"us": "US", "usa": "US", "u s": "US", "emea": "EMEA", "global": "GLOBAL"}


def _extract_multi_period_specs(query: str, anchor_date) -> list[PeriodSpec]:
    text = (query or "").strip()
    if not text:
        return []

    candidates: list[str] = []

    for match in _HALF_MULTI_RE.finditer(text):
        year = match.group(3)
        candidates.extend([f"H{match.group(1)} {year}", f"H{match.group(2)} {year}"])
    for match in _QUARTER_MULTI_RE.finditer(text):
        year = match.group(3)
        candidates.extend([f"Q{match.group(1)} {year}", f"Q{match.group(2)} {year}"])

    for match in _HALF_EXPLICIT_RE.finditer(text):
        candidates.append(f"H{match.group(1)} {match.group(2)}")
    for match in _QUARTER_EXPLICIT_RE.finditer(text):
        candidates.append(f"Q{match.group(1)} {match.group(2)}")

    if not candidates:
        year_match = _YEAR_RE.search(text)
        if year_match:
            year = year_match.group(1)
            halves = re.findall(r"\bH([12])\b", text, flags=re.IGNORECASE)
            quarters = re.findall(r"\bQ([1-4])\b", text, flags=re.IGNORECASE)
            if len(set(halves)) >= 2:
                for half in halves:
                    candidates.append(f"H{half} {year}")
            elif len(set(quarters)) >= 2:
                for quarter in quarters:
                    candidates.append(f"Q{quarter} {year}")

    if not candidates:
        return []

    resolver = PeriodResolver()
    seen_keys: set[tuple] = set()
    specs: list[PeriodSpec] = []
    for candidate in candidates:
        bounds, label, period_type = resolver.resolve(candidate, anchor_date=anchor_date)
        if not bounds:
            continue
        key = (period_type, label, bounds.start, bounds.end)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        specs.append(
            PeriodSpec(
                type=period_type,
                value=label,
                start=bounds.start,
                end=bounds.end,
            )
        )
    return specs


def _normalize_text(value: str) -> str:
    return " ".join(_TOKEN_RE.findall((value or "").lower()))


def _query_ngrams(query: str, max_words: int = 4) -> list[str]:
    tokens = _TOKEN_RE.findall((query or "").lower())
    if not tokens:
        return []
    grams = {" ".join(tokens)}
    n = len(tokens)
    for size in range(1, min(max_words, n) + 1):
        for i in range(0, n - size + 1):
            grams.add(" ".join(tokens[i : i + size]))
    return [g for g in grams if g]


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


def _fuzzy_metric_from_query(query: str, catalog: list[MetricCatalogEntry]) -> str | None:
    alias_to_metric: dict[str, str] = {}
    for entry in catalog:
        alias_to_metric[_normalize_text(entry.name)] = entry.metric_id
        alias_to_metric[_normalize_text(entry.metric_id.replace("_", " "))] = entry.metric_id
        for alias in entry.aliases:
            if alias.alias:
                alias_to_metric[_normalize_text(alias.alias)] = entry.metric_id
    return _best_fuzzy_match(probes=_query_ngrams(query, max_words=5), choices=alias_to_metric, min_ratio=0.82)


def _fuzzy_client_from_query(query: str, clients: list[str]) -> str | None:
    choices = {_normalize_text(client): client for client in clients if client}
    return _best_fuzzy_match(probes=_query_ngrams(query, max_words=4), choices=choices, min_ratio=0.74)


def _fuzzy_region_from_query(query: str) -> str | None:
    return _best_fuzzy_match(probes=_query_ngrams(query, max_words=3), choices=_REGION_CANONICAL, min_ratio=0.78)


def _fuzzy_period_from_query(query: str, anchor_date) -> list[PeriodSpec]:
    candidates: list[str] = []
    for match in _COMPACT_PERIOD_RE.finditer(query or ""):
        code, number, year = match.group(1).upper(), match.group(2), match.group(3)
        if code == "H" and number in {"1", "2"}:
            candidates.append(f"H{number} {year}")
        if code == "Q" and number in {"1", "2", "3", "4"}:
            candidates.append(f"Q{number} {year}")
    return _resolve_period_candidates(candidates, anchor_date=anchor_date)


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


def _coerce_period_specs(raw_periods: list[object] | None) -> list[PeriodSpec]:
    period_specs: list[PeriodSpec] = []
    for period in raw_periods or []:
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
    return period_specs


def _resolve_period_candidates(candidates: list[str], anchor_date) -> list[PeriodSpec]:
    resolver = PeriodResolver()
    seen_keys: set[tuple] = set()
    specs: list[PeriodSpec] = []
    for candidate in candidates:
        bounds, label, period_type = resolver.resolve(candidate, anchor_date=anchor_date)
        if not bounds:
            continue
        key = (period_type, label, bounds.start, bounds.end)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        specs.append(
            PeriodSpec(
                type=period_type,
                value=label,
                start=bounds.start,
                end=bounds.end,
            )
        )
    return specs


def _apply_proposed_entities(
    *,
    intent: ResolvedMetricIntent,
    proposals: CandidateSet | None,
    query: str,
    catalog: list[MetricCatalogEntry],
    clients: list[str],
    anchor_date,
) -> ResolvedMetricIntent:
    if proposals is None:
        return intent

    updated = ResolvedMetricIntent.model_validate(intent.model_dump())
    metric_resolver = MetricResolver(catalog)
    client_resolver = ClientResolver(clients)
    region_resolver = RegionResolver()

    if not updated.metric_ids and proposals.metric_ids:
        metric_hits: list[str] = []
        for candidate in proposals.metric_ids:
            resolved, _ = metric_resolver.resolve(candidate)
            if resolved:
                metric_hits.append(resolved[0])
        if not metric_hits:
            fuzzy_metric = _fuzzy_metric_from_query(" ".join(proposals.metric_ids), catalog)
            if fuzzy_metric:
                metric_hits.append(fuzzy_metric)
        metric_hits = list(dict.fromkeys(metric_hits))
        if len(metric_hits) == 1:
            updated.metric_ids = [metric_hits[0]]

    if not updated.client and proposals.clients:
        client_hits: list[str] = []
        for candidate in proposals.clients:
            resolution = client_resolver.resolve(candidate)
            if resolution.value:
                client_hits.append(resolution.value)
        if not client_hits:
            client_hits = _like_match(clients, proposals.clients)
        if not client_hits:
            fuzzy_client = _fuzzy_client_from_query(" ".join(proposals.clients), clients)
            if fuzzy_client:
                client_hits = [fuzzy_client]
        client_hits = list(dict.fromkeys(client_hits))
        if client_hits:
            updated.client = client_hits

    if not updated.region and proposals.regions:
        region_hits: list[str] = []
        candidate_hits = _query_token_match(query, proposals.regions) or proposals.regions
        for candidate in candidate_hits:
            resolution = region_resolver.resolve(candidate)
            if resolution.value:
                region_hits.append(resolution.value)
        if not region_hits:
            fuzzy_region = _fuzzy_region_from_query(" ".join(proposals.regions))
            if fuzzy_region:
                region_hits.append(fuzzy_region)
        region_hits = list(dict.fromkeys(region_hits))
        if len(region_hits) == 1:
            updated.region = region_hits

    proposed_periods = _resolve_period_candidates(proposals.periods, anchor_date=anchor_date)
    if not updated.period and proposed_periods:
        updated.period = proposed_periods
    elif len(updated.period) < 2 and len(proposed_periods) >= 2:
        # Let proposed multi-periods upgrade single-period parse for comparisons.
        updated.period = proposed_periods

    return updated


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

    period_specs = _coerce_period_specs(intent.period if isinstance(intent.period, list) else [intent.period])
    multi_period_specs = _extract_multi_period_specs(canonical_query, anchor_date)
    if len(multi_period_specs) >= 2:
        period_specs = multi_period_specs

    resolved_pre_proposals = ResolvedMetricIntent(
        metric_ids=list(intent.metric_ids),
        client=intent.client if isinstance(intent.client, list) else ([intent.client] if intent.client else []),
        region=intent.region if isinstance(intent.region, list) else ([intent.region] if intent.region else []),
        period=period_specs,
        aggregation=intent.aggregation,
        grouping=intent.grouping,
        limit=intent.limit,
    )
    resolved_after_proposals = _apply_proposed_entities(
        intent=resolved_pre_proposals,
        proposals=args.proposed_entities,
        query=canonical_query,
        catalog=list(engine._catalog or []),
        clients=list(engine._clients or []),
        anchor_date=anchor_date,
    )
    catalog_entries = list(engine._catalog or [])
    client_list = list(engine._clients or [])

    if not resolved_after_proposals.metric_ids:
        metric_fuzzy = _fuzzy_metric_from_query(canonical_query, catalog_entries)
        if metric_fuzzy:
            resolved_after_proposals.metric_ids = [metric_fuzzy]
    if not resolved_after_proposals.client:
        client_fuzzy = _fuzzy_client_from_query(canonical_query, client_list)
        if client_fuzzy:
            resolved_after_proposals.client = [client_fuzzy]
    if not resolved_after_proposals.region:
        region_fuzzy = _fuzzy_region_from_query(canonical_query)
        if region_fuzzy:
            resolved_after_proposals.region = [region_fuzzy]
    if not resolved_after_proposals.period:
        period_fuzzy = _fuzzy_period_from_query(canonical_query, anchor_date=anchor_date)
        if period_fuzzy:
            resolved_after_proposals.period = period_fuzzy

    confidence = FieldConfidence(
        metric=max(debug.metrics.values(), default=(0.8 if resolved_after_proposals.metric_ids else 0.0)),
        client=max(debug.client_confidence, 0.8 if resolved_after_proposals.client else 0.0),
        region=max(debug.region_confidence, 0.9 if resolved_after_proposals.region else 0.0),
        period=1.0 if resolved_after_proposals.period else 0.0,
    )

    client_values = resolved_after_proposals.client
    region_values = resolved_after_proposals.region
    if not region_values:
        fallback = RegionResolver().resolve(canonical_query)
        if fallback.value:
            region_values = [fallback.value]
        elif fallback.candidates:
            region_values = list(fallback.candidates)

    resolved = ResolvedMetricIntent(
        metric_ids=list(resolved_after_proposals.metric_ids),
        client=client_values,
        region=region_values,
        period=resolved_after_proposals.period,
        aggregation=resolved_after_proposals.aggregation,
        grouping=resolved_after_proposals.grouping,
        limit=resolved_after_proposals.limit,
    )

    result = ResolveMetricIntentResult(intent=resolved, confidence=confidence)
    interaction_metadata = ctx.tool_context.get("interaction_metadata")
    if isinstance(interaction_metadata, dict):
        interaction_metadata["resolved_metric_intent"] = result.intent.model_dump(mode="json")
    logging.getLogger(__name__).info("RESOLVE_INTENT_RESULT %s", result.model_dump())
    return result
