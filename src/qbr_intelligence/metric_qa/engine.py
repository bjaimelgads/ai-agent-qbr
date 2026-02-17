"""Metric QA engine combining intent, planning, and answers."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import date
import logging
import re
from typing import Any, Callable
import os

from qbr_intelligence.metrics.catalog import build_metric_catalog
from qbr_intelligence.metrics.models import MetricAliasSpec, MetricCatalogEntry
from qbr_intelligence.schemas.metric_qa import MetricAnswer, QueryIntent

from .cache import LruCache
from .composer import AnswerComposer
from .dao import MetricFactStore
from .faiss_index import MetricFactFaissIndex
from .intent import DeterministicIntentExtractor, IntentDebug, classify_intent_type
from .resolvers import ClientResolver, MetricResolver, PeriodResolver, RegionResolver
from .planner import build_plan

_LOGGER = logging.getLogger("uvicorn.error")
_CONTEXT_QUALIFIER_TOKENS = (
    "campaign",
    "roadblock",
    "bundle",
    "creative",
    "spotlight",
    "sponsorship",
    "carousel",
    "video",
    "static",
    "segment",
    "targeting",
    "lapsed",
    "installed",
    "ros",
)


def _debug_log_enabled() -> bool:
    return (os.getenv("METRIC_QA_DEBUG_LOG") or "").lower() in {"1", "true", "yes", "on"}


@dataclass
class MetricQueryResult:
    answer: MetricAnswer
    intent: QueryIntent
    debug: dict | None = None


@dataclass(frozen=True)
class RagSlideRange:
    document_id: int
    start_slide: int | None = None
    end_slide: int | None = None


def _hash_payload(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class MetricQueryEngine:
    def __init__(
        self,
        *,
        database_url: str,
        embeddings_provider: Any | None = None,
        llm_intent_enabled: bool = False,
        llm_answer_enabled: bool = False,
        llm_max_calls_per_query: int = 1,
        intent_llm: Callable[[str, dict], dict] | None = None,
        answer_llm: Callable[[str, dict], str] | None = None,
    ) -> None:
        self._store = MetricFactStore(database_url)
        self._embeddings = embeddings_provider
        self._llm_intent_enabled = llm_intent_enabled
        self._llm_answer_enabled = llm_answer_enabled
        self._llm_max_calls = llm_max_calls_per_query
        self._intent_llm = intent_llm
        self._answer_llm = answer_llm
        self._intent_cache = LruCache(max_size=256)
        self._answer_cache = LruCache(max_size=256)
        self._catalog: list[MetricCatalogEntry] | None = None
        self._clients: list[str] | None = None
        self._semantic_rerank_threshold = int(os.getenv("METRIC_SEMANTIC_RERANK_THRESHOLD", "5"))
        self._semantic_rerank_top_k = int(os.getenv("METRIC_SEMANTIC_TOP_K", "5"))
        self._semantic_faiss_top_k = int(os.getenv("METRIC_FAISS_TOP_K", "200"))
        self._semantic_min_score = float(os.getenv("METRIC_SEMANTIC_MIN_SCORE", "0.6"))
        self._semantic_score_margin = float(os.getenv("METRIC_SEMANTIC_SCORE_MARGIN", "0.05"))
        self._metric_faiss = MetricFactFaissIndex.from_env()

    async def resolve_intent(
        self,
        query: str,
        *,
        use_cache: bool = True,
    ) -> tuple[QueryIntent, IntentDebug | None]:
        await self._load_catalogs()
        anchor_date = await self._select_anchor_date(query)
        cache_key = f"intent::{query.strip().lower()}"
        if use_cache:
            cached = self._intent_cache.get(cache_key)
            if cached:
                return cached
        intent, intent_debug = await self._extract_intent(query, anchor_date=anchor_date)
        if use_cache:
            self._intent_cache.set(cache_key, (intent, intent_debug))
        return intent, intent_debug

    async def _load_catalogs(self) -> None:
        if self._catalog is not None and self._clients is not None:
            return
        catalog_rows = await self._store.fetch_metric_catalog()
        if not catalog_rows:
            self._catalog = build_metric_catalog()
            catalog_source = "fallback_builtin"
        else:
            catalog_source = "database"
            catalog_map: dict[int, dict[str, Any]] = {}
            for row in catalog_rows:
                key = int(row["id"])
                entry = catalog_map.setdefault(
                    key,
                    {
                        "metric_id": row.get("slug") or row.get("name"),
                        "name": row.get("name"),
                        "aliases": [],
                        "expected_unit": row.get("default_unit") or "unknown",
                        "priority": 0,
                    },
                )
                alias = row.get("alias")
                if alias:
                    entry["aliases"].append(
                        MetricAliasSpec(
                            alias=alias,
                            pattern=row.get("pattern"),
                            priority=row.get("priority"),
                            unit_override=row.get("unit_override"),
                        )
                    )
            self._catalog = [
                MetricCatalogEntry(
                    metric_id=value["metric_id"],
                    name=value["name"],
                    aliases=tuple(value["aliases"]) or (MetricAliasSpec(alias=value["name"]),),
                    expected_unit=value["expected_unit"],
                    priority=value["priority"],
                )
                for value in catalog_map.values()
            ]
        self._clients = await self._store.fetch_clients()
        if _debug_log_enabled():
            _LOGGER.info(
                "metric_qa_catalog_loaded source=%s catalog_size=%s client_count=%s",
                catalog_source,
                len(self._catalog or []),
                len(self._clients or []),
            )

    async def query(
        self,
        query: str,
        *,
        debug: bool = False,
        trace_id: str | None = None,
        intent_override: QueryIntent | None = None,
        strict_entity_filters: bool = False,
        rag_document_ids: list[int] | None = None,
        rag_slide_ranges: list[RagSlideRange] | None = None,
        rag_slide_ids: list[int] | None = None,
        rag_hit_chunks: list[dict[str, Any]] | None = None,
    ) -> MetricQueryResult:
        await self._load_catalogs()
        if intent_override is not None:
            intent = intent_override.model_copy(deep=True)
            intent_debug = None
        else:
            intent, intent_debug = await self.resolve_intent(query)

        intent_type = classify_intent_type(query, intent)
        if intent_type == "DEFINITION":
            answer = await self._answer_definition(intent)
            if debug:
                answer.debug = {"intent": intent.model_dump(), "intent_type": intent_type}
            return MetricQueryResult(answer=answer, intent=intent, debug=answer.debug)

        if intent.aggregation == "latest":
            intent.aggregation = "all"

        plan = build_plan(intent)
        assumptions: list[str] = []
        rag_scope_ranges = rag_slide_ranges or []
        requested_context_terms = _extract_requested_context_terms(query)
        rag_context_signal = _assess_rag_context_signal(
            requested_context_terms=requested_context_terms,
            rag_hit_chunks=rag_hit_chunks,
        )
        prioritize_overall_context_override: bool | None = None
        if requested_context_terms:
            prioritize_overall_context_override = not bool(rag_context_signal.get("is_confident"))
        debug_payload: dict[str, Any] = {
            "intent": intent.model_dump(),
            "intent_debug": intent_debug.__dict__ if intent_debug else None,
            "strict_entity_filters": bool(strict_entity_filters),
            "requested_context_terms": requested_context_terms,
            "rag_context_signal": rag_context_signal,
            "prioritize_overall_context_override": prioritize_overall_context_override,
            "rag_scope": {
                "document_ids": sorted(set(rag_document_ids or [])),
                "slide_ids": sorted(set(rag_slide_ids or [])),
                "slide_ranges": [
                    {
                        "document_id": item.document_id,
                        "start_slide": item.start_slide,
                        "end_slide": item.end_slide,
                    }
                    for item in rag_scope_ranges
                ],
            },
        }
        if _debug_log_enabled():
            _LOGGER.info(
                "metric_qa_plan trace_id=%s intent_type=%s metrics=%s client=%s region=%s periods=%s aggregation=%s grouping=%s limit=%s",
                trace_id,
                intent_type,
                plan.metric_ids,
                plan.client,
                plan.region,
                [getattr(period, "value", None) for period in intent.period],
                plan.aggregation,
                plan.grouping,
                plan.limit,
            )

        rows, query_debug = await self._query_plan_rows(
            query=query,
            plan=plan,
            trace_id=trace_id,
            rag_document_ids=rag_document_ids,
            rag_slide_ranges=rag_scope_ranges,
            rag_slide_ids=rag_slide_ids,
            prioritize_overall_context_override=prioritize_overall_context_override,
        )
        rows = _filter_rows_by_explicit_periods(rows, intent.period)
        debug_payload.update(query_debug)
        if (os.getenv("METRIC_QA_RETRIEVAL_LOG") or "").lower() in {"1", "true", "yes"}:
            _LOGGER.info(
                "metric_qa_select trace_id=%s sql=%s params=%s row_count=%s",
                trace_id,
                query_debug.get("sql"),
                query_debug.get("params"),
                len(rows),
            )
            for row in rows[:20]:
                snippet = (row.snippet or "").replace("\n", " ").strip()[:240]
                context_label = row.llm_context_label or ""
                _LOGGER.info(
                    "metric_qa_row trace_id=%s metric=%s value=%s unit=%s period=%s client=%s region=%s "
                    "slide_id=%s context_label=%r snippet=%r",
                    trace_id,
                    row.metric_name,
                    row.value,
                    row.unit,
                    row.period_label or row.period_end,
                    row.client_name,
                    row.region,
                    row.slide_id,
                    context_label,
                    snippet,
                )

        if not rows:
            rows, assumptions, debug_payload = await self._fallback_query(
                query=query,
                intent=intent,
                plan=plan,
                assumptions=assumptions,
                debug_payload=debug_payload,
                strict_entity_filters=strict_entity_filters,
                prioritize_overall_context=bool(query_debug.get("prioritize_overall_context")),
                rag_document_ids=rag_document_ids,
                rag_slide_ranges=rag_scope_ranges,
                rag_slide_ids=rag_slide_ids,
            )
            rows = _filter_rows_by_explicit_periods(rows, intent.period)
        if _debug_log_enabled():
            _LOGGER.info(
                "metric_qa_post_query trace_id=%s row_count=%s fallback_reason=%s assumptions=%s",
                trace_id,
                len(rows),
                debug_payload.get("fallback_reason"),
                assumptions,
            )

        ambiguous = False

        answer = AnswerComposer().compose(intent=intent, rows=rows, assumptions=assumptions)
        if self._llm_answer_enabled and self._answer_llm:
            answer = await self._phrase_answer(answer, intent=intent, rows=rows)

        if ambiguous and rows:
            options = _format_clarification_options(rows, max_items=self._semantic_rerank_top_k)
            if options:
                answer.summary = "I found multiple plausible matches and need a bit more detail."
                answer.summary_text = (
                    "I found multiple plausible matches and need a bit more detail. "
                    "Which one did you mean? Options: " + " ".join(options)
                )
                answer.followups = options[:3]

        if debug:
            answer.debug = debug_payload
        if _debug_log_enabled():
            _LOGGER.info(
                "metric_qa_answer trace_id=%s confidence=%s rows=%s followups=%s",
                trace_id,
                answer.confidence,
                len(answer.data or []),
                len(answer.followups or []),
            )

        return MetricQueryResult(answer=answer, intent=intent, debug=debug_payload if debug else None)

    async def _select_anchor_date(self, query: str):
        lowered = query.lower()
        if "half" in lowered:
            anchor = await self._store.fetch_latest_period_end_by_granularity("half")
            if anchor:
                return anchor
        if "quarter" in lowered or _mentions_quarter(lowered):
            anchor = await self._store.fetch_latest_period_end_by_granularity("quarter")
            if anchor:
                return anchor
        return await self._store.fetch_latest_period_end()

    async def _query_plan_rows(
        self,
        *,
        query: str,
        plan,
        trace_id: str | None,
        rag_document_ids: list[int] | None,
        rag_slide_ranges: list[RagSlideRange],
        rag_slide_ids: list[int] | None,
        prioritize_overall_context_override: bool | None = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        default_prioritize_overall_context = _should_prioritize_overall_context(
            query=query,
            aggregation=plan.aggregation,
            grouping=plan.grouping,
            metric_ids=plan.metric_ids,
        )
        prioritize_overall_context = (
            prioritize_overall_context_override
            if prioritize_overall_context_override is not None
            else default_prioritize_overall_context
        )
        order_by = _build_order_by_with_overall_priority(
            base_order_by=plan.order_by,
            prioritize_overall_context=prioritize_overall_context,
        )
        rag_scope_active = bool((rag_document_ids or []) or rag_slide_ranges or (rag_slide_ids or []))
        scopes = self._build_comparison_scopes(plan, order_by=order_by)
        if not scopes:
            rows, sql, params = await self._store.query_facts(
                metric_ids=plan.metric_ids,
                client_name=plan.client,
                region=plan.region,
                period_ranges=plan.period_ranges,
                limit=plan.limit,
                order_by=order_by,
            )
            filtered = self._apply_rag_scope(
                rows,
                rag_document_ids=rag_document_ids,
                rag_slide_ranges=rag_slide_ranges,
                rag_slide_ids=rag_slide_ids,
            )
            rag_pruned_all_rows = rag_scope_active and bool(rows) and not filtered
            return filtered, {
                "sql": sql,
                "params": params,
                "row_count": len(filtered),
                "unscoped_row_count": len(rows),
                "rag_pruned_all_rows": rag_pruned_all_rows,
                "prioritize_overall_context": prioritize_overall_context,
                "default_prioritize_overall_context": default_prioritize_overall_context,
                "prioritize_overall_context_override": prioritize_overall_context_override,
            }

        async def _run_scope(scope: dict[str, Any]) -> tuple[list[Any], int, bool, str, dict[str, Any]]:
            rows, sql, params = await self._store.query_facts(
                metric_ids=scope["metric_ids"],
                client_name=scope["client_name"],
                region=scope["region"],
                period_ranges=scope["period_ranges"],
                limit=scope["limit"],
                order_by=scope["order_by"],
            )
            filtered = self._apply_rag_scope(
                rows,
                rag_document_ids=rag_document_ids,
                rag_slide_ranges=rag_slide_ranges,
                rag_slide_ids=rag_slide_ids,
            )
            rag_pruned_all_rows = rag_scope_active and bool(rows) and not filtered
            return filtered, len(rows), rag_pruned_all_rows, sql, params

        scope_results = await asyncio.gather(*(_run_scope(scope) for scope in scopes))
        merged: list[Any] = []
        seen_fact_ids: set[int] = set()
        scope_debug: list[dict[str, Any]] = []
        rag_pruned_all_any = False
        for idx, (rows, unscoped_count, rag_pruned_all_rows, sql, params) in enumerate(scope_results):
            rag_pruned_all_any = rag_pruned_all_any or rag_pruned_all_rows
            scope_debug.append(
                {
                    "scope_index": idx,
                    "scope": scopes[idx],
                    "sql": sql,
                    "params": params,
                    "row_count": len(rows),
                    "unscoped_row_count": unscoped_count,
                    "rag_pruned_all_rows": rag_pruned_all_rows,
                }
            )
            for row in rows:
                fact_id = getattr(row, "fact_id", None)
                if fact_id is not None:
                    if fact_id in seen_fact_ids:
                        continue
                    seen_fact_ids.add(fact_id)
                merged.append(row)

        if _debug_log_enabled():
            _LOGGER.info(
                "metric_qa_parallel_scopes trace_id=%s scope_count=%s merged_rows=%s",
                trace_id,
                len(scopes),
                len(merged),
            )

        return merged, {
            "parallel_scope": True,
            "scope_count": len(scopes),
            "scope_debug": scope_debug,
            "sql": scope_debug[0]["sql"] if scope_debug else None,
            "params": scope_debug[0]["params"] if scope_debug else {},
            "row_count": len(merged),
            "rag_pruned_all_rows": rag_pruned_all_any,
            "prioritize_overall_context": prioritize_overall_context,
            "default_prioritize_overall_context": default_prioritize_overall_context,
            "prioritize_overall_context_override": prioritize_overall_context_override,
        }

    def _build_comparison_scopes(self, plan, *, order_by: str) -> list[dict[str, Any]]:
        should_split = bool(plan.aggregation == "compare")
        should_split = should_split or len(plan.period_ranges) > 1
        should_split = should_split or len(plan.client) > 1
        should_split = should_split or len(plan.region) > 1
        if not should_split:
            return []

        scopes: list[dict[str, Any]] = []
        if len(plan.period_ranges) > 1:
            for period_range in plan.period_ranges:
                scopes.append(
                    {
                        "metric_ids": plan.metric_ids,
                        "client_name": plan.client,
                        "region": plan.region,
                        "period_ranges": [period_range],
                        "limit": plan.limit,
                        "order_by": order_by,
                    }
                )
            return scopes

        if len(plan.client) > 1:
            for client in plan.client:
                scopes.append(
                    {
                        "metric_ids": plan.metric_ids,
                        "client_name": [client],
                        "region": plan.region,
                        "period_ranges": plan.period_ranges,
                        "limit": plan.limit,
                        "order_by": order_by,
                    }
                )
            return scopes

        if len(plan.region) > 1:
            for region in plan.region:
                scopes.append(
                    {
                        "metric_ids": plan.metric_ids,
                        "client_name": plan.client,
                        "region": [region],
                        "period_ranges": plan.period_ranges,
                        "limit": plan.limit,
                        "order_by": order_by,
                    }
                )
            return scopes

        return []

    def _apply_rag_scope(
        self,
        rows: list[Any],
        *,
        rag_document_ids: list[int] | None,
        rag_slide_ranges: list[RagSlideRange],
        rag_slide_ids: list[int] | None,
    ) -> list[Any]:
        if not rows:
            return rows
        doc_filter = set(rag_document_ids or [])
        slide_filter = set(int(value) for value in (rag_slide_ids or []))
        ranges_by_doc: dict[int, list[tuple[int | None, int | None]]] = {}
        for item in rag_slide_ranges:
            ranges_by_doc.setdefault(int(item.document_id), []).append(
                (item.start_slide, item.end_slide)
            )
        if not doc_filter and not ranges_by_doc and not slide_filter:
            return rows

        filtered: list[Any] = []
        for row in rows:
            doc_id = int(getattr(row, "document_id", 0) or 0)
            if doc_filter and doc_id not in doc_filter:
                continue
            row_slide_id = getattr(row, "slide_id", None)
            if slide_filter:
                if row_slide_id is None or int(row_slide_id) not in slide_filter:
                    continue

            doc_ranges = ranges_by_doc.get(doc_id)
            if not doc_ranges:
                filtered.append(row)
                continue

            slide_number = getattr(row, "slide_number", None)
            if slide_number is None:
                continue
            slide_value = int(slide_number)
            matched = False
            for start, end in doc_ranges:
                if start is None and end is None:
                    matched = True
                    break
                if start is None:
                    if slide_value <= int(end):
                        matched = True
                        break
                    continue
                if end is None:
                    if slide_value >= int(start):
                        matched = True
                        break
                    continue
                if int(start) <= slide_value <= int(end):
                    matched = True
                    break
            if matched:
                filtered.append(row)
        return filtered

    async def _extract_intent(self, query: str, *, anchor_date):
        if not self._catalog or self._clients is None:
            await self._load_catalogs()
        extractor = DeterministicIntentExtractor(self._catalog or [], self._clients or [])
        det_intent, debug = extractor.extract(query, anchor_date=anchor_date)
        llm_intent = None
        if self._llm_intent_enabled and self._intent_llm and self._llm_max_calls > 0:
            llm_intent = await self._run_llm_intent(query)
        if llm_intent:
            merged, assumptions = self._merge_intents(
                query=query,
                llm_intent=llm_intent,
                deterministic=det_intent,
                anchor_date=anchor_date,
            )
            if assumptions:
                merged.clarifications_needed.extend(assumptions)
            return merged, debug
        return det_intent, debug

    async def _run_llm_intent(self, query: str) -> QueryIntent | None:
        if not self._intent_llm or not self._catalog:
            return None
        payload = self._build_llm_payload(query)
        raw = self._intent_llm(query, payload)
        if not raw:
            return None
        try:
            intent = QueryIntent.model_validate(raw)
        except Exception:
            return None
        return intent

    def _build_llm_payload(self, query: str) -> dict[str, Any]:
        metrics = []
        for entry in self._catalog or []:
            aliases = [alias.alias for alias in entry.aliases if alias.alias]
            metrics.append({"metric_id": entry.metric_id, "aliases": aliases[:5]})
        return {
            "query": query,
            "metrics": metrics,
            "regions": ["US", "EMEA", "GLOBAL"],
            "aggregations": ["latest", "trend", "compare", "average", "sum", "min", "max"],
            "period_examples": ["Q1 2025", "H2 2024", "last quarter", "past year"],
        }

    def _merge_intents(
        self,
        *,
        query: str,
        llm_intent: QueryIntent,
        deterministic: QueryIntent,
        anchor_date,
    ) -> tuple[QueryIntent, list[str]]:
        assumptions: list[str] = []
        merged = llm_intent.model_copy()

        metric_resolver = MetricResolver(self._catalog or [])
        client_resolver = ClientResolver(self._clients or [])
        region_resolver = RegionResolver()
        period_resolver = PeriodResolver()

        metric_ids, _ = metric_resolver.resolve(query)
        if metric_ids:
            if merged.metric_ids and set(merged.metric_ids) != set(metric_ids):
                assumptions.append("Metric IDs overridden by deterministic resolver")
            merged.metric_ids = metric_ids

        client_resolution = client_resolver.resolve(query)
        if client_resolution.value:
            if merged.client and merged.client != [client_resolution.value]:
                assumptions.append("Client overridden by deterministic resolver")
            merged.client = [client_resolution.value]

        region_resolution = region_resolver.resolve(query)
        if region_resolution.value:
            if merged.region and merged.region != [region_resolution.value]:
                assumptions.append("Region overridden by deterministic resolver")
            merged.region = [region_resolution.value]

        bounds, label, period_type = period_resolver.resolve(query, anchor_date=anchor_date)
        if bounds:
            merged.period = deterministic.period
            if merged.period and merged.period[0].value and merged.period[0].value != label:
                assumptions.append("Period overridden by deterministic resolver")
            if deterministic.period:
                merged.period = deterministic.period
        return merged, assumptions

    async def _answer_definition(self, intent: QueryIntent) -> MetricAnswer:
        if not intent.metric_ids:
            return MetricAnswer(
                summary="I couldn't identify a metric to define.",
                summary_text="I couldn't identify a metric to define.",
                confidence=0.0,
                assumptions=["Metric not resolved."],
            )
        metric_ids = intent.metric_ids
        definitions = await self._store.fetch_metric_definitions(metric_ids)
        if not definitions and self._catalog:
            catalog_map = {entry.metric_id: entry for entry in self._catalog}
            entry = catalog_map.get(metric_ids[0])
            if entry:
                definitions = [
                    {
                        "slug": entry.metric_id,
                        "name": entry.name,
                        "description": entry.description,
                        "formula": entry.formula,
                        "default_unit": entry.expected_unit,
                    }
                ]
        if not definitions:
            return MetricAnswer(
                summary="I couldn't find a definition for that metric.",
                summary_text="I couldn't find a definition for that metric.",
                confidence=0.0,
                assumptions=["Metric definition missing."],
            )
        entry = definitions[0]
        parts = [f"{entry.get('name') or entry.get('slug')} definition:"]
        if entry.get("description"):
            parts.append(entry["description"])
        if entry.get("formula"):
            parts.append(f"Formula: {entry['formula']}")
        if entry.get("default_unit"):
            parts.append(f"Expected unit: {entry['default_unit']}")
        summary = " ".join(parts)
        return MetricAnswer(
            summary=summary,
            summary_text=summary,
            confidence=0.7,
            assumptions=[],
            citations=[],
        )

    async def _phrase_answer(
        self,
        answer: MetricAnswer,
        *,
        intent: QueryIntent,
        rows,
    ) -> MetricAnswer:
        cache_key = _hash_payload({"intent": intent.model_dump(), "rows": [r.__dict__ for r in rows]})
        cached = self._answer_cache.get(cache_key)
        if cached:
            answer.summary_text = cached
            return answer
        if not self._answer_llm:
            return answer
        payload = {
            "intent": intent.model_dump(),
            "summary": answer.summary_text,
            "table": answer.table_data,
        }
        try:
            response = self._answer_llm(answer.summary_text, payload)
        except Exception:
            response = None
        if isinstance(response, str) and response.strip():
            answer.summary_text = response.strip()
            self._answer_cache.set(cache_key, answer.summary_text)
        return answer

    async def _fallback_query(
        self,
        *,
        query: str,
        intent: QueryIntent,
        plan,
        assumptions: list[str],
        debug_payload: dict[str, Any],
        strict_entity_filters: bool,
        prioritize_overall_context: bool,
        rag_document_ids: list[int] | None,
        rag_slide_ranges: list[RagSlideRange],
        rag_slide_ids: list[int] | None,
    ):
        rows = []
        reason = []
        order_by = _build_order_by_with_overall_priority(
            base_order_by=plan.order_by,
            prioritize_overall_context=prioritize_overall_context,
        )

        if plan.client and not strict_entity_filters:
            assumptions.append("Relaxed client filter")
            reason.append("client")
            rows, sql, params = await self._store.query_facts(
                metric_ids=plan.metric_ids,
                client_name=None,
                region=plan.region,
                period_ranges=plan.period_ranges,
                limit=plan.limit,
                order_by=order_by,
            )
            rows = self._apply_rag_scope(
                rows,
                rag_document_ids=rag_document_ids,
                rag_slide_ranges=rag_slide_ranges,
                rag_slide_ids=rag_slide_ids,
            )
            debug_payload.update({"fallback_client_sql": sql, "fallback_client_params": params})
            if rows:
                debug_payload["fallback_reason"] = "client"
                return rows, assumptions, debug_payload

        if plan.region and not strict_entity_filters:
            assumptions.append("Relaxed region filter")
            reason.append("region")
            rows, sql, params = await self._store.query_facts(
                metric_ids=plan.metric_ids,
                client_name=plan.client,
                region=None,
                period_ranges=plan.period_ranges,
                limit=plan.limit,
                order_by=order_by,
            )
            rows = self._apply_rag_scope(
                rows,
                rag_document_ids=rag_document_ids,
                rag_slide_ranges=rag_slide_ranges,
                rag_slide_ids=rag_slide_ids,
            )
            debug_payload.update({"fallback_region_sql": sql, "fallback_region_params": params})
            if rows:
                debug_payload["fallback_reason"] = "region"
                return rows, assumptions, debug_payload

        if strict_entity_filters and (plan.client or plan.region or plan.period_ranges):
            debug_payload["fallback_reason"] = "strict_entity_filters"
            return rows, assumptions, debug_payload

        if not plan.metric_ids:
            matched_ids = await self._embedding_match_metric(query)
            if matched_ids:
                assumptions.append("Used embedding match for metric catalog")
                rows, sql, params = await self._store.query_facts(
                    metric_ids=matched_ids,
                    client_name=plan.client,
                    region=plan.region,
                    period_ranges=plan.period_ranges,
                    limit=plan.limit,
                    order_by=order_by,
                )
                rows = self._apply_rag_scope(
                    rows,
                    rag_document_ids=rag_document_ids,
                    rag_slide_ranges=rag_slide_ranges,
                    rag_slide_ids=rag_slide_ids,
                )
                debug_payload.update({"fallback_metric_sql": sql, "fallback_metric_params": params})
                if rows:
                    debug_payload["fallback_reason"] = "metric_embeddings"
                    return rows, assumptions, debug_payload

        debug_payload["fallback_reason"] = ",".join(reason) if reason else "none"
        return rows, assumptions, debug_payload

    async def _semantic_rerank_rows(self, query: str, rows):
        debug: dict[str, Any] = {"skipped": False, "reason": None}
        if not self._embeddings:
            debug.update({"skipped": True, "reason": "no_embeddings_provider"})
            return rows, debug
        try:
            query_embedding = await self._embeddings.embed_query(query)
        except Exception:
            debug.update({"skipped": True, "reason": "query_embedding_failed"})
            return rows, debug

        vector = query_embedding.vector
        if hasattr(vector, "values"):
            vector = list(vector.values)
        else:
            vector = list(vector)

        embedding_model = getattr(query_embedding, "model", None)
        fact_ids = [row.fact_id for row in rows if row.fact_id is not None]
        embeddings = await self._store.fetch_metric_fact_embeddings(
            fact_ids,
            embedding_model=embedding_model,
        )

        scores: dict[int, float] = {}
        faiss_scores = self._metric_faiss.search(
            vector=vector,
            top_k=self._semantic_faiss_top_k,
            embedding_model=embedding_model or self._metric_faiss.embedding_model,
        )
        for fact_id, score in faiss_scores:
            scores[int(fact_id)] = float(score)
        if not embeddings and not scores:
            debug.update({"skipped": True, "reason": "no_metric_embeddings"})
            return rows, debug

        missing_embeddings = 0
        for row in rows:
            fact_id = row.fact_id
            score = scores.get(fact_id)
            if score is None:
                embedding = embeddings.get(fact_id)
                if embedding:
                    score = _cosine_similarity(vector, embedding)
                else:
                    missing_embeddings += 1
                    score = 0.0
            row.semantic_score = float(score)

        ranked = sorted(rows, key=lambda r: (r.semantic_score or 0.0), reverse=True)
        top_score = ranked[0].semantic_score or 0.0
        second_score = ranked[1].semantic_score if len(ranked) > 1 else None
        ambiguous = False
        if top_score < self._semantic_min_score:
            ambiguous = True
        if second_score is not None and (top_score - second_score) < self._semantic_score_margin:
            ambiguous = True
        top_k = min(self._semantic_rerank_top_k, len(ranked))
        selected = ranked[:top_k]
        selected = sorted(
            selected,
            key=lambda r: (
                r.slide_number is None,
                r.slide_number or r.slide_id or 0,
                -(r.semantic_score or 0.0),
            ),
        )

        debug.update(
            {
                "skipped": False,
                "total_rows": len(rows),
                "selected_rows": len(selected),
                "missing_embeddings": missing_embeddings,
                "top_score": top_score,
                "second_score": second_score,
                "ambiguous": ambiguous,
            }
        )
        return selected, debug

    async def _embedding_match_metric(self, query: str) -> list[str]:
        if not self._embeddings or not self._catalog:
            return []
        try:
            query_embedding = await self._embeddings.embed_query(query)
        except Exception:
            return []
        candidates = self._catalog
        scored: list[tuple[str, float]] = []
        for entry in candidates:
            try:
                entry_emb = await self._embeddings.embed_query(entry.name)
            except Exception:
                continue
            score = _cosine_similarity(query_embedding.vector, entry_emb.vector)
            scored.append((entry.metric_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        if not scored:
            return []
        top_id, top_score = scored[0]
        if top_score < 0.7:
            return []
        return [top_id]


def _cosine_similarity(vec_a, vec_b) -> float:
    if not vec_a or not vec_b:
        return 0.0
    total = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    return float(total)


def _mentions_quarter(text: str) -> bool:
    return bool(re.search(r"\bq[1-4]\b", text))


def _extract_requested_context_terms(query: str) -> list[str]:
    lowered = query.lower()
    requested = [token for token in _CONTEXT_QUALIFIER_TOKENS if token in lowered]
    seen: set[str] = set()
    deduped: list[str] = []
    for token in requested:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _assess_rag_context_signal(
    *,
    requested_context_terms: list[str],
    rag_hit_chunks: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    threshold = float(os.getenv("METRIC_CONTEXT_RAG_CONFIDENCE_MIN", "0.35"))
    if not requested_context_terms:
        return {
            "explicit_context_requested": False,
            "is_confident": False,
            "matched_term": None,
            "best_score": None,
            "threshold": threshold,
            "reason": "no_context_requested",
        }
    if not rag_hit_chunks:
        return {
            "explicit_context_requested": True,
            "is_confident": False,
            "matched_term": None,
            "best_score": None,
            "threshold": threshold,
            "reason": "no_rag_hits",
        }
    best_score = -1.0
    matched_term: str | None = None
    for chunk in rag_hit_chunks:
        content = str(chunk.get("content") or "").lower()
        summary = str(chunk.get("summary") or "").lower()
        topics = chunk.get("topics") or []
        topics_text = " ".join(str(item) for item in topics).lower()
        metadata = chunk.get("metadata") or {}
        metadata_text = " ".join(f"{k} {v}" for k, v in metadata.items()).lower()
        haystack = " ".join([content, summary, topics_text, metadata_text])
        score = float(chunk.get("score") or 0.0)
        for term in requested_context_terms:
            if term not in haystack:
                continue
            if score > best_score:
                best_score = score
                matched_term = term
    if matched_term is None:
        return {
            "explicit_context_requested": True,
            "is_confident": False,
            "matched_term": None,
            "best_score": None,
            "threshold": threshold,
            "reason": "no_context_match_in_hits",
        }
    return {
        "explicit_context_requested": True,
        "is_confident": best_score >= threshold,
        "matched_term": matched_term,
        "best_score": best_score,
        "threshold": threshold,
        "reason": "matched_context_in_hits",
    }


def _should_prioritize_overall_context(
    *,
    query: str,
    aggregation: str | None,
    grouping: str | None,
    metric_ids: list[str],
) -> bool:
    if not metric_ids:
        return False
    del aggregation, grouping
    lowered = query.lower()
    return not any(token in lowered for token in _CONTEXT_QUALIFIER_TOKENS)


def _build_order_by_with_overall_priority(
    *,
    base_order_by: str,
    prioritize_overall_context: bool,
) -> str:
    if not prioritize_overall_context:
        return base_order_by
    return (
        f"{base_order_by}, "
        "CASE "
        "WHEN lower(coalesce(baseline_type, '')) = 'overall' THEN 0 "
        "ELSE 1 END ASC"
    )


def _filter_rows_by_explicit_periods(rows: list[Any], periods: list[Any] | None) -> list[Any]:
    if not rows or not periods:
        return rows
    explicit_specs = []
    for period in periods:
        start = getattr(period, "start", None)
        end = getattr(period, "end", None)
        if isinstance(start, date) and isinstance(end, date):
            explicit_specs.append(
                {
                    "start": start,
                    "end": end,
                    "value": str(getattr(period, "value", "") or "").strip().lower(),
                }
            )
    if not explicit_specs:
        return rows

    filtered: list[Any] = []
    for row in rows:
        row_start = _safe_parse_date(getattr(row, "period_start", None))
        row_end = _safe_parse_date(getattr(row, "period_end", None))
        row_label = str(getattr(row, "period_label", "") or "").strip().lower()
        matched = False
        for spec in explicit_specs:
            if row_label and spec["value"] and not _period_label_compatible(row_label, spec["value"]):
                continue
            if row_start and row_end:
                # Keep original overlap semantics for inferred periods,
                # while label compatibility above prevents H1/H2 leakage.
                if row_end >= spec["start"] and row_start <= spec["end"]:
                    matched = True
                    break
            if spec["value"] and row_label and row_label == spec["value"]:
                matched = True
                break
        if matched:
            filtered.append(row)
    return filtered


def _safe_parse_date(raw: Any) -> date | None:
    if raw is None:
        return None
    try:
        return date.fromisoformat(str(raw))
    except ValueError:
        return None


def _period_label_compatible(row_label: str, spec_label: str) -> bool:
    row = row_label.lower()
    spec = spec_label.lower()

    row_half = re.search(r"\bh([12])\b", row)
    spec_halves = set(re.findall(r"\bh([12])\b", spec))
    if spec_halves:
        if not row_half or row_half.group(1) not in spec_halves:
            return False

    row_quarter = re.search(r"\bq([1-4])\b", row)
    spec_quarters = set(re.findall(r"\bq([1-4])\b", spec))
    if spec_quarters:
        if not row_quarter or row_quarter.group(1) not in spec_quarters:
            return False

    spec_year = re.search(r"\b(20\d{2})\b", spec)
    if spec_year:
        year = spec_year.group(1)
        if year not in row and f"fy{year[-2:]}" not in row:
            return False
    return True


def _format_clarification_options(rows, *, max_items: int) -> list[str]:
    options: list[str] = []
    for row in rows[:max_items]:
        parts: list[str] = []
        if row.metric_name:
            parts.append(row.metric_name)
        if row.period_label or row.period_end:
            parts.append(f"period {row.period_label or row.period_end}")
        if row.client_name:
            parts.append(f"client {row.client_name}")
        if row.region:
            parts.append(f"region {row.region}")
        if row.llm_context_label:
            parts.append(f"context {row.llm_context_label}")
        if row.value is not None:
            parts.append(f"value {row.value}")
        label = ", ".join(parts)
        if label:
            options.append(label + ".")
    return options
