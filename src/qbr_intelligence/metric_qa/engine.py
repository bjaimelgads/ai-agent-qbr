"""Metric QA engine combining intent, planning, and answers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
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
from .intent import DeterministicIntentExtractor, IntentDebug, classify_intent_type
from .resolvers import ClientResolver, MetricResolver, PeriodResolver, RegionResolver
from .planner import build_plan

_LOGGER = logging.getLogger(__name__)

@dataclass
class MetricQueryResult:
    answer: MetricAnswer
    intent: QueryIntent
    debug: dict | None = None


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

    async def _load_catalogs(self) -> None:
        if self._catalog is not None and self._clients is not None:
            return
        catalog_rows = await self._store.fetch_metric_catalog()
        if not catalog_rows:
            self._catalog = build_metric_catalog()
        else:
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

    async def query(
        self,
        query: str,
        *,
        debug: bool = False,
        trace_id: str | None = None,
    ) -> MetricQueryResult:
        await self._load_catalogs()
        anchor_date = await self._select_anchor_date(query)

        cache_key = f"intent::{query.strip().lower()}"
        cached = self._intent_cache.get(cache_key)
        if cached:
            intent, intent_debug = cached
        else:
            intent, intent_debug = await self._extract_intent(query, anchor_date=anchor_date)
            self._intent_cache.set(cache_key, (intent, intent_debug))

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
        debug_payload: dict[str, Any] = {
            "intent": intent.model_dump(),
            "intent_debug": intent_debug.__dict__ if intent_debug else None,
        }

        rows, sql, params = await self._store.query_facts(
            metric_ids=plan.metric_ids,
            client_name=plan.client,
            region=plan.region,
            period_start=plan.period_start,
            period_end=plan.period_end,
            limit=plan.limit,
            order_by=plan.order_by,
        )
        debug_payload.update({"sql": sql, "params": params, "row_count": len(rows)})
        if (os.getenv("METRIC_QA_RETRIEVAL_LOG") or "").lower() in {"1", "true", "yes"}:
            _LOGGER.info(
                "metric_qa_select trace_id=%s sql=%s params=%s row_count=%s",
                trace_id,
                sql,
                params,
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
            )

        answer = AnswerComposer().compose(intent=intent, rows=rows, assumptions=assumptions)
        if self._llm_answer_enabled and self._answer_llm:
            answer = await self._phrase_answer(answer, intent=intent, rows=rows)

        if debug:
            answer.debug = debug_payload

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
            if merged.client and merged.client != client_resolution.value:
                assumptions.append("Client overridden by deterministic resolver")
            merged.client = client_resolution.value

        region_resolution = region_resolver.resolve(query)
        if region_resolution.value:
            if merged.region and merged.region != region_resolution.value:
                assumptions.append("Region overridden by deterministic resolver")
            merged.region = region_resolution.value

        bounds, label, period_type = period_resolver.resolve(query, anchor_date=anchor_date)
        if bounds:
            merged.period = deterministic.period
            if merged.period and merged.period.value and merged.period.value != label:
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
    ):
        rows = []
        reason = []

        if plan.client:
            assumptions.append("Relaxed client filter")
            reason.append("client")
            rows, sql, params = await self._store.query_facts(
                metric_ids=plan.metric_ids,
                client_name=None,
                region=plan.region,
                period_start=plan.period_start,
                period_end=plan.period_end,
                limit=plan.limit,
                order_by=plan.order_by,
            )
            debug_payload.update({"fallback_client_sql": sql, "fallback_client_params": params})
            if rows:
                debug_payload["fallback_reason"] = "client"
                return rows, assumptions, debug_payload

        if plan.region:
            assumptions.append("Relaxed region filter")
            reason.append("region")
            rows, sql, params = await self._store.query_facts(
                metric_ids=plan.metric_ids,
                client_name=plan.client,
                region=None,
                period_start=plan.period_start,
                period_end=plan.period_end,
                limit=plan.limit,
                order_by=plan.order_by,
            )
            debug_payload.update({"fallback_region_sql": sql, "fallback_region_params": params})
            if rows:
                debug_payload["fallback_reason"] = "region"
                return rows, assumptions, debug_payload

        if not plan.metric_ids:
            matched_ids = await self._embedding_match_metric(query)
            if matched_ids:
                assumptions.append("Used embedding match for metric catalog")
                rows, sql, params = await self._store.query_facts(
                    metric_ids=matched_ids,
                    client_name=plan.client,
                    region=plan.region,
                    period_start=plan.period_start,
                    period_end=plan.period_end,
                    limit=plan.limit,
                    order_by=plan.order_by,
                )
                debug_payload.update({"fallback_metric_sql": sql, "fallback_metric_params": params})
                if rows:
                    debug_payload["fallback_reason"] = "metric_embeddings"
                    return rows, assumptions, debug_payload

        debug_payload["fallback_reason"] = ",".join(reason) if reason else "none"
        return rows, assumptions, debug_payload

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
