"""Metric QA engine combining intent, planning, and answers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
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


@dataclass
class MetricQueryResult:
    answer: MetricAnswer
    intent: QueryIntent
    debug: dict | None = None


def _hash_payload(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _merge_followup_intent(base: QueryIntent, followup: QueryIntent) -> QueryIntent:
    """Merge a follow-up intent onto a prior intent deterministically."""
    merged = base.model_copy()
    if followup.metric_ids:
        merged.metric_ids = followup.metric_ids
    if followup.client:
        merged.client = followup.client
    if followup.region:
        merged.region = followup.region
    if followup.period:
        merged.period = followup.period
    if followup.aggregation:
        merged.aggregation = followup.aggregation
    if followup.grouping:
        merged.grouping = followup.grouping
    if followup.limit:
        merged.limit = followup.limit
    if followup.clarifications_needed:
        merged.clarifications_needed.extend(followup.clarifications_needed)
    return merged


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

    async def query(self, query: str, *, debug: bool = False) -> MetricQueryResult:
        await self._load_catalogs()
        anchor_date = await self._select_anchor_date(query)

        cache_key = f"intent::{query.strip().lower()}"
        cached = self._intent_cache.get(cache_key)
        if cached:
            intent, intent_debug = cached
        else:
            intent, intent_debug = await self._extract_intent(query, anchor_date=anchor_date)
            self._intent_cache.set(cache_key, (intent, intent_debug))

        return await self._answer_from_intent(
            query=query,
            intent=intent,
            intent_debug=intent_debug,
            debug=debug,
        )

    async def query_with_prior_intent(
        self,
        query: str,
        *,
        prior_intent: dict[str, Any],
        debug: bool = False,
    ) -> MetricQueryResult:
        await self._load_catalogs()
        anchor_date = await self._select_anchor_date(query)
        extractor = DeterministicIntentExtractor(self._catalog or [], self._clients or [])
        followup_intent, intent_debug = extractor.extract(query, anchor_date=anchor_date)
        try:
            base_intent = QueryIntent.model_validate(prior_intent)
        except Exception:
            base_intent = QueryIntent()
        merged = _merge_followup_intent(base_intent, followup_intent)
        return await self._answer_from_intent(
            query=query,
            intent=merged,
            intent_debug=intent_debug,
            debug=debug,
        )

    async def _answer_from_intent(
        self,
        *,
        query: str,
        intent: QueryIntent,
        intent_debug: IntentDebug | None,
        debug: bool,
    ) -> MetricQueryResult:
        intent_type = classify_intent_type(query, intent)
        if intent_type == "DEFINITION":
            answer = await self._answer_definition(intent)
            if debug:
                answer.debug = {"intent": intent.model_dump(), "intent_type": intent_type}
            return MetricQueryResult(answer=answer, intent=intent, debug=answer.debug)

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
            period_ranges=plan.period_ranges,
            limit=plan.limit,
            order_by=plan.order_by,
        )
        debug_payload.update({"sql": sql, "params": params, "row_count": len(rows)})

        if not rows:
            rows, assumptions, debug_payload = await self._fallback_query(
                query=query,
                intent=intent,
                plan=plan,
                assumptions=assumptions,
                debug_payload=debug_payload,
            )

        ambiguous = False
        if rows and len(rows) > self._semantic_rerank_threshold:
            rows, rerank_debug = await self._semantic_rerank_rows(query, rows)
            ambiguous = bool(rerank_debug.get("ambiguous"))
            debug_payload["semantic_rerank"] = rerank_debug

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
                period_ranges=plan.period_ranges,
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
                period_ranges=plan.period_ranges,
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
                    period_ranges=plan.period_ranges,
                    limit=plan.limit,
                    order_by=plan.order_by,
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
