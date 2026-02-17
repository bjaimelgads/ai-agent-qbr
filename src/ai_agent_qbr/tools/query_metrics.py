"""Structured metric query tool."""

from __future__ import annotations

import asyncio
import inspect
import os
from dataclasses import dataclass
from datetime import date
from typing import Any

from penguiflow.catalog import tool
from penguiflow.planner import ToolContext

from ai_agent_qbr.models import MetricQueryArgs
from ai_agent_qbr.tools.question_normalization import normalize_question_arg
from ai_agent_qbr.tools.status import ToolStatusEmitter
from qbr_intelligence.metric_qa import MetricQueryEngine, RagSlideRange
from qbr_intelligence.metric_qa.planner import build_plan
from qbr_intelligence.schemas.metric_qa import (
    AnswerCitation,
    MetricAnswer,
    MetricEvidenceAnswer,
    MetricEvidenceChunk,
    MetricEvidenceIntent,
    MetricEvidenceLink,
    MetricEvidenceMetric,
    MetricEvidenceSource,
    PeriodSpec,
    QueryIntent,
)

import logging
import time

_LOGGER = logging.getLogger("uvicorn.error")


def _ctx_or_env_int(ctx: ToolContext, key: str, env_key: str, default: int) -> int:
    value = ctx.tool_context.get(key)
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    raw = os.getenv(env_key)
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            pass
    return default


@dataclass(frozen=True)
class _RagScope:
    document_ids: list[int]
    slide_ids: list[int]
    slide_ranges: list[RagSlideRange]
    debug: dict[str, object]


def _is_overall_baseline_type(value: str | None) -> bool:
    lowered = (value or "").strip().lower()
    if not lowered:
        return False
    return lowered == "overall"


async def _load_overall_metric_rows(engine: MetricQueryEngine, plan) -> list[Any]:
    rows, _, _ = await engine._store.query_facts(
        metric_ids=plan.metric_ids,
        client_name=plan.client,
        region=plan.region,
        period_ranges=plan.period_ranges,
        limit=max(int(plan.limit), 400),
        order_by="period_end DESC",
    )
    return [row for row in rows if _is_overall_baseline_type(getattr(row, "baseline_type", None))]


async def _load_overall_metric_rows_with_fallback(engine: MetricQueryEngine, plan) -> tuple[list[Any], dict[str, Any]]:
    limit = max(int(plan.limit), 400)
    attempts: list[dict[str, Any]] = []

    strict_rows, strict_sql, strict_params = await engine._store.query_facts(
        metric_ids=plan.metric_ids,
        client_name=plan.client,
        region=plan.region,
        period_ranges=plan.period_ranges,
        limit=limit,
        order_by="period_end DESC",
    )
    strict_overall = [row for row in strict_rows if _is_overall_baseline_type(getattr(row, "baseline_type", None))]
    attempts.append(
        {
            "mode": "strict",
            "sql": strict_sql,
            "params": strict_params,
            "overall_count": len(strict_overall),
        }
    )
    if strict_overall:
        return strict_overall, {"mode": "strict", "attempts": attempts}

    relaxed_rows, relaxed_sql, relaxed_params = await engine._store.query_facts(
        metric_ids=plan.metric_ids,
        client_name=plan.client,
        region=None,
        period_ranges=None,
        limit=limit,
        order_by="period_end DESC",
    )
    relaxed_overall = [
        row for row in relaxed_rows if _is_overall_baseline_type(getattr(row, "baseline_type", None))
    ]
    attempts.append(
        {
            "mode": "client_metric",
            "sql": relaxed_sql,
            "params": relaxed_params,
            "overall_count": len(relaxed_overall),
        }
    )
    if relaxed_overall:
        return relaxed_overall, {"mode": "client_metric", "attempts": attempts}

    metric_only_rows, metric_only_sql, metric_only_params = await engine._store.query_facts(
        metric_ids=plan.metric_ids,
        client_name=None,
        region=None,
        period_ranges=None,
        limit=limit,
        order_by="period_end DESC",
    )
    metric_only_overall = [
        row for row in metric_only_rows if _is_overall_baseline_type(getattr(row, "baseline_type", None))
    ]
    attempts.append(
        {
            "mode": "metric_only",
            "sql": metric_only_sql,
            "params": metric_only_params,
            "overall_count": len(metric_only_overall),
        }
    )
    return metric_only_overall, {"mode": "metric_only" if metric_only_overall else "none", "attempts": attempts}


def _rows_to_slide_ranges(rows: list[Any]) -> list[RagSlideRange]:
    ranges: list[RagSlideRange] = []
    seen: set[tuple[int, int | None, int | None]] = set()
    for row in rows:
        doc_id = int(getattr(row, "document_id", 0) or 0)
        if not doc_id:
            continue
        slide_number = getattr(row, "slide_number", None)
        key = (
            doc_id,
            int(slide_number) if slide_number is not None else None,
            int(slide_number) if slide_number is not None else None,
        )
        if key in seen:
            continue
        seen.add(key)
        ranges.append(
            RagSlideRange(
                document_id=doc_id,
                start_slide=key[1],
                end_slide=key[2],
            )
        )
    return ranges


def _build_metric_slide_index(rows: list[Any]) -> dict[int, set[int]]:
    index: dict[int, set[int]] = {}
    for row in rows:
        doc_id = int(getattr(row, "document_id", 0) or 0)
        slide_number = getattr(row, "slide_number", None)
        if not doc_id or slide_number is None:
            continue
        index.setdefault(doc_id, set()).add(int(slide_number))
    return index


def _merge_slide_indexes(*indexes: dict[int, set[int]]) -> dict[int, set[int]]:
    merged: dict[int, set[int]] = {}
    for index in indexes:
        for doc_id, slides in index.items():
            merged.setdefault(doc_id, set()).update(slides)
    return merged


def _chunk_overlaps_metric_slides(item: Any, metric_slide_index: dict[int, set[int]]) -> bool:
    chunk = getattr(item, "chunk", None)
    if chunk is None:
        return False
    doc_obj = getattr(chunk, "document_id", None)
    doc_id = int(getattr(doc_obj, "value", 0) or 0)
    if not doc_id:
        return False
    metric_slides = metric_slide_index.get(doc_id)
    if not metric_slides:
        return False
    start_slide = getattr(chunk, "start_slide", None)
    end_slide = getattr(chunk, "end_slide", None)
    if start_slide is None and end_slide is None:
        return False
    if start_slide is None:
        return any(slide <= int(end_slide) for slide in metric_slides)
    if end_slide is None:
        return any(slide >= int(start_slide) for slide in metric_slides)
    low = min(int(start_slide), int(end_slide))
    high = max(int(start_slide), int(end_slide))
    return any(low <= slide <= high for slide in metric_slides)


async def _build_rag_scope_for_metric_query(
    *,
    query: str,
    intent: QueryIntent,
    ctx: ToolContext,
    engine: MetricQueryEngine,
) -> _RagScope:
    use_case = ctx.tool_context.get("qbr_search_use_case")
    if use_case is None or not hasattr(use_case, "execute") or not hasattr(use_case, "repository"):
        return _RagScope(
            document_ids=[],
            slide_ids=[],
            slide_ranges=[],
            debug={"enabled": False, "reason": "no_search_use_case"},
        )

    top_k = int(ctx.tool_context.get("retrieval_top_k", 5))
    min_score = ctx.tool_context.get("retrieval_min_score")
    plan = build_plan(intent)
    doc_limit = int(ctx.tool_context.get("metric_prefilter_doc_limit", 25))
    filtered_doc_ids, doc_sql, doc_params = await engine._store.query_document_ids(
        metric_ids=plan.metric_ids,
        client_name=plan.client,
        region=plan.region,
        period_ranges=plan.period_ranges,
        period_specs=[
            period.model_dump(mode="json") if hasattr(period, "model_dump") else dict(period)
            for period in (intent.period or [])
        ],
        limit=doc_limit,
    )
    metric_rows, metric_sql, metric_params = await engine._store.query_facts(
        metric_ids=plan.metric_ids,
        client_name=plan.client,
        region=plan.region,
        period_ranges=plan.period_ranges,
        limit=max(int(plan.limit), 800),
        order_by="period_end DESC",
    )
    metric_slide_index = _build_metric_slide_index(metric_rows)
    metric_scoped_doc_ids = sorted(metric_slide_index.keys())
    metric_scoped_slide_ids = sorted(
        {int(getattr(row, "slide_id", 0) or 0) for row in metric_rows if getattr(row, "slide_id", None)}
    )
    metric_scoped_slide_ranges = _rows_to_slide_ranges(metric_rows)
    if metric_scoped_doc_ids:
        allowed_docs = set(metric_scoped_doc_ids)
        filtered_doc_ids = [doc_id for doc_id in filtered_doc_ids if doc_id in allowed_docs]
        if not filtered_doc_ids:
            filtered_doc_ids = metric_scoped_doc_ids[:]
    overall_rows, overall_debug = await _load_overall_metric_rows_with_fallback(engine, plan)
    overall_doc_ids = sorted(
        {int(getattr(row, "document_id", 0) or 0) for row in overall_rows if getattr(row, "document_id", None)}
    )
    overall_slide_ids = sorted(
        {int(getattr(row, "slide_id", 0) or 0) for row in overall_rows if getattr(row, "slide_id", None)}
    )
    overall_slide_ranges = _rows_to_slide_ranges(overall_rows)
    overall_slide_index = _build_metric_slide_index(overall_rows)
    combined_slide_index = _merge_slide_indexes(metric_slide_index, overall_slide_index)
    retrieval_doc_ids = sorted(set(filtered_doc_ids) | set(overall_doc_ids))
    filtered_doc_names = await engine._store.query_document_names(document_ids=filtered_doc_ids)
    retrieval_doc_names = await engine._store.query_document_names(document_ids=retrieval_doc_ids)
    prefiltered_docs = [
        {"document_id": doc_id, "document_name": filtered_doc_names.get(doc_id)}
        for doc_id in filtered_doc_ids
    ]
    retrieval_docs = [
        {"document_id": doc_id, "document_name": retrieval_doc_names.get(doc_id)}
        for doc_id in retrieval_doc_ids
    ]
    if not retrieval_doc_ids:
        return _RagScope(
            document_ids=[],
            slide_ids=[],
            slide_ranges=[],
            debug={
                "enabled": True,
                "prefiltered_doc_ids": [],
                "prefiltered_docs": [],
                "prefilter_sql": doc_sql,
                "prefilter_params": doc_params,
                "metric_scope_sql": metric_sql,
                "metric_scope_params": metric_params,
                "metric_scope_doc_ids": metric_scoped_doc_ids,
                "metric_scope_slide_ids_count": len(metric_scoped_slide_ids),
                "metric_scope_slide_ranges_count": len(metric_scoped_slide_ranges),
                "overall_scope_mode": overall_debug.get("mode"),
                "overall_scope_attempts": overall_debug.get("attempts"),
                "appended_overall_row_count": len(overall_rows),
                "appended_overall_doc_ids": overall_doc_ids,
                "appended_overall_slide_ids_count": len(overall_slide_ids),
                "appended_overall_slide_ranges_count": len(overall_slide_ranges),
                "retrieval_doc_ids": retrieval_doc_ids,
                "retrieval_docs": retrieval_docs,
                "retrieved_hits": 0,
            },
        )

    base_per_doc_k = _ctx_or_env_int(
        ctx,
        "metric_prefilter_per_doc_k",
        "METRIC_PREFILTER_PER_DOC_K",
        max(2, top_k),
    )
    candidate_total_limit = _ctx_or_env_int(
        ctx,
        "metric_prefilter_candidate_total_limit",
        "METRIC_PREFILTER_CANDIDATE_TOTAL_LIMIT",
        70,
    )
    candidate_single_doc_limit = _ctx_or_env_int(
        ctx,
        "metric_prefilter_candidate_single_doc_limit",
        "METRIC_PREFILTER_CANDIDATE_SINGLE_DOC_LIMIT",
        40,
    )
    candidate_min_per_doc = _ctx_or_env_int(
        ctx,
        "metric_prefilter_candidate_min_per_doc",
        "METRIC_PREFILTER_CANDIDATE_MIN_PER_DOC",
        6,
    )
    final_per_doc_limit = _ctx_or_env_int(
        ctx,
        "metric_prefilter_final_per_doc_limit",
        "METRIC_PREFILTER_FINAL_PER_DOC_LIMIT",
        10,
    )
    final_total_limit = _ctx_or_env_int(
        ctx,
        "metric_prefilter_final_total_limit",
        "METRIC_PREFILTER_FINAL_TOTAL_LIMIT",
        40,
    )
    doc_count = max(1, len(retrieval_doc_ids))
    progressive_candidate_k = max(candidate_min_per_doc, candidate_total_limit // doc_count)
    if doc_count == 1:
        progressive_candidate_k = min(candidate_single_doc_limit, candidate_total_limit)
    else:
        progressive_candidate_k = min(candidate_single_doc_limit, progressive_candidate_k)
    per_doc_k = max(base_per_doc_k, progressive_candidate_k)
    execute_fn = getattr(use_case, "execute", None)
    supports_doc_cap = False
    execute_signature = "unknown"
    execute_qualname = "unknown"
    if callable(execute_fn):
        execute_qualname = getattr(execute_fn, "__qualname__", getattr(execute_fn, "__name__", "unknown"))
        try:
            execute_signature = str(inspect.signature(execute_fn))
            supports_doc_cap = "max_chunks_per_doc" in inspect.signature(execute_fn).parameters
        except Exception:
            supports_doc_cap = False
            execute_signature = "unavailable"
    _LOGGER.info(
        "METRIC_QA_PREFILTER_COMPAT start supports_doc_cap=%s execute=%s signature=%s",
        supports_doc_cap,
        execute_qualname,
        execute_signature,
    )

    async def _run_prefilter_for_doc(doc_id: int):
        kwargs = {
            "query": query,
            "document_id": doc_id,
            "top_k": per_doc_k,
            "min_score": min_score,
        }
        if supports_doc_cap:
            # Keep a wide candidate set from retrieval; final score-based caps are applied below.
            kwargs["max_chunks_per_doc"] = per_doc_k
        _LOGGER.info(
            "METRIC_QA_PREFILTER_COMPAT call doc_id=%s kwargs_keys=%s",
            doc_id,
            sorted(kwargs.keys()),
        )
        try:
            return await use_case.execute(**kwargs)
        except (TypeError, NameError) as exc:
            _LOGGER.warning(
                "METRIC_QA_PREFILTER_COMPAT fallback doc_id=%s error=%s execute=%s signature=%s",
                doc_id,
                repr(exc),
                execute_qualname,
                execute_signature,
            )
            kwargs.pop("max_chunks_per_doc", None)
            _LOGGER.info(
                "METRIC_QA_PREFILTER_COMPAT retry_without_doc_cap doc_id=%s kwargs_keys=%s",
                doc_id,
                sorted(kwargs.keys()),
            )
            try:
                return await use_case.execute(**kwargs)
            except Exception as retry_exc:  # noqa: BLE001
                _LOGGER.exception(
                    "METRIC_QA_PREFILTER_COMPAT retry_failed doc_id=%s error=%s",
                    doc_id,
                    repr(retry_exc),
                )
                raise

    per_doc_hits = await asyncio.gather(
        *(
            _run_prefilter_for_doc(doc_id)
            for doc_id in retrieval_doc_ids
        )
    )
    raw_hits = [item for group in per_doc_hits for item in group]
    pre_slide_scope_hit_count = len(raw_hits)
    if combined_slide_index:
        raw_hits = [item for item in raw_hits if _chunk_overlaps_metric_slides(item, combined_slide_index)]
    if raw_hits:
        ranked_hits = sorted(
            raw_hits,
            key=lambda item: float(
                item.score.value if hasattr(item.score, "value") else item.score or 0.0
            ),
            reverse=True,
        )
        per_doc_counts: dict[int, int] = {}
        per_doc_limited_hits = []
        for item in ranked_hits:
            doc_id = item.chunk.document_id.value
            if per_doc_counts.get(doc_id, 0) >= final_per_doc_limit:
                continue
            per_doc_counts[doc_id] = per_doc_counts.get(doc_id, 0) + 1
            per_doc_limited_hits.append(item)
        hits = per_doc_limited_hits[:final_total_limit]
    else:
        hits = []
    hit_chunks = [
        {
            "chunk_id": item.chunk.chunk_id.value,
            "document_id": item.chunk.document_id.value,
            "start_slide": item.chunk.start_slide,
            "end_slide": item.chunk.end_slide,
            "score": float(item.score.value if hasattr(item.score, "value") else item.score or 0.0),
            "content": item.chunk.content,
            "summary": item.chunk.summary,
            "topics": item.chunk.topics,
            "importance_score": item.chunk.importance_score,
            "metadata": item.chunk.metadata or {},
            "matched_metric_slide": _chunk_overlaps_metric_slides(item, metric_slide_index),
            "matched_overall_slide": _chunk_overlaps_metric_slides(item, overall_slide_index),
        }
        for item in hits
    ]
    per_doc_hit_count: dict[int, int] = {}
    per_doc_max_score: dict[int, float] = {}
    for item in hits:
        doc_id = item.chunk.document_id.value
        per_doc_hit_count[doc_id] = per_doc_hit_count.get(doc_id, 0) + 1
        score = float(item.score.value if hasattr(item.score, "value") else item.score or 0.0)
        prev = per_doc_max_score.get(doc_id)
        if prev is None or score > prev:
            per_doc_max_score[doc_id] = score
    if not hits:
        combined_doc_ids = sorted(set(retrieval_doc_ids) | set(metric_scoped_doc_ids))
        combined_slide_ids = sorted(set(metric_scoped_slide_ids) | set(overall_slide_ids))
        range_keyed: dict[tuple[int, int | None, int | None], RagSlideRange] = {}
        for item in metric_scoped_slide_ranges + overall_slide_ranges:
            key = (int(item.document_id), item.start_slide, item.end_slide)
            range_keyed[key] = item
        combined_slide_ranges = list(range_keyed.values())
        return _RagScope(
            document_ids=combined_doc_ids,
            slide_ids=combined_slide_ids,
            slide_ranges=combined_slide_ranges,
            debug={
                "enabled": True,
                "prefiltered_doc_ids": filtered_doc_ids,
                "prefiltered_docs": prefiltered_docs,
                "retrieval_doc_ids": retrieval_doc_ids,
                "retrieval_docs": retrieval_docs,
                "prefilter_sql": doc_sql,
                "prefilter_params": doc_params,
                "metric_scope_sql": metric_sql,
                "metric_scope_params": metric_params,
                "metric_scope_doc_ids": metric_scoped_doc_ids,
                "metric_scope_slide_ids_count": len(metric_scoped_slide_ids),
                "metric_scope_slide_ranges_count": len(metric_scoped_slide_ranges),
                "retrieved_hits": 0,
                "appended_overall_row_count": len(overall_rows),
                "appended_overall_doc_ids": overall_doc_ids,
                "appended_overall_slide_ids_count": len(overall_slide_ids),
                "appended_overall_slide_ranges_count": len(overall_slide_ranges),
                "overall_scope_mode": overall_debug.get("mode"),
                "overall_scope_attempts": overall_debug.get("attempts"),
                "post_rerank_hit_count": 0,
                "per_doc_hit_count": {},
                "per_doc_max_score": {},
                "retrieved_hit_chunks": [],
                "post_rerank_chunks_full": [],
                "prefilter_raw_retrieved_hits": pre_slide_scope_hit_count,
                "metric_slide_filtered_hit_count": len(raw_hits),
                "pre_rerank_hit_count": len(raw_hits),
                "prefilter_candidate_total_limit": candidate_total_limit,
                "prefilter_candidate_single_doc_limit": candidate_single_doc_limit,
                "prefilter_candidate_min_per_doc": candidate_min_per_doc,
                "prefilter_progressive_candidate_k": progressive_candidate_k,
                "prefilter_final_per_doc_limit": final_per_doc_limit,
                "prefilter_final_total_limit": final_total_limit,
                "prefilter_per_doc_k": per_doc_k,
                "prefilter_supports_dynamic_doc_cap": supports_doc_cap,
            },
        )

    selected_doc_ids: set[int] = set()
    slide_ranges: list[RagSlideRange] = []
    seen_ranges: set[tuple[int, int | None, int | None]] = set()
    for item in hits:
        doc_id = item.chunk.document_id.value
        selected_doc_ids.add(doc_id)
        key = (doc_id, item.chunk.start_slide, item.chunk.end_slide)
        if key in seen_ranges:
            continue
        seen_ranges.add(key)
        slide_ranges.append(
            RagSlideRange(
                document_id=doc_id,
                start_slide=item.chunk.start_slide,
                end_slide=item.chunk.end_slide,
            )
        )

    slide_ids, slide_sql, slide_params = await engine._store.query_slide_ids_for_ranges(
        ranges=[
            (item.document_id, item.start_slide, item.end_slide)
            for item in slide_ranges
        ],
        limit=int(ctx.tool_context.get("metric_prefilter_slide_limit", 5000)),
    )
    selected_doc_list = sorted(selected_doc_ids)
    combined_doc_list = sorted(set(selected_doc_list) | set(overall_doc_ids))
    combined_slide_ids = sorted(set(slide_ids) | set(overall_slide_ids))
    range_keyed: dict[tuple[int, int | None, int | None], RagSlideRange] = {}
    for item in slide_ranges + overall_slide_ranges:
        key = (int(item.document_id), item.start_slide, item.end_slide)
        range_keyed[key] = item
    combined_slide_ranges = list(range_keyed.values())
    selected_doc_names = await engine._store.query_document_names(document_ids=selected_doc_list)
    selected_docs = [
        {
            "document_id": doc_id,
            "document_name": selected_doc_names.get(doc_id),
            "hit_count": per_doc_hit_count.get(doc_id, 0),
            "max_score": per_doc_max_score.get(doc_id),
        }
        for doc_id in selected_doc_list
    ]

    return _RagScope(
        document_ids=combined_doc_list,
        slide_ids=combined_slide_ids,
        slide_ranges=combined_slide_ranges,
        debug={
            "enabled": True,
            "prefiltered_doc_ids": filtered_doc_ids,
            "prefiltered_docs": prefiltered_docs,
            "prefilter_sql": doc_sql,
            "prefilter_params": doc_params,
            "metric_scope_sql": metric_sql,
            "metric_scope_params": metric_params,
            "metric_scope_doc_ids": metric_scoped_doc_ids,
            "metric_scope_slide_ids_count": len(metric_scoped_slide_ids),
            "metric_scope_slide_ranges_count": len(metric_scoped_slide_ranges),
            "retrieved_hits": len(hits),
            "post_rerank_hit_count": len(hits),
            "document_ids": combined_doc_list,
            "selected_docs": selected_docs,
            "appended_overall_row_count": len(overall_rows),
            "appended_overall_doc_ids": overall_doc_ids,
            "appended_overall_slide_ids_count": len(overall_slide_ids),
            "appended_overall_slide_ranges_count": len(overall_slide_ranges),
            "overall_scope_mode": overall_debug.get("mode"),
            "overall_scope_attempts": overall_debug.get("attempts"),
            "per_doc_hit_count": per_doc_hit_count,
            "per_doc_max_score": per_doc_max_score,
            "retrieved_hit_chunks": hit_chunks,
            "post_rerank_chunks_full": hit_chunks,
            "prefilter_raw_retrieved_hits": pre_slide_scope_hit_count,
            "metric_slide_filtered_hit_count": len(raw_hits),
            "pre_rerank_hit_count": len(raw_hits),
            "slide_ids_count": len(combined_slide_ids),
            "slide_id_sql": slide_sql,
            "slide_id_params": slide_params,
            "slide_range_count": len(combined_slide_ranges),
            "prefilter_candidate_total_limit": candidate_total_limit,
            "prefilter_candidate_single_doc_limit": candidate_single_doc_limit,
            "prefilter_candidate_min_per_doc": candidate_min_per_doc,
            "prefilter_progressive_candidate_k": progressive_candidate_k,
            "prefilter_final_per_doc_limit": final_per_doc_limit,
            "prefilter_final_total_limit": final_total_limit,
            "prefilter_per_doc_k": per_doc_k,
            "prefilter_supports_dynamic_doc_cap": supports_doc_cap,
        },
    )


def _coerce_period_specs(periods: Any) -> list[PeriodSpec]:
    out: list[PeriodSpec] = []
    if not isinstance(periods, list):
        return out
    for item in periods:
        if isinstance(item, PeriodSpec):
            out.append(item)
            continue
        if isinstance(item, dict):
            start = item.get("start")
            end = item.get("end")
            if isinstance(start, str):
                start = date.fromisoformat(start)
            if isinstance(end, str):
                end = date.fromisoformat(end)
            out.append(
                PeriodSpec(
                    type=item.get("type"),
                    value=item.get("value"),
                    start=start,
                    end=end,
                )
            )
    return out


def _intent_from_interaction_metadata(metadata: dict[str, Any] | None) -> QueryIntent | None:
    if not isinstance(metadata, dict):
        return None
    payload = metadata.get("refined_metric_intent") or metadata.get("resolved_metric_intent")
    if not isinstance(payload, dict):
        return None
    period_specs = _coerce_period_specs(payload.get("period"))
    return QueryIntent(
        metric_ids=list(payload.get("metric_ids") or []),
        client=list(payload.get("client") or []),
        region=list(payload.get("region") or []),
        period=period_specs,
        aggregation=payload.get("aggregation"),
        grouping=payload.get("grouping"),
        limit=payload.get("limit"),
    )


def _intent_from_args(payload: Any) -> QueryIntent | None:
    if payload is None:
        return None
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    if not isinstance(payload, dict):
        return None
    period_specs = _coerce_period_specs(payload.get("period"))
    return QueryIntent(
        metric_ids=list(payload.get("metric_ids") or []),
        client=list(payload.get("client") or []),
        region=list(payload.get("region") or []),
        period=period_specs,
        aggregation=payload.get("aggregation"),
        grouping=payload.get("grouping"),
        limit=payload.get("limit"),
    )


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _truncate_text(value: Any, *, max_chars: int) -> str | None:
    if not isinstance(value, str):
        return None
    compact = " ".join(value.split())
    if not compact:
        return None
    if len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 1)] + "…"


def _slide_url_index(answer: MetricAnswer) -> dict[tuple[int, int | None], str]:
    index: dict[tuple[int, int | None], str] = {}
    for citation in answer.citations:
        doc_id = _to_int(getattr(citation, "document_id", None))
        slide_number = _to_int(getattr(citation, "slide_number", None))
        if doc_id is None:
            continue
        slide_url = getattr(citation, "slide_url", None) or getattr(citation, "document_url", None)
        if isinstance(slide_url, str) and slide_url:
            index[(doc_id, slide_number)] = slide_url
    if isinstance(answer.table_data, list):
        for row in answer.table_data:
            if not isinstance(row, dict):
                continue
            doc_id = _to_int(row.get("document_id"))
            slide_number = _to_int(row.get("slide_number"))
            if doc_id is None:
                continue
            slide_url = row.get("slide_url") or row.get("document_url")
            if isinstance(slide_url, str) and slide_url and (doc_id, slide_number) not in index:
                index[(doc_id, slide_number)] = slide_url
    return index


def _cited_slide_pairs(answer: MetricAnswer) -> set[tuple[int, int | None]]:
    pairs: set[tuple[int, int | None]] = set()
    for citation in answer.citations:
        doc_id = _to_int(getattr(citation, "document_id", None))
        if doc_id is None:
            continue
        slide_number = _to_int(getattr(citation, "slide_number", None))
        pairs.add((doc_id, slide_number))
    if isinstance(answer.table_data, list):
        for row in answer.table_data:
            if not isinstance(row, dict):
                continue
            doc_id = _to_int(row.get("document_id"))
            if doc_id is None:
                continue
            slide_number = _to_int(row.get("slide_number"))
            pairs.add((doc_id, slide_number))
    return pairs


def _build_cited_slide_context_chunks(
    answer: MetricAnswer,
    rag_scope: dict[str, Any],
    *,
    max_total: int,
    max_per_slide: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    source = rag_scope.get("post_rerank_chunks_full")
    if not isinstance(source, list):
        source = rag_scope.get("retrieved_hit_chunks")
    if not isinstance(source, list):
        return []
    cited_pairs = _cited_slide_pairs(answer)
    if not cited_pairs:
        return []
    url_index = _slide_url_index(answer)
    per_slide_count: dict[tuple[int, int | None], int] = {}
    out: list[dict[str, Any]] = []
    for item in source:
        if not isinstance(item, dict):
            continue
        doc_id = _to_int(item.get("document_id"))
        if doc_id is None:
            continue
        metadata = item.get("metadata")
        slide_number = None
        if isinstance(metadata, dict):
            slide_number = _to_int(metadata.get("slide_number"))
        if slide_number is None:
            slide_number = _to_int(item.get("start_slide"))
        key = (doc_id, slide_number)
        if key not in cited_pairs:
            continue
        if per_slide_count.get(key, 0) >= max_per_slide:
            continue
        content = _truncate_text(item.get("content"), max_chars=max_chars)
        if content is None:
            content = _truncate_text(item.get("summary"), max_chars=max_chars)
        out.append(
            {
                "chunk_id": _to_int(item.get("chunk_id")),
                "document_id": doc_id,
                "slide_number": slide_number,
                "slide_title": metadata.get("slide_title") if isinstance(metadata, dict) else None,
                "slide_url": url_index.get(key) or url_index.get((doc_id, None)),
                "score": item.get("score"),
                "content": content,
            }
        )
        per_slide_count[key] = per_slide_count.get(key, 0) + 1
        if len(out) >= max_total:
            break
    return out


def _compact_rag_scope_for_planner(
    rag_scope: dict[str, Any],
) -> dict[str, Any]:
    compact_selected_docs = []
    selected_docs = rag_scope.get("selected_docs")
    if isinstance(selected_docs, list):
        for item in selected_docs:
            if not isinstance(item, dict):
                continue
            compact_selected_docs.append(
                {
                    "document_id": item.get("document_id"),
                    "document_name": item.get("document_name"),
                    "hit_count": item.get("hit_count"),
                    "max_score": item.get("max_score"),
                }
            )
    return {
        "enabled": bool(rag_scope.get("enabled", False)),
        "prefiltered_doc_ids": rag_scope.get("prefiltered_doc_ids"),
        "document_ids": rag_scope.get("document_ids"),
        "selected_docs": compact_selected_docs,
        "retrieved_hits": rag_scope.get("retrieved_hits"),
        "post_rerank_hit_count": rag_scope.get("post_rerank_hit_count"),
        "slide_ids_count": rag_scope.get("slide_ids_count"),
        "slide_range_count": rag_scope.get("slide_range_count"),
    }


def _compact_planner_table_data(rows: list[dict[str, Any]] | None, *, max_rows: int) -> list[dict[str, Any]] | None:
    if not isinstance(rows, list):
        return None
    compact: list[dict[str, Any]] = []
    for row in rows[:max_rows]:
        if not isinstance(row, dict):
            continue
        compact.append(
            {
                "metric": row.get("metric"),
                "value": row.get("value"),
                "unit": row.get("unit"),
                "period": row.get("period"),
                "client": row.get("client"),
                "region": row.get("region"),
                "llm_context_label": row.get("llm_context_label"),
                "document_name": row.get("document_name"),
                "document_url": row.get("document_url"),
                "slide_number": row.get("slide_number"),
                "slide_title": row.get("slide_title"),
                "slide_url": row.get("slide_url") or row.get("document_url"),
                "snippet": _truncate_text(row.get("snippet"), max_chars=180),
            }
        )
    return compact


def _compact_planner_citations(citations: list[Any], *, max_items: int) -> list[AnswerCitation]:
    compact: list[AnswerCitation] = []
    for citation in citations[:max_items]:
        compact.append(
            AnswerCitation(
                document_id=getattr(citation, "document_id", None),
                document_name=getattr(citation, "document_name", None),
                document_url=getattr(citation, "document_url", None),
                slide_id=getattr(citation, "slide_id", None),
                slide_number=getattr(citation, "slide_number", None),
                slide_title=getattr(citation, "slide_title", None),
                slide_google_id=getattr(citation, "slide_google_id", None),
                slide_url=getattr(citation, "slide_url", None) or getattr(citation, "document_url", None),
                snippet=_truncate_text(getattr(citation, "snippet", None), max_chars=180),
            )
        )
    return compact


def _planner_safe_answer(answer: MetricAnswer, ctx: ToolContext) -> MetricAnswer:
    debug_payload = answer.debug if isinstance(answer.debug, dict) else None
    if debug_payload is None:
        return answer
    planner_answer = answer.model_copy(deep=True)
    planner_debug = dict(planner_answer.debug) if isinstance(planner_answer.debug, dict) else {}
    rag_scope = planner_debug.get("rag_scope")
    if isinstance(rag_scope, dict):
        planner_debug["rag_scope"] = _compact_rag_scope_for_planner(rag_scope)
    table_rows_max = _ctx_or_env_int(
        ctx,
        "metric_planner_table_rows_max",
        "METRIC_PLANNER_TABLE_ROWS_MAX",
        12,
    )
    citation_max = _ctx_or_env_int(
        ctx,
        "metric_planner_citations_max",
        "METRIC_PLANNER_CITATIONS_MAX",
        12,
    )
    planner_answer.table_data = _compact_planner_table_data(planner_answer.table_data, max_rows=table_rows_max)
    planner_answer.citations = _compact_planner_citations(planner_answer.citations, max_items=citation_max)
    planner_answer.data = None
    planner_answer.debug = planner_debug
    return planner_answer


def _answer_rows(answer: MetricAnswer) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(answer.table_data, list):
        for row in answer.table_data:
            if isinstance(row, dict):
                rows.append(dict(row))
    if not rows and isinstance(answer.data, list):
        for row in answer.data:
            if hasattr(row, "model_dump"):
                rows.append(row.model_dump(mode="json"))
            elif isinstance(row, dict):
                rows.append(dict(row))
    return rows


def _link_metrics_to_chunks(
    metrics: list[MetricEvidenceMetric],
    chunks: list[MetricEvidenceChunk],
) -> list[MetricEvidenceLink]:
    links: list[MetricEvidenceLink] = []
    for metric_idx, metric in enumerate(metrics):
        m_doc = metric.source.document_id
        m_slide = metric.source.slide_number
        if m_doc is None or m_slide is None:
            continue
        for chunk_idx, chunk in enumerate(chunks):
            if chunk.document_id != m_doc:
                continue
            if chunk.start_slide is None and chunk.end_slide is None:
                continue
            if chunk.start_slide is None:
                if m_slide <= int(chunk.end_slide):
                    links.append(
                        MetricEvidenceLink(
                            metric_index=metric_idx,
                            chunk_index=chunk_idx,
                            overlap_type="range_overlap",
                            link_confidence=1.0,
                        )
                    )
                continue
            if chunk.end_slide is None:
                if m_slide >= int(chunk.start_slide):
                    links.append(
                        MetricEvidenceLink(
                            metric_index=metric_idx,
                            chunk_index=chunk_idx,
                            overlap_type="range_overlap",
                            link_confidence=1.0,
                        )
                    )
                continue
            low = min(int(chunk.start_slide), int(chunk.end_slide))
            high = max(int(chunk.start_slide), int(chunk.end_slide))
            if low <= m_slide <= high:
                overlap_type = "exact" if low == high == m_slide else "range_overlap"
                links.append(
                    MetricEvidenceLink(
                        metric_index=metric_idx,
                        chunk_index=chunk_idx,
                        overlap_type=overlap_type,
                        link_confidence=1.0,
                    )
                )
    return links


def _build_metric_evidence_answer(
    *,
    query: str,
    intent: QueryIntent,
    answer: MetricAnswer,
    rag_scope_debug: dict[str, Any],
) -> MetricEvidenceAnswer:
    rows = _answer_rows(answer)
    metrics: list[MetricEvidenceMetric] = []
    for row in rows:
        source = MetricEvidenceSource(
            document_id=_to_int(row.get("document_id")),
            document_name=row.get("document_name"),
            document_url=row.get("document_url"),
            slide_id=_to_int(row.get("slide_id")),
            slide_number=_to_int(row.get("slide_number")),
            slide_title=row.get("slide_title"),
            slide_url=row.get("slide_url") or row.get("document_url"),
        )
        metrics.append(
            MetricEvidenceMetric(
                metric_id=row.get("metric_id"),
                metric_name=row.get("metric"),
                value=row.get("value"),
                unit=row.get("unit"),
                period=row.get("period"),
                client=row.get("client"),
                region=row.get("region"),
                context_label=row.get("llm_context_label"),
                confidence=row.get("confidence"),
                source=source,
            )
        )

    chunk_payload = rag_scope_debug.get("post_rerank_chunks_full")
    if not isinstance(chunk_payload, list):
        chunk_payload = rag_scope_debug.get("retrieved_hit_chunks")
    retrieval_chunks: list[MetricEvidenceChunk] = []
    if isinstance(chunk_payload, list):
        doc_name_index: dict[int, str | None] = {}
        if isinstance(rag_scope_debug.get("selected_docs"), list):
            for item in rag_scope_debug.get("selected_docs") or []:
                if isinstance(item, dict):
                    doc_id = _to_int(item.get("document_id"))
                    if doc_id is not None:
                        doc_name_index[doc_id] = item.get("document_name")
        for item in chunk_payload:
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            doc_id = _to_int(item.get("document_id"))
            start_slide = _to_int(item.get("start_slide"))
            end_slide = _to_int(item.get("end_slide"))
            retrieval_chunks.append(
                MetricEvidenceChunk(
                    chunk_id=_to_int(item.get("chunk_id")),
                    document_id=doc_id,
                    document_name=doc_name_index.get(doc_id) if doc_id is not None else None,
                    document_url=metadata.get("document_url"),
                    start_slide=start_slide,
                    end_slide=end_slide,
                    slide_title=metadata.get("slide_title"),
                    slide_url=metadata.get("slide_url") or metadata.get("document_url"),
                    score=item.get("score"),
                    content=_truncate_text(item.get("content"), max_chars=400),
                    matched_metric_slide=bool(item.get("matched_metric_slide", False)),
                )
            )

    links = _link_metrics_to_chunks(metrics, retrieval_chunks)
    period_values = [str(period.value) for period in intent.period if getattr(period, "value", None)]
    retrieval_debug = {
        "prefiltered_doc_ids": rag_scope_debug.get("prefiltered_doc_ids"),
        "document_ids": rag_scope_debug.get("document_ids"),
        "selected_docs": rag_scope_debug.get("selected_docs"),
        "pre_rerank_hit_count": rag_scope_debug.get("pre_rerank_hit_count"),
        "post_rerank_hit_count": rag_scope_debug.get("post_rerank_hit_count"),
        "retrieved_hits": rag_scope_debug.get("retrieved_hits"),
        "slide_ids_count": rag_scope_debug.get("slide_ids_count"),
        "slide_range_count": rag_scope_debug.get("slide_range_count"),
        "metric_scope_doc_ids": rag_scope_debug.get("metric_scope_doc_ids"),
        "metric_scope_slide_ids_count": rag_scope_debug.get("metric_scope_slide_ids_count"),
        "metric_scope_slide_ranges_count": rag_scope_debug.get("metric_scope_slide_ranges_count"),
        "overall_scope_mode": rag_scope_debug.get("overall_scope_mode"),
        "appended_overall_row_count": rag_scope_debug.get("appended_overall_row_count"),
    }

    return MetricEvidenceAnswer(
        query=query,
        intent=MetricEvidenceIntent(
            metric_ids=list(intent.metric_ids),
            client=list(intent.client),
            region=list(intent.region),
            period=period_values,
        ),
        metrics=metrics,
        retrieval_chunks=retrieval_chunks,
        metric_chunk_links=links,
        retrieval_debug=retrieval_debug,
        status="ok" if metrics else "empty",
        error=None,
    )


@tool(
    desc=(
        "Query structured metric facts with deterministic filters. "
        "Call this after resolve_metric_intent (and refine_metric_intent if needed)."
    ),
    side_effects="read",
    tags=["planner"],
)
async def query_metrics(args: MetricQueryArgs, ctx: ToolContext) -> MetricEvidenceAnswer:
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
        return MetricEvidenceAnswer(
            query=canonical_query,
            intent=MetricEvidenceIntent(),
            metrics=[],
            retrieval_chunks=[],
            metric_chunk_links=[],
            retrieval_debug=None,
            status="error",
            error="Metric query engine is not available.",
        )

    _LOGGER.info("Metric query start: %s", canonical_query)
    start = time.perf_counter()
    trace_id = None
    if isinstance(ctx.tool_context, dict):
        trace_id = ctx.tool_context.get("trace_id")
    interaction_metadata = ctx.tool_context.get("interaction_metadata")
    intent = _intent_from_args(args.intent)
    intent_source = "args.intent" if intent is not None else "interaction_metadata"
    if intent is None:
        intent = _intent_from_interaction_metadata(interaction_metadata)
    if intent is None:
        _LOGGER.warning("METRIC_QA_INTENT_SOURCE missing")
        return MetricEvidenceAnswer(
            query=canonical_query,
            intent=MetricEvidenceIntent(),
            metrics=[],
            retrieval_chunks=[],
            metric_chunk_links=[],
            retrieval_debug=None,
            status="error",
            error="Missing refined/resolved metric intent in tool context.",
        )
    _LOGGER.info(
        "METRIC_QA_INTENT_SOURCE source=%s metric_ids=%s client=%s region=%s periods=%s",
        intent_source,
        intent.metric_ids,
        intent.client,
        intent.region,
        [period.value for period in intent.period],
    )
    rag_scope = await _build_rag_scope_for_metric_query(
        query=canonical_query,
        intent=intent,
        ctx=ctx,
        engine=engine,
    )
    mlflow_trace = ctx.tool_context.get("mlflow_trace")
    if mlflow_trace is not None and hasattr(mlflow_trace, "span") and hasattr(mlflow_trace, "set_outputs"):
        with mlflow_trace.span(
            name="query_metrics.rag",
            span_type="RETRIEVER",
            attributes={
                "source_tool": "query_metrics",
                "enabled": bool(rag_scope.debug.get("enabled", False)),
                "direct_tool_emission": True,
            },
            inputs={
                "query": canonical_query,
                "prefiltered_doc_ids": rag_scope.debug.get("prefiltered_doc_ids"),
            },
        ) as rag_span:
            mlflow_trace.set_outputs(
                rag_span,
                {
                    "prefiltered_docs": rag_scope.debug.get("prefiltered_docs"),
                    "document_ids": rag_scope.debug.get("document_ids"),
                    "selected_docs": rag_scope.debug.get("selected_docs"),
                    "per_doc_hit_count": rag_scope.debug.get("per_doc_hit_count"),
                    "per_doc_max_score": rag_scope.debug.get("per_doc_max_score"),
                    "retrieved_hit_chunks": rag_scope.debug.get("retrieved_hit_chunks"),
                    "post_rerank_chunks_full": rag_scope.debug.get("post_rerank_chunks_full"),
                    "slide_ids_count": rag_scope.debug.get("slide_ids_count"),
                    "slide_range_count": rag_scope.debug.get("slide_range_count"),
                    "retrieved_hits": rag_scope.debug.get("retrieved_hits"),
                    "post_rerank_hit_count": rag_scope.debug.get("post_rerank_hit_count"),
                    "pre_rerank_hit_count": rag_scope.debug.get("pre_rerank_hit_count"),
                },
            )
    _LOGGER.info(
        "METRIC_QA_RAG_SCOPE query=%r enabled=%s docs=%s ranges=%s",
        canonical_query,
        rag_scope.debug.get("enabled"),
        rag_scope.debug.get("document_ids"),
        rag_scope.debug.get("slide_range_count"),
    )

    result = await engine.query(
        canonical_query,
        debug=args.debug,
        trace_id=trace_id,
        intent_override=intent,
        strict_entity_filters=True,
        rag_document_ids=rag_scope.document_ids,
        rag_slide_ids=rag_scope.slide_ids,
        rag_slide_ranges=rag_scope.slide_ranges,
        rag_hit_chunks=(
            rag_scope.debug.get("post_rerank_chunks_full")
            if isinstance(rag_scope.debug.get("post_rerank_chunks_full"), list)
            else None
        ),
    )
    answer = result.answer
    citation_before_count = len(answer.citations)
    if intent.region:
        allowed_regions = {str(code).upper() for code in intent.region if code}
        cited_doc_ids = sorted({int(c.document_id) for c in answer.citations if c.document_id is not None})
        doc_regions = await engine._store.query_document_regions(document_ids=cited_doc_ids)
        filtered_citations = [
            citation
            for citation in answer.citations
            if doc_regions.get(int(citation.document_id)) in allowed_regions
        ]
        answer.citations = filtered_citations
    debug_payload = dict(answer.debug) if isinstance(answer.debug, dict) else {}
    debug_payload["rag_scope"] = rag_scope.debug
    if intent.region:
        debug_payload["citation_region_filter"] = {
            "requested_regions": [str(code).upper() for code in intent.region if code],
            "before_count": citation_before_count,
            "after_count": len(answer.citations),
        }
    answer.debug = debug_payload
    planner_answer = _planner_safe_answer(answer, ctx)
    evidence_answer = _build_metric_evidence_answer(
        query=canonical_query,
        intent=intent,
        answer=answer,
        rag_scope_debug=rag_scope.debug,
    )
    if mlflow_trace is not None and hasattr(mlflow_trace, "span") and hasattr(mlflow_trace, "set_outputs"):
        with mlflow_trace.span(
            name="query_metrics.result",
            span_type="TOOL",
            attributes={
                "source_tool": "query_metrics",
                "intent_source": intent_source,
                "result_kind": "full",
            },
            inputs={
                "query": canonical_query,
                "intent": intent.model_dump(mode="json"),
                "debug_requested": bool(args.debug),
            },
        ) as result_span:
            mlflow_trace.set_outputs(
                result_span,
                {
                    "metric_intent": result.intent.model_dump(mode="json"),
                    "metric_answer": answer.model_dump(mode="json"),
                    "metric_answer_row_count": len(answer.data or []),
                    "metric_rag_scope": rag_scope.debug,
                },
            )
        with mlflow_trace.span(
            name="query_metrics.planner_payload",
            span_type="TOOL",
            attributes={
                "source_tool": "query_metrics",
                "intent_source": intent_source,
                "result_kind": "planner_safe",
            },
            inputs={
                "query": canonical_query,
                "intent": intent.model_dump(mode="json"),
            },
        ) as planner_payload_span:
            mlflow_trace.set_outputs(
                planner_payload_span,
                {
                    "metric_answer_planner_safe": planner_answer.model_dump(mode="json"),
                    "metric_evidence_answer": evidence_answer.model_dump(mode="json"),
                    "metric_answer_row_count": len(planner_answer.data or []),
                },
            )
    _LOGGER.info("METRIC_QA_RAW_ANSWER %s", answer.model_dump(mode="json"))
    _LOGGER.info("Metric query done: %.2fs rows=%s", time.perf_counter() - start, len(answer.data or []))
    if isinstance(interaction_metadata, dict):
        # Keep metadata JSON-serializable for planner memory/llm_context round-trips.
        interaction_metadata["metric_intent"] = result.intent.model_dump(mode="json")
        interaction_metadata["metric_answer"] = answer.model_dump(mode="json")
        interaction_metadata["metric_answer_planner_safe"] = planner_answer.model_dump(mode="json")
        interaction_metadata["metric_evidence_answer"] = evidence_answer.model_dump(mode="json")
        interaction_metadata["metric_answer_row_count"] = len(answer.data or [])
        interaction_metadata["metric_rag_scope"] = rag_scope.debug
    return evidence_answer
