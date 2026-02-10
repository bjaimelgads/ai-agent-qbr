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
from qbr_intelligence.schemas.metric_qa import MetricAnswer, PeriodSpec, QueryIntent

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
    filtered_doc_names = await engine._store.query_document_names(document_ids=filtered_doc_ids)
    prefiltered_docs = [
        {"document_id": doc_id, "document_name": filtered_doc_names.get(doc_id)}
        for doc_id in filtered_doc_ids
    ]
    if not filtered_doc_ids:
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
    doc_count = max(1, len(filtered_doc_ids))
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
            for doc_id in filtered_doc_ids
        )
    )
    raw_hits = [item for group in per_doc_hits for item in group]
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
        return _RagScope(
            document_ids=filtered_doc_ids,
            slide_ids=[],
            slide_ranges=[],
            debug={
                "enabled": True,
                "prefiltered_doc_ids": filtered_doc_ids,
                "prefiltered_docs": prefiltered_docs,
                "prefilter_sql": doc_sql,
                "prefilter_params": doc_params,
                "retrieved_hits": 0,
                "post_rerank_hit_count": 0,
                "per_doc_hit_count": {},
                "per_doc_max_score": {},
                "retrieved_hit_chunks": [],
                "post_rerank_chunks_full": [],
                "prefilter_raw_retrieved_hits": len(raw_hits),
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
        document_ids=selected_doc_list,
        slide_ids=slide_ids,
        slide_ranges=slide_ranges,
        debug={
            "enabled": True,
            "prefiltered_doc_ids": filtered_doc_ids,
            "prefiltered_docs": prefiltered_docs,
            "prefilter_sql": doc_sql,
            "prefilter_params": doc_params,
            "retrieved_hits": len(hits),
            "post_rerank_hit_count": len(hits),
            "document_ids": selected_doc_list,
            "selected_docs": selected_docs,
            "per_doc_hit_count": per_doc_hit_count,
            "per_doc_max_score": per_doc_max_score,
            "retrieved_hit_chunks": hit_chunks,
            "post_rerank_chunks_full": hit_chunks,
            "prefilter_raw_retrieved_hits": len(raw_hits),
            "pre_rerank_hit_count": len(raw_hits),
            "slide_ids_count": len(slide_ids),
            "slide_id_sql": slide_sql,
            "slide_id_params": slide_params,
            "slide_range_count": len(slide_ranges),
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


@tool(
    desc=(
        "Query structured metric facts with deterministic filters. "
        "Call this after resolve_metric_intent (and refine_metric_intent if needed)."
    ),
    side_effects="read",
    tags=["planner"],
)
async def query_metrics(args: MetricQueryArgs, ctx: ToolContext) -> MetricAnswer:
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
        return MetricAnswer(
            summary_text="Metric query engine is not available.",
            citations=[],
            confidence=0.0,
            assumptions=["Metric engine missing."],
            followups=["Please try again later."],
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
        return MetricAnswer(
            summary_text=(
                "I need structured metric filters first. "
                "Please run intent resolution before querying metrics."
            ),
            citations=[],
            confidence=0.0,
            assumptions=["Missing refined/resolved metric intent in tool context."],
            followups=["Run resolve_metric_intent (and refine_metric_intent if needed), then retry."],
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
    )
    answer = result.answer
    debug_payload = dict(answer.debug) if isinstance(answer.debug, dict) else {}
    debug_payload["rag_scope"] = rag_scope.debug
    answer.debug = debug_payload
    _LOGGER.info("METRIC_QA_RAW_ANSWER %s", answer.model_dump(mode="json"))
    _LOGGER.info("Metric query done: %.2fs rows=%s", time.perf_counter() - start, len(answer.data or []))
    if isinstance(interaction_metadata, dict):
        # Keep metadata JSON-serializable for planner memory/llm_context round-trips.
        interaction_metadata["metric_intent"] = result.intent.model_dump(mode="json")
        interaction_metadata["metric_answer"] = answer.model_dump(mode="json")
        interaction_metadata["metric_answer_row_count"] = len(answer.data or [])
        interaction_metadata["metric_rag_scope"] = rag_scope.debug
    return answer
