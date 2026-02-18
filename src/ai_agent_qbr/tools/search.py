"""Search tool."""

from __future__ import annotations

import logging

from penguiflow.catalog import tool
from penguiflow.planner import ToolContext

from ai_agent_qbr.infrastructure.region_filter import filter_items_by_region
from ai_agent_qbr.models import Query, SearchResult, SearchResults
from ai_agent_qbr.tools.intent_filters import (
    DocumentFilters,
    build_document_filters_from_query,
    score_document_match,
)
from ai_agent_qbr.tools.question_normalization import normalize_question_arg
from ai_agent_qbr.tools.status import ToolStatusEmitter
from qbr_agent.application.use_cases import HybridSearchKnowledge
from qbr_agent.domain.entities import Document
from qbr_agent.domain.value_objects import DocumentId
from qbr_intelligence.metric_qa import MetricQueryEngine
from qbr_intelligence.metric_qa.planner import build_plan


async def _build_document_filters(
    *,
    query: str,
    ctx: ToolContext,
) -> DocumentFilters:
    engine = ctx.tool_context.get("metric_query_engine")
    if not isinstance(engine, MetricQueryEngine):
        return DocumentFilters(clients=set(), regions=set(), periods=set())
    filters, _intent = await build_document_filters_from_query(query=query, engine=engine)
    return filters


@tool(desc="Search internal QBR knowledge", side_effects="read", tags=["planner"])
async def search_documents(args: Query, ctx: ToolContext) -> SearchResults:
    status = ToolStatusEmitter(ctx, tool_name="search_documents")
    await status.step("Searching QBR materials for relevant details.", step_name="Searching QBR's")
    logger = logging.getLogger("uvicorn.error")
    use_case = ctx.tool_context.get("qbr_search_use_case")
    if not isinstance(use_case, HybridSearchKnowledge):
        return SearchResults(results=[])

    top_k = int(ctx.tool_context.get("retrieval_top_k", 5))
    min_score = ctx.tool_context.get("retrieval_min_score")
    canonical_query = normalize_question_arg(args.question, ctx.tool_context)
    logger.info(
        "SEARCH_REQUEST raw=%r canonical=%r top_k=%s min_score=%s comparison=%s",
        args.question,
        canonical_query,
        top_k,
        min_score,
        bool(args.comparison_intent),
    )

    comparison_intent = bool(args.comparison_intent)
    if comparison_intent:
        await status.step("Comparing across multiple decks.", step_name="Comparings decks")

    prefiltered_doc_ids: list[int] = []
    prefilter_intent = None
    candidate_document_count: int | None = None
    engine = ctx.tool_context.get("metric_query_engine")
    if isinstance(engine, MetricQueryEngine):
        intent, _debug = await engine.resolve_intent(canonical_query)
        prefilter_intent = intent
        plan = build_plan(intent)
        has_filters = bool(plan.client or plan.region or plan.period_ranges)
        if has_filters:
            doc_limit = int(ctx.tool_context.get("search_prefilter_doc_limit", 25))
            prefiltered_doc_ids, _sql, _params = await engine._store.query_document_ids(
                metric_ids=plan.metric_ids,
                client_name=plan.client,
                region=plan.region,
                period_ranges=plan.period_ranges,
                limit=doc_limit,
            )
            logger.info(
                "SEARCH_PREFILTER_DOCS count=%s filters(client=%s region=%s periods=%s)",
                len(prefiltered_doc_ids),
                plan.client,
                plan.region,
                len(plan.period_ranges),
            )
            candidate_document_count = len(prefiltered_doc_ids)
        else:
            clarify_limit = int(ctx.tool_context.get("search_clarify_doc_threshold", 6))
            probe_ids, _sql, _params = await engine._store.query_document_ids(
                metric_ids=plan.metric_ids,
                client_name=plan.client,
                region=plan.region,
                period_ranges=plan.period_ranges,
                limit=clarify_limit + 1,
            )
            candidate_document_count = len(probe_ids)
            if candidate_document_count > clarify_limit:
                suggested_filters: list[str] = []
                if not plan.client:
                    suggested_filters.append("client")
                if not plan.region:
                    suggested_filters.append("region")
                if not plan.period_ranges:
                    suggested_filters.append("period")
                clarification = (
                    "I found many possible QBR documents. "
                    "Can you narrow this by client, region, or period (for example: Disney+ US FY25 H1)?"
                )
                logger.info(
                    "SEARCH_CLARIFICATION required candidate_docs>%s suggested=%s intent=%s",
                    clarify_limit,
                    suggested_filters,
                    prefilter_intent.model_dump(mode="json") if prefilter_intent is not None else None,
                )
                return SearchResults(
                    results=[],
                    needs_clarification=True,
                    clarification_question=clarification,
                    suggested_filters=suggested_filters,
                    candidate_document_count=candidate_document_count,
                )

    if comparison_intent and prefiltered_doc_ids:
        results = await _search_prefiltered_documents(
            query=canonical_query,
            use_case=use_case,
            top_k=top_k,
            min_score=min_score,
            document_ids=prefiltered_doc_ids,
            ctx=ctx,
        )
    elif comparison_intent:
        results = await _search_comparison_documents(
            query=canonical_query,
            use_case=use_case,
            top_k=top_k,
            min_score=min_score,
            ctx=ctx,
        )
    elif prefiltered_doc_ids:
        results = await _search_prefiltered_documents(
            query=canonical_query,
            use_case=use_case,
            top_k=top_k,
            min_score=min_score,
            document_ids=prefiltered_doc_ids,
            ctx=ctx,
        )
    else:
        results = await use_case.execute(
            query=canonical_query,
            top_k=top_k,
            min_score=min_score,
        )

    region_focus = ctx.tool_context.get("region_focus")
    if isinstance(region_focus, str):
        logger.info("SEARCH_REGION_FOCUS %s", region_focus)
        results = filter_items_by_region(results, region_focus)

    docs_by_id = await _fetch_documents_map(use_case, results)
    doc_filters = await _build_document_filters(query=canonical_query, ctx=ctx)
    if doc_filters.enabled() and docs_by_id:
        kept: list[tuple[float, object]] = []
        removed_count = 0
        for item in results:
            doc = docs_by_id.get(item.chunk.document_id.value)
            if doc is None:
                kept.append((item.score.value, item))
                continue
            passes, boost = score_document_match(doc, doc_filters)
            if not passes:
                removed_count += 1
                continue
            # Preserve raw score values; boost is applied only for ranking.
            kept.append((item.score.value + boost, item))
        if kept:
            results = [item for _, item in sorted(kept, key=lambda pair: pair[0], reverse=True)]
        logger.info(
            "SEARCH_DOC_FILTERS clients=%s regions=%s periods=%s removed=%s kept=%s",
            sorted(doc_filters.clients),
            sorted(doc_filters.regions),
            sorted(doc_filters.periods),
            removed_count,
            len(results),
        )

    include_path = bool(ctx.tool_context.get("retrieval_include_document_path", False))
    formatted = await _format_results(
        results,
        use_case,
        include_path,
        documents_by_id=docs_by_id,
    )
    logger.info(
        "SEARCH_RESULTS count=%s top_titles=%s",
        len(formatted.results),
        [item.title for item in formatted.results[:5]],
    )
    if candidate_document_count is not None:
        formatted.candidate_document_count = candidate_document_count
    return formatted


async def _search_comparison_documents(
    *,
    query: str,
    use_case: HybridSearchKnowledge,
    top_k: int,
    min_score: float | None,
    ctx: ToolContext,
) -> list:
    comparison_top_docs = int(ctx.tool_context.get("comparison_top_docs", 3))
    comparison_per_doc_k = int(ctx.tool_context.get("comparison_per_doc_k", 3))
    stage1_override = ctx.tool_context.get("comparison_stage1_top_k")
    comparison_stage1_top_k = (
        int(stage1_override)
        if stage1_override is not None
        else max(top_k * 2, comparison_top_docs * comparison_per_doc_k)
    )

    stage1_results = await use_case.execute(
        query=query,
        top_k=comparison_stage1_top_k,
        min_score=min_score,
    )
    doc_scores: dict[int, float] = {}
    for item in stage1_results:
        doc_id = item.chunk.document_id.value
        doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + item.score.value
    top_doc_ids = [
        doc_id
        for doc_id, _ in sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
        [:comparison_top_docs]
    ]
    if len(top_doc_ids) < 2:
        return stage1_results[:top_k]

    import asyncio

    per_doc_results = await asyncio.gather(
        *(
            use_case.execute(
                query=query,
                document_id=doc_id,
                top_k=comparison_per_doc_k,
                min_score=min_score,
            )
            for doc_id in top_doc_ids
        )
    )
    combined: list = []
    for doc_id, results in zip(top_doc_ids, per_doc_results, strict=False):
        combined.extend(results)
    return combined or stage1_results[:top_k]


async def _search_prefiltered_documents(
    *,
    query: str,
    use_case: HybridSearchKnowledge,
    top_k: int,
    min_score: float | None,
    document_ids: list[int],
    ctx: ToolContext,
) -> list:
    per_doc_k = int(ctx.tool_context.get("search_prefilter_per_doc_k", max(2, top_k)))
    import asyncio

    per_doc_results = await asyncio.gather(
        *(
            use_case.execute(
                query=query,
                document_id=doc_id,
                top_k=per_doc_k,
                min_score=min_score,
            )
            for doc_id in document_ids
        )
    )
    merged = [item for group in per_doc_results for item in group]
    merged.sort(key=lambda item: item.score.value, reverse=True)
    return merged[: max(top_k, per_doc_k)]


async def _format_results(
    results: list,
    use_case: HybridSearchKnowledge,
    include_path: bool,
    *,
    documents_by_id: dict[int, Document] | None = None,
) -> SearchResults:
    if not results:
        return SearchResults(results=[])
    if documents_by_id is None:
        documents_by_id = await _fetch_documents_map(use_case, results)
    return SearchResults(
        results=[
            SearchResult(
                title=_format_document_title(
                    documents_by_id.get(item.chunk.document_id.value),
                    include_path,
                    item.chunk.document_id.value,
                ),
                document_title=_format_document_title(
                    documents_by_id.get(item.chunk.document_id.value),
                    include_path,
                    item.chunk.document_id.value,
                ),
                slide_title=_chunk_metadata(item).get("slide_title"),
                snippet=item.chunk.content,
                chunk_id=item.chunk.chunk_id.value,
                document_id=item.chunk.document_id.value,
                score=item.score.value,
                slide_range=_format_slide_range(item.chunk.start_slide, item.chunk.end_slide),
                document_url=_document_url(documents_by_id.get(item.chunk.document_id.value), item),
                slide_url=_slide_url(documents_by_id.get(item.chunk.document_id.value), item),
                source_url=_slide_url(documents_by_id.get(item.chunk.document_id.value), item)
                or _document_url(documents_by_id.get(item.chunk.document_id.value), item),
            )
            for item in results
        ]
    )


async def _fetch_documents_map(
    use_case: HybridSearchKnowledge,
    results: list,
) -> dict[int, Document]:
    document_ids = {item.chunk.document_id.value for item in results}
    if not document_ids:
        return {}
    documents = await use_case.repository.fetch_documents_by_ids(
        [DocumentId(doc_id) for doc_id in document_ids]
    )
    return {doc.document_id.value: doc for doc in documents}


def _format_slide_range(start: int | None, end: int | None) -> str | None:
    if start is None and end is None:
        return None
    if start is None:
        return f"slides ?-{end}"
    if end is None or end == start:
        return f"slide {start}"
    return f"slides {start}-{end}"


def _format_document_title(document, include_path: bool, fallback_id: int) -> str:
    if document is None or not document.filename:
        return f"Document {fallback_id}"
    if include_path and document.file_path:
        return f"{document.filename} ({document.file_path})"
    return document.filename


def _chunk_metadata(item) -> dict:
    meta = item.chunk.metadata
    if isinstance(meta, dict):
        return meta
    return {}


def _document_url(document, item) -> str | None:
    meta = _chunk_metadata(item)
    meta_doc_url = meta.get("document_url")
    if isinstance(meta_doc_url, str) and meta_doc_url:
        return meta_doc_url
    if document is not None and isinstance(document.file_path, str) and document.file_path:
        return document.file_path
    return None


def _slide_url(document, item) -> str | None:
    meta = _chunk_metadata(item)
    meta_slide_url = meta.get("slide_url")
    if isinstance(meta_slide_url, str) and meta_slide_url:
        return meta_slide_url
    return _document_url(document, item)
