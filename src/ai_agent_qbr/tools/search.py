"""Search tool."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date

from penguiflow.catalog import tool
from penguiflow.planner import ToolContext

from ai_agent_qbr.infrastructure.region_filter import filter_items_by_region
from ai_agent_qbr.models import Query, SearchResult, SearchResults
from ai_agent_qbr.tools.question_normalization import normalize_question_arg
from ai_agent_qbr.tools.status import ToolStatusEmitter
from qbr_agent.application.use_cases import HybridSearchKnowledge
from qbr_agent.domain.entities import Document
from qbr_agent.domain.value_objects import DocumentId
from qbr_intelligence.metric_qa import MetricQueryEngine
from qbr_intelligence.metric_qa.intent import DeterministicIntentExtractor


_REGION_ALIAS = {
    "US": {"US", "USA", "UNITED STATES"},
    "EMEA": {"EMEA"},
    "GLOBAL": {"GLOBAL"},
}

_PERIOD_HALF_RE = re.compile(r"\b(H[12])\s*(?:FY)?\s*(20\d{2}|\d{2})\b", re.IGNORECASE)
_PERIOD_FY_HALF_RE = re.compile(r"\bFY\s*(20\d{2}|\d{2})\s*(H[12])\b", re.IGNORECASE)
_PERIOD_QUARTER_RE = re.compile(r"\b(Q[1-4])\s*(?:FY)?\s*(20\d{2}|\d{2})\b", re.IGNORECASE)


@dataclass(frozen=True)
class _DocumentFilters:
    clients: set[str]
    regions: set[str]
    periods: set[str]

    def enabled(self) -> bool:
        return bool(self.clients or self.regions or self.periods)


def _normalize_token(value: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]+", " ", value.upper()).strip()
    return re.sub(r"\s+", " ", normalized)


def _normalize_year(raw: str) -> str:
    if len(raw) == 2:
        return f"20{raw}"
    return raw


def _period_tokens_from_text(text: str | None) -> set[str]:
    if not text:
        return set()
    source = _normalize_token(text)
    tokens: set[str] = set()
    for half, year in _PERIOD_HALF_RE.findall(source):
        year4 = _normalize_year(year)
        tokens.add(f"{half.upper()} {year4}")
        tokens.add(f"FY{year4[2:]} {half.upper()}")
    for year, half in _PERIOD_FY_HALF_RE.findall(source):
        year4 = _normalize_year(year)
        tokens.add(f"{half.upper()} {year4}")
        tokens.add(f"FY{year4[2:]} {half.upper()}")
    for quarter, year in _PERIOD_QUARTER_RE.findall(source):
        year4 = _normalize_year(year)
        tokens.add(f"{quarter.upper()} {year4}")
        tokens.add(f"FY{year4[2:]} {quarter.upper()}")
    return tokens


def _region_tokens_from_text(text: str | None) -> set[str]:
    if not text:
        return set()
    source = _normalize_token(text)
    found: set[str] = set()
    for canonical, aliases in _REGION_ALIAS.items():
        if any(alias in source for alias in aliases):
            found.add(canonical)
    return found


def _client_tokens_from_document(document: Document) -> set[str]:
    tokens: set[str] = set()
    if document.client_name:
        tokens.add(_normalize_token(document.client_name))
    if document.filename:
        filename = _normalize_token(document.filename)
        if "DISNEY" in filename:
            tokens.add("DISNEY")
            tokens.add("DISNEY+")
        if "NETFLIX" in filename:
            tokens.add("NETFLIX")
        if "HULU" in filename:
            tokens.add("HULU")
    return tokens


def _period_tokens_from_document(document: Document) -> set[str]:
    tokens = _period_tokens_from_text(document.period)
    if document.filename:
        tokens.update(_period_tokens_from_text(document.filename))
    return tokens


def _region_tokens_from_document(document: Document) -> set[str]:
    text = " ".join(
        part for part in (document.filename, document.file_path, document.period) if part
    )
    return _region_tokens_from_text(text)


def _score_document_match(document: Document, filters: _DocumentFilters) -> tuple[bool, float]:
    """Return (passes_filter, score_boost) for a document-level intent filter."""
    boost = 0.0

    if filters.clients:
        doc_clients = _client_tokens_from_document(document)
        if doc_clients:
            if doc_clients & filters.clients:
                boost += 0.25
            else:
                return False, 0.0

    if filters.regions:
        doc_regions = _region_tokens_from_document(document)
        if doc_regions:
            if doc_regions & filters.regions:
                boost += 0.20
            else:
                return False, 0.0

    if filters.periods:
        doc_periods = _period_tokens_from_document(document)
        if doc_periods:
            if doc_periods & filters.periods:
                boost += 0.20
            else:
                return False, 0.0

    return True, boost


async def _build_document_filters(
    *,
    query: str,
    ctx: ToolContext,
) -> _DocumentFilters:
    engine = ctx.tool_context.get("metric_query_engine")
    if not isinstance(engine, MetricQueryEngine):
        return _DocumentFilters(clients=set(), regions=set(), periods=set())

    await engine._load_catalogs()
    anchor_date = await engine._select_anchor_date(query)
    extractor = DeterministicIntentExtractor(engine._catalog or [], engine._clients or [])
    intent, _ = extractor.extract(query, anchor_date=anchor_date)

    clients: set[str] = set()
    raw_clients = intent.client if isinstance(intent.client, list) else ([intent.client] if intent.client else [])
    for client in raw_clients:
        if client:
            clients.add(_normalize_token(client))

    regions: set[str] = set()
    raw_regions = intent.region if isinstance(intent.region, list) else ([intent.region] if intent.region else [])
    for region in raw_regions:
        normalized = _normalize_token(region)
        if normalized in _REGION_ALIAS:
            regions.add(normalized)
        else:
            for canonical, aliases in _REGION_ALIAS.items():
                if normalized in aliases:
                    regions.add(canonical)

    periods: set[str] = set()
    raw_periods = intent.period if isinstance(intent.period, list) else ([intent.period] if intent.period else [])
    for period in raw_periods:
        value = getattr(period, "value", None)
        if isinstance(value, str):
            periods.update(_period_tokens_from_text(value))
        start = getattr(period, "start", None)
        end = getattr(period, "end", None)
        kind = getattr(period, "type", None)
        if isinstance(start, date) and isinstance(end, date):
            year = end.year if end.month >= 7 else start.year
            if isinstance(kind, str) and kind.lower() == "half":
                half = "H1" if start.month <= 3 else "H2"
                periods.add(f"{half} {year}")
                periods.add(f"FY{str(year)[2:]} {half}")

    return _DocumentFilters(clients=clients, regions=regions, periods=periods)


@tool(desc="Search internal QBR knowledge", side_effects="read", tags=["planner"])
async def search_documents(args: Query, ctx: ToolContext) -> SearchResults:
    status = ToolStatusEmitter(ctx, tool_name="search_documents")
    await status.step("Searching QBR materials for relevant details.", step_name="Search QBR")
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
        await status.step("Comparing across multiple decks.", step_name="Compare decks")

    if comparison_intent:
        results = await _search_comparison_documents(
            query=canonical_query,
            use_case=use_case,
            top_k=top_k,
            min_score=min_score,
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
            passes, boost = _score_document_match(doc, doc_filters)
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
                snippet=item.chunk.content,
                chunk_id=item.chunk.chunk_id.value,
                document_id=item.chunk.document_id.value,
                score=item.score.value,
                slide_range=_format_slide_range(item.chunk.start_slide, item.chunk.end_slide),
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
