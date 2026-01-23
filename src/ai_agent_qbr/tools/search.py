"""Search tool."""

from __future__ import annotations

from penguiflow.catalog import tool
from penguiflow.planner import ToolContext

from ai_agent_qbr.models import Query, SearchResult, SearchResults
from qbr_agent.application.use_cases import HybridSearchKnowledge
from qbr_agent.domain.value_objects import DocumentId


@tool(desc="Search internal QBR knowledge", side_effects="read", tags=["planner"])
async def search_documents(args: Query, ctx: ToolContext) -> SearchResults:
    status_publisher = ctx.tool_context.get("status_publisher")
    if callable(status_publisher):
        status_publisher("Searching QBR materials for relevant details.", "Searching")
    use_case = ctx.tool_context.get("qbr_search_use_case")
    if not isinstance(use_case, HybridSearchKnowledge):
        return SearchResults(results=[])

    top_k = int(ctx.tool_context.get("retrieval_top_k", 5))
    min_score = ctx.tool_context.get("retrieval_min_score")

    comparison_intent = bool(args.comparison_intent)
    if comparison_intent and callable(status_publisher):
        status_publisher("Comparing across multiple decks.", "Comparing")

    if comparison_intent:
        results = await _search_comparison_documents(
            query=args.question,
            use_case=use_case,
            top_k=top_k,
            min_score=min_score,
            ctx=ctx,
        )
    else:
        results = await use_case.execute(
            query=args.question,
            top_k=top_k,
            min_score=min_score,
        )

    include_path = bool(ctx.tool_context.get("retrieval_include_document_path", False))
    return await _format_results(results, use_case, include_path)


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
) -> SearchResults:
    if not results:
        return SearchResults(results=[])
    document_ids = {item.chunk.document_id.value for item in results}
    documents = await use_case.repository.fetch_documents_by_ids(
        [DocumentId(doc_id) for doc_id in document_ids]
    )
    documents_by_id = {doc.document_id.value: doc for doc in documents}
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
