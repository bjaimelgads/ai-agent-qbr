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

    results = await use_case.execute(
        query=args.question,
        top_k=top_k,
        min_score=min_score,
    )

    include_path = bool(ctx.tool_context.get("retrieval_include_document_path", False))
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
