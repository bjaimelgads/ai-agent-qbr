"""Search tool."""

from __future__ import annotations

from penguiflow.catalog import tool
from penguiflow.planner import ToolContext

from ai_agent_qbr.models import Query, SearchResult, SearchResults
from qbr_agent.application.use_cases import HybridSearchKnowledge


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

    return SearchResults(
        results=[
            SearchResult(
                title=f"Document {item.chunk.document_id.value}",
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
