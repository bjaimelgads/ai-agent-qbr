"""Analysis tool."""

from __future__ import annotations

from penguiflow.catalog import tool
from penguiflow.planner import ToolContext

from ai_agent_qbr.models import FinalAnswer, SearchResults


@tool(desc="Summarise search results for the user", tags=["planner"])
async def analyze_results(args: SearchResults, ctx: ToolContext) -> FinalAnswer:
    status_publisher = ctx.tool_context.get("status_publisher")
    if callable(status_publisher):
        status_publisher("Summarizing the most relevant QBR insights.", "Summarizing")
    user = ctx.tool_context.get("user_id", "user")
    context = ctx.tool_context.get("qbr_answer_context")

    summary = "; ".join(hit.snippet for hit in args.results) or "No results found."
    if context:
        summary = f"{summary}\n\nContext:\n{context}"

    return FinalAnswer(text=f"{user}: {summary}")
