"""Analysis tool."""

from __future__ import annotations

from penguiflow.catalog import tool
from penguiflow.planner import ToolContext

from ai_agent_qbr.models import FinalAnswer, SearchResults
from ai_agent_qbr.tools.status import ToolStatusEmitter


@tool(desc="Summarise search results for the user", tags=["planner"])
async def analyze_results(args: SearchResults, ctx: ToolContext) -> FinalAnswer:
    status = ToolStatusEmitter(ctx, tool_name="analyze_results")
    await status.step("Summarizing the most relevant QBR insights.", step_name="Summarize")
    user = ctx.tool_context.get("user_id", "user")
    context = ctx.tool_context.get("qbr_answer_context")

    summary = "; ".join(hit.snippet for hit in args.results) or "No results found."
    if context:
        summary = f"{summary}\n\nContext:\n{context}"

    return FinalAnswer(text=f"{user}: {summary}")
