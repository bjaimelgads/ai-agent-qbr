"""Agent capabilities tool."""

from __future__ import annotations

from penguiflow.catalog import tool
from penguiflow.planner import ToolContext

from ai_agent_qbr.models import AgentCapabilitiesArgs, AgentCapabilitiesResult


@tool(
    desc=(
        "Return a user-facing overview of the LG Ads QBR agent's capabilities and example queries."
    ),
    side_effects="pure",
    tags=["info", "help"],
)
async def agent_capabilities(
    args: AgentCapabilitiesArgs, ctx: ToolContext
) -> AgentCapabilitiesResult:
    del ctx
    capabilities_text = (
        "I can help you get answers grounded in LG Ads QBRs, including:\n"
        "- Executive summaries of a quarter's performance and outcomes.\n"
        "- Performance trends across campaigns, markets, and time.\n"
        "- Inventory dynamics across CTV, OTT, and FAST.\n"
        "- Pacing risks, headwinds, and upside opportunities.\n"
        "- Strategic recommendations and next steps captured in QBRs.\n"
        "- Cross-quarter or cross-advertiser comparisons when covered in QBRs.\n"
        "- Slide-aware responses with references when available."
    )

    if not args.include_examples:
        return AgentCapabilitiesResult(
            capabilities_text=capabilities_text,
            sample_queries=[],
        )

    sample_queries = [
        "Summarize the latest QBR for LG Electronics and highlight key outcomes.",
        "What performance trends stood out this quarter for CTV inventory?",
        "Call out any pacing risks mentioned in the Q2 QBR for advertiser X.",
        "Compare Q1 vs Q2 insights for FAST inventory across top advertisers.",
        "What strategic recommendations were made for next quarter in the QBR?",
        "Which markets or regions were flagged as opportunities or risks?",
    ]

    return AgentCapabilitiesResult(
        capabilities_text=capabilities_text,
        sample_queries=sample_queries,
    )
