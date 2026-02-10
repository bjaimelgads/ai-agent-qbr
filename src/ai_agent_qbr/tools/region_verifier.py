"""Region verification tool."""

from __future__ import annotations

from penguiflow.catalog import tool
from penguiflow.planner import ToolContext

from ai_agent_qbr.infrastructure.region_verifier import RegionVerifier
from ai_agent_qbr.models import RegionFilterVerificationArgs, RegionFilterVerificationResult
from ai_agent_qbr.tools.status import ToolStatusEmitter


@tool(
    desc=(
        "Verify the regional scope of the query (US vs EMEA vs global) using a small LLM "
        "so retrieval can filter to the correct region."
    ),
    side_effects="read",
    tags=["planner"],
)
async def verify_region_filter(
    args: RegionFilterVerificationArgs,
    ctx: ToolContext,
) -> RegionFilterVerificationResult:
    status = ToolStatusEmitter(ctx, tool_name="verify_region_filter")
    await status.step("Verifying regional scope for this QBR.", step_name="Verify region")

    verifier = ctx.tool_context.get("region_verifier")
    if not isinstance(verifier, RegionVerifier):
        return RegionFilterVerificationResult(
            region_focus=None,
            needs_clarification=False,
            reason="region verifier unavailable",
            confidence=None,
        )

    return await verifier.verify(args.question)
