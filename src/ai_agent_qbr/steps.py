"""User-facing step messaging for planner and tool progress."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StepDisplay:
    label: str
    message: str
    index: int | None = None


_STEP_MAP = {
    "agent_capabilities": StepDisplay(
        label="Explaining Capabilities",
        message="Sharing what I can help with.",
        index=1,
    ),
    "search_documents": StepDisplay(
        label="Searching",
        message="Searching QBR materials for relevant details.",
        index=2,
    ),
    "metadata_catalog": StepDisplay(
        label="Inspecting Metadata",
        message="Inspecting metadata catalog and access scope.",
        index=2,
    ),
    "analyze_results": StepDisplay(
        label="Summarizing",
        message="Summarizing the most relevant QBR insights.",
        index=3,
    ),
}


def _normalize_step_name(name: str | None) -> str | None:
    if not name:
        return None
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized or None


def resolve_step_display(name: str | None) -> StepDisplay | None:
    normalized = _normalize_step_name(name)
    if not normalized:
        return None
    return _STEP_MAP.get(normalized)


def format_progress_update(
    *,
    step_name: str | None,
    message: str | None,
    step: int | str | None,
) -> tuple[str | None, int | str | None]:
    display = resolve_step_display(step_name)
    if display:
        return display.message, display.index
    return message or step_name, step


def format_completion(step_name: str | None) -> str | None:
    display = resolve_step_display(step_name)
    if display:
        return f"Completed: {display.label}"
    if step_name:
        return f"Completed: {step_name}"
    return None
