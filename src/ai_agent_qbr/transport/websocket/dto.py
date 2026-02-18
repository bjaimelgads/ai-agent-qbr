"""Utilities for mapping planner events to websocket DTOs."""

from __future__ import annotations

from typing import Any

from .schemas import OutputThinking, PlannerEventPayload, ServerMessage
from ...steps import format_completion, format_progress_update


def parse_planner_event(event: Any) -> ServerMessage | None:
    """Map PenguiFlow planner events to WebSocket DTOs.

    Handles PlannerEvent objects from penguiflow.planner as well as
    dict-based and custom event formats.
    """
    if isinstance(event, PlannerEventPayload):
        payload = event.payload
        event_type = event.type
    else:
        payload = None
        event_type = None
        if hasattr(event, "event_type"):
            event_type = getattr(event, "event_type")
        if hasattr(event, "extra"):
            payload = getattr(event, "extra", {})
        elif hasattr(event, "to_payload"):
            payload = event.to_payload()
        if isinstance(event, dict):
            event_type = event.get("type") or event.get("event_type")
            payload = event.get("payload") or event.get("extra")

    if not event_type:
        return None

    event_type = str(event_type).lower()
    payload = payload or {}

    # Map planner events to thinking updates
    if event_type in {"thinking", "step", "plan", "node_start", "iteration_start", "step_start"}:
        step_name = payload.get("step_name") or payload.get("node_name")
        message, step = format_progress_update(
            step_name=step_name,
            message=payload.get("message"),
            step=payload.get("step"),
        )
        if message is None and step is None:
            return None
        return OutputThinking(message=message, step=step)

    # node_end or step_complete events indicate progress
    if event_type in {"node_end", "step_complete"}:
        node_name = payload.get("node_name") or payload.get("step_name")
        message = format_completion(node_name)
        return OutputThinking(message=message) if message else None

    if event_type == "stream_chunk":
        meta = payload.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        message, step = format_progress_update(
            step_name=meta.get("step_name"),
            message=payload.get("text") or meta.get("message"),
            step=meta.get("step"),
        )
        if message is None and step is None:
            return None
        return OutputThinking(message=message, step=step)

    return None
