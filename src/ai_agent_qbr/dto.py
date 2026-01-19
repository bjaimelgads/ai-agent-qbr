"""Utilities for mapping planner events to WebSocket DTOs."""

from __future__ import annotations

from typing import Any

from .api_models import OutputThinking, PlannerEventPayload, ServerMessage


def parse_planner_event(event: Any) -> ServerMessage | None:
    if isinstance(event, PlannerEventPayload):
        payload = event.payload
        event_type = event.type
    else:
        payload = None
        event_type = None
        if hasattr(event, "event_type"):
            event_type = getattr(event, "event_type")
        if hasattr(event, "to_payload"):
            payload = event.to_payload()
        if isinstance(event, dict):
            event_type = event.get("type")
            payload = event.get("payload")

    if not event_type:
        return None

    event_type = str(event_type).lower()
    payload = payload or {}

    if event_type in {"thinking", "step", "plan", "node_start"}:
        return OutputThinking(message=payload.get("message"), step=payload.get("step"))
    return None
