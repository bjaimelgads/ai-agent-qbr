"""Telemetry utilities for planner events."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any


class AgentTelemetry:
    def __init__(
        self,
        *,
        flow_name: str,
        logger: logging.Logger,
        status_callback: Callable[[str | None, int | None], None] | None = None,
        event_callback: Callable[[Any], None] | None = None,
    ) -> None:
        self._flow_name = flow_name
        self._logger = logger
        self._status_callbacks: list[Callable[[str | None, int | None], None]] = []
        self._event_callbacks: list[Callable[[Any], None]] = []
        if status_callback:
            self._status_callbacks.append(status_callback)
        if event_callback:
            self._event_callbacks.append(event_callback)

    def publish_status(self, message: str | None = None, step: int | None = None) -> None:
        for callback in list(self._status_callbacks):
            callback(message, step)

    def record_planner_event(self, event: Any) -> None:
        event_type = None
        payload: Any = None
        if isinstance(event, dict):
            event_type = event.get("event_type") or event.get("type")
            payload = event.get("payload") or event.get("extra")
        else:
            if hasattr(event, "event_type"):
                event_type = getattr(event, "event_type")
            if hasattr(event, "extra"):
                payload = getattr(event, "extra", None)
            elif hasattr(event, "to_payload"):
                payload = event.to_payload()
        node_name = None
        if isinstance(payload, dict):
            node_name = payload.get("node_name") or payload.get("step_name")

        debug_events = os.getenv("PLANNER_DEBUG_EVENTS", "false").lower() in {"1", "true", "yes", "on"}
        if debug_events and event_type:
            payload_preview = payload
            if isinstance(payload_preview, dict):
                # Keep logs concise.
                payload_preview = {
                    key: payload_preview.get(key)
                    for key in ("tool_name", "tool_call_id", "message", "step_name", "node_name", "channel", "text")
                    if key in payload_preview
                }
            self._logger.info("Planner event: %s node=%s payload=%s", event_type, node_name, payload_preview)
        elif self._logger.isEnabledFor(logging.DEBUG) and event_type:
            self._logger.debug("Planner event: %s node=%s", event_type, node_name)
        for callback in list(self._event_callbacks):
            callback(event)

    def add_status_listener(self, callback: Callable[[str | None, int | None], None]) -> None:
        self._status_callbacks.append(callback)

    def remove_status_listener(self, callback: Callable[[str | None, int | None], None]) -> None:
        if callback in self._status_callbacks:
            self._status_callbacks.remove(callback)

    def add_event_listener(self, callback: Callable[[Any], None]) -> None:
        self._event_callbacks.append(callback)

    def remove_event_listener(self, callback: Callable[[Any], None]) -> None:
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)

    @contextmanager
    def subscribe(
        self,
        *,
        status_callback: Callable[[str | None, int | None], None] | None = None,
        event_callback: Callable[[Any], None] | None = None,
    ):
        if status_callback:
            self.add_status_listener(status_callback)
        if event_callback:
            self.add_event_listener(event_callback)
        try:
            yield
        finally:
            if status_callback:
                self.remove_status_listener(status_callback)
            if event_callback:
                self.remove_event_listener(event_callback)
