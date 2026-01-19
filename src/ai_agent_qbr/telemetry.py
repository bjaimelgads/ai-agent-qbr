"""Telemetry utilities for planner events."""

from __future__ import annotations

import logging
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
