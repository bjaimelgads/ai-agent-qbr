"""Stub ag_ui.core module for tests."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RunAgentInput(BaseModel):
    thread_id: str
    run_id: str
    parent_run_id: str | None = None
    state: dict[str, Any] = Field(default_factory=dict)
    messages: list[Any] = Field(default_factory=list)
    tools: list[Any] = Field(default_factory=list)
    context: list[Any] = Field(default_factory=list)
    forwarded_props: dict[str, Any] = Field(default_factory=dict)

