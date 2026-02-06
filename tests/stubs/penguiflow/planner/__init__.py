"""Stub planner module."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlannerFinish:
    payload: object
    metadata: dict
    reason: str = "answer_complete"


@dataclass
class PlannerPause:
    reason: str | None = None
    payload: object | None = None
    metadata: dict | None = None


class ToolContext:
    def __init__(self, tool_context=None):
        self.tool_context = tool_context or {}


class ReactPlanner:
    def __init__(self, *args, **kwargs):
        pass

    async def run(self, *, query, llm_context, tool_context):
        return PlannerFinish(reason="stub", payload={"raw_answer": ""}, metadata={})


@dataclass
class PlannerEvent:
    event_type: str
    node_name: str | None = None
    trajectory_step: int | None = None
    extra: dict | None = None


PlannerEventCallback = object
