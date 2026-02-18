"""Stub react planner types."""

from dataclasses import dataclass


@dataclass
class PlannerAction:
    thought: str | None = None
    next_node: str | None = None
    args: dict | None = None
    plan: list | None = None
    join: dict | None = None
