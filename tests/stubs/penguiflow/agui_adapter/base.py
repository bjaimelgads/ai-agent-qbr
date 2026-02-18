"""Stub AG-UI adapter base classes for tests."""

from __future__ import annotations

import asyncio
import itertools
from typing import Any, AsyncIterator

from pydantic import BaseModel, Field

_id_counter = itertools.count(1)


def generate_id(prefix: str) -> str:
    return f"{prefix}_{next(_id_counter)}"


class AGUIEvent(BaseModel):
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    tool_call_id: str | None = None
    delta: str | None = None


class AGUIAdapter:
    def __init__(self) -> None:
        self._message_started = False

    def text_start(self) -> AGUIEvent:
        self._message_started = True
        return AGUIEvent(type="TEXT_MESSAGE_START")

    def text_content(self, text: str) -> AGUIEvent:
        return AGUIEvent(type="TEXT_MESSAGE_CONTENT", payload={"delta": text}, delta=text)

    def text_end(self) -> AGUIEvent:
        return AGUIEvent(type="TEXT_MESSAGE_END")

    def step_start(self, step_name: str, **extra: Any) -> AGUIEvent:
        return AGUIEvent(type="STEP_START", payload={"step": step_name, **extra})

    def step_end(self, step_name: str, **extra: Any) -> AGUIEvent:
        return AGUIEvent(type="STEP_END", payload={"step": step_name, **extra})

    def tool_start(self, tool_name: str, tool_call_id: str | None = None) -> AGUIEvent:
        call_id = tool_call_id or generate_id("tool")
        return AGUIEvent(type="TOOL_CALL_START", payload={"tool_name": tool_name}, tool_call_id=call_id)

    def tool_args(self, tool_call_id: str, args_text: str) -> AGUIEvent:
        return AGUIEvent(type="TOOL_CALL_ARGS", payload={"args": args_text}, tool_call_id=tool_call_id)

    def tool_end(self, tool_call_id: str) -> AGUIEvent:
        return AGUIEvent(type="TOOL_CALL_END", tool_call_id=tool_call_id)

    def tool_result(self, tool_call_id: str, result_text: str) -> AGUIEvent:
        return AGUIEvent(type="TOOL_CALL_RESULT", payload={"result": result_text}, tool_call_id=tool_call_id)

    def custom(self, event_type: str, payload: dict[str, Any]) -> AGUIEvent:
        return AGUIEvent(type=event_type, payload=payload)

    async def with_run_lifecycle(self, run_input: Any, events: AsyncIterator[Any]):
        yield AGUIEvent(type="RUN_STARTED", payload={"run_id": getattr(run_input, "run_id", None)})
        async for event in events:
            yield event
        yield AGUIEvent(type="RUN_FINISHED", payload={"run_id": getattr(run_input, "run_id", None)})
