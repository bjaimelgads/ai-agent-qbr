"""Status emission helpers for tool progress updates."""

from __future__ import annotations

from penguiflow.planner import ToolContext


class ToolStatusEmitter:
    """Emit progress updates from tools in a transport-agnostic way."""

    def __init__(self, ctx: ToolContext, *, tool_name: str) -> None:
        self._ctx = ctx
        self._tool_name = tool_name

    async def step(self, message: str, *, step_name: str | None = None) -> None:
        """Emit a human-friendly status step message."""
        output_protocol = str(self._ctx.tool_context.get("output_protocol") or "").lower()
        if output_protocol == "websocket":
            label = step_name or message
            self._publish_status(message, label)
            return

        meta = {"tool_name": self._tool_name}
        if step_name:
            meta["step_name"] = step_name
        await self._ctx.emit_chunk(
            stream_id="status",
            seq=self._next_seq(),
            text=message,
            done=False,
            meta=meta,
        )

    def _next_seq(self) -> int:
        key = "_status_seq"
        current = self._ctx.tool_context.get(key)
        if isinstance(current, int):
            next_seq = current + 1
        else:
            next_seq = 1
        self._ctx.tool_context[key] = next_seq
        return next_seq

    def _publish_status(self, message: str, step: str) -> None:
        candidate = self._ctx.tool_context.get("status_publisher")
        if not callable(candidate):
            return
        try:
            candidate(message, step)
        except TypeError:
            candidate(message)
