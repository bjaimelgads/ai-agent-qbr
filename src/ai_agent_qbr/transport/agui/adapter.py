"""AG-UI adapter for streaming PenguiFlow planner events over WebSockets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import quote

from ag_ui.core import RunAgentInput
from penguiflow.agui_adapter.base import AGUIAdapter, AGUIEvent, generate_id
from penguiflow.planner import PlannerEvent
try:
    from penguiflow.sessions.projections import PlannerEventProjector
except ImportError:  # pragma: no cover - fallback for older penguiflow
    class PlannerEventProjector:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def project(self, event: PlannerEvent):
            return []


class AGUIWebsocketAdapter(AGUIAdapter):
    """Translate planner events into AG-UI events for WebSocket delivery."""

    def __init__(
        self,
        *,
        artifact_url_prefix: str = "/artifacts",
        resource_url_prefix: str = "/resources",
    ) -> None:
        super().__init__()
        self._artifact_url_prefix = artifact_url_prefix.rstrip("/")
        self._resource_url_prefix = resource_url_prefix.rstrip("/")
        self._streamed_answer = False
        self._session_id: str | None = None
        self._task_id: str | None = None
        self._trace_id: str | None = None

    @property
    def streamed_answer(self) -> bool:
        return self._streamed_answer

    def start_run(self, input: RunAgentInput) -> None:
        self._streamed_answer = False
        self._session_id = input.thread_id
        self._task_id = input.run_id
        self._trace_id = input.run_id

    async def run(self, input: RunAgentInput):  # pragma: no cover - required by base class
        raise NotImplementedError("AGUIWebsocketAdapter does not execute runs directly")

    def end_run(self) -> None:
        self._streamed_answer = False
        self._session_id = None
        self._task_id = None
        self._trace_id = None

    def convert_planner_event(self, event: PlannerEvent) -> list[AGUIEvent]:
        extra = dict(event.extra or {})
        mapped: list[AGUIEvent] = []

        if self._session_id and self._task_id:
            projector = PlannerEventProjector(
                session_id=self._session_id,
                task_id=self._task_id,
                trace_id=self._trace_id,
            )
            for state_update in projector.project(event):
                mapped.append(self.custom("state_update", state_update.model_dump(mode="json")))

        if event.event_type == "step_start":
            step_name = extra.get("step_name") or event.node_name or f"step_{event.trajectory_step}"
            mapped.append(self.step_start(step_name, **extra))
            return mapped

        if event.event_type == "step_complete":
            step_name = event.node_name or extra.get("step_name") or f"step_{event.trajectory_step}"
            mapped.append(self.step_end(step_name, **extra))
            return mapped

        if event.event_type == "stream_chunk":
            text = str(extra.get("text") or "")
            if text:
                mapped.append(
                    self.custom(
                        "thinking",
                        {
                            "text": text,
                            "done": bool(extra.get("done")),
                            "stream_id": extra.get("stream_id"),
                            "seq": extra.get("seq"),
                            "meta": extra.get("meta", {}),
                        },
                    )
                )
            return mapped

        if event.event_type == "llm_stream_chunk":
            channel = extra.get("channel")
            text = str(extra.get("text") or "")
            done = bool(extra.get("done"))
            phase = extra.get("phase")

            if channel == "thinking":
                if text:
                    mapped.append(
                        self.custom(
                            "thinking",
                            {"text": text, "phase": phase, "done": done},
                        )
                    )
                return mapped

            if channel == "revision":
                if text:
                    mapped.append(self.custom("revision", {"text": text, "done": done}))
                return mapped

            if channel == "answer":
                if not self._message_started:
                    mapped.append(self.text_start())
                if text:
                    mapped.append(self.text_content(text))
                self._streamed_answer = True
                return mapped

            return mapped

        if event.event_type == "tool_call_start":
            raw_id = extra.get("tool_call_id")
            tool_call_id = str(raw_id) if raw_id else None
            tool_name = str(extra.get("tool_name") or "")
            args_json = extra.get("args_json")
            if not self._message_started:
                mapped.append(self.text_start())
            start_event = self.tool_start(tool_name, tool_call_id=tool_call_id)
            mapped.append(start_event)
            tool_call_id = start_event.tool_call_id
            if args_json is not None:
                args_text = str(args_json)
                if args_text:
                    mapped.append(self.tool_args(tool_call_id, args_text))
            return mapped

        if event.event_type == "tool_call_end":
            raw_id = extra.get("tool_call_id")
            if raw_id:
                mapped.append(self.tool_end(str(raw_id)))
            return mapped

        if event.event_type == "tool_call_result":
            raw_id = extra.get("tool_call_id")
            if raw_id:
                result_json = extra.get("result_json")
                mapped.append(self.tool_result(str(raw_id), str(result_json)))
            return mapped

        if event.event_type == "artifact_chunk":
            mapped.append(self._artifact_chunk_custom_event(extra))
            return mapped

        if event.event_type == "artifact_stored":
            mapped.append(self._artifact_custom_event(extra))
            return mapped

        if event.event_type == "resource_updated":
            mapped.append(self._resource_custom_event(extra))
            return mapped

        return mapped

    def emit_text_block(self, text: str) -> list[AGUIEvent]:
        if not text:
            return []
        return [self.text_start(), self.text_content(text), self.text_end()]

    def _artifact_custom_event(self, extra: Mapping[str, Any]) -> AGUIEvent:
        artifact_id = str(extra.get("artifact_id") or "")
        artifact = {
            "id": artifact_id,
            "mime_type": extra.get("mime_type"),
            "size_bytes": extra.get("size_bytes"),
            "filename": extra.get("artifact_filename") or extra.get("filename"),
            "source": extra.get("source") or {},
        }
        return self.custom(
            "artifact_stored",
            {
                "artifact": artifact,
                "download_url": f"{self._artifact_url_prefix}/{artifact_id}",
            },
        )

    def _artifact_chunk_custom_event(self, extra: Mapping[str, Any]) -> AGUIEvent:
        return self.custom(
            "artifact_chunk",
            {
                "stream_id": extra.get("stream_id"),
                "seq": extra.get("seq"),
                "done": extra.get("done", False),
                "artifact_type": extra.get("artifact_type"),
                "chunk": extra.get("chunk"),
                "meta": extra.get("meta", {}),
            },
        )

    def _resource_custom_event(self, extra: Mapping[str, Any]) -> AGUIEvent:
        uri = str(extra.get("uri") or "")
        namespace = str(extra.get("namespace") or "")
        encoded_uri = quote(uri, safe="")
        return self.custom(
            "resource_updated",
            {
                "namespace": namespace,
                "uri": uri,
                "read_url": f"{self._resource_url_prefix}/{namespace}/{encoded_uri}",
            },
        )


def build_run_input(
    *,
    session_id: str,
    message: str,
    run_id: str | None = None,
) -> RunAgentInput:
    """Build a RunAgentInput from a simple chat message."""
    return RunAgentInput(
        thread_id=session_id,
        run_id=run_id or generate_id("run"),
        parent_run_id=None,
        state={},
        messages=[
            {
                "id": generate_id("user"),
                "role": "user",
                "content": message,
            }
        ],
        tools=[],
        context=[],
        forwarded_props={},
    )


def pick_query(messages: Iterable[Any]) -> str:
    for msg in reversed(list(messages or [])):
        role = getattr(msg, "role", None)
        content = getattr(msg, "content", None)
        if role is None and isinstance(msg, Mapping):
            role = msg.get("role")
            content = msg.get("content")
        if role != "user":
            continue
        text = _extract_text_content(content)
        if text:
            return text
    return ""


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        text = content.get("text")
        if isinstance(text, str):
            return text
    text_attr = getattr(content, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    if isinstance(content, Iterable) and not isinstance(content, (str, bytes, Mapping)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        if parts:
            return "\n".join(parts)
    return ""
