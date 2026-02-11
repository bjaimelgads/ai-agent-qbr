"""AG-UI adapter for streaming PenguiFlow planner events over WebSockets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import re
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
        reasoning_source: str = "status",
    ) -> None:
        super().__init__()
        self._artifact_url_prefix = artifact_url_prefix.rstrip("/")
        self._resource_url_prefix = resource_url_prefix.rstrip("/")
        self._reasoning_source = reasoning_source if reasoning_source in {"status", "thinking"} else "status"
        self._streamed_answer = False
        self._session_id: str | None = None
        self._task_id: str | None = None
        self._trace_id: str | None = None
        self._current_tool_step: str | None = None
        self._reasoning_message_id: str | None = None

    @property
    def streamed_answer(self) -> bool:
        return self._streamed_answer

    def start_run(self, input: RunAgentInput) -> None:
        self._streamed_answer = False
        self._session_id = input.thread_id
        self._task_id = input.run_id
        self._trace_id = input.run_id
        self._current_tool_step = None
        self._reasoning_message_id = None

    async def run(self, input: RunAgentInput):  # pragma: no cover - required by base class
        raise NotImplementedError("AGUIWebsocketAdapter does not execute runs directly")

    def end_run(self) -> None:
        self._streamed_answer = False
        self._session_id = None
        self._task_id = None
        self._trace_id = None
        self._current_tool_step = None
        self._reasoning_message_id = None

    def convert_planner_event(self, event: PlannerEvent) -> list[AGUIEvent]:
        extra = dict(event.extra or {})
        mapped: list[AGUIEvent] = []
        suppress_reasoning_from_state_update = self._reasoning_source == "thinking"
        if event.event_type == "stream_chunk":
            stream_id = str(extra.get("stream_id") or "")
            meta = extra.get("meta")
            if stream_id == "status":
                suppress_reasoning_from_state_update = True
            elif isinstance(meta, Mapping) and meta.get("tool_name"):
                suppress_reasoning_from_state_update = True

        if self._session_id and self._task_id:
            projector = PlannerEventProjector(
                session_id=self._session_id,
                task_id=self._task_id,
                trace_id=self._trace_id,
            )
            for state_update in projector.project(event):
                state_payload = state_update.model_dump(mode="json")
                state_payload = _sanitize_state_update_reasoning_payload(state_payload)
                mapped.append(self.custom("state_update", state_payload))
                if suppress_reasoning_from_state_update:
                    continue
                reasoning_text = _reasoning_text_from_state_update(state_payload)
                if reasoning_text:
                    mapped.extend(self._emit_reasoning_events(text=reasoning_text, done=False))

        if event.event_type == "step_start":
            step_name = extra.get("step_name") or event.node_name or f"step_{event.trajectory_step}"
            if _should_skip_step(step_name):
                return mapped
            step_name = _friendly_step_name(step_name)
            mapped.extend(self._close_active_steps())
            mapped.append(self.step_start(step_name, **extra))
            return mapped

        if event.event_type == "step_complete":
            step_name = event.node_name or extra.get("step_name") or f"step_{event.trajectory_step}"
            if _should_skip_step(step_name):
                return mapped
            if self._reasoning_source == "thinking":
                reasoning_text = _reasoning_text_from_planner_event(event)
                if reasoning_text:
                    mapped.extend(self._emit_reasoning_events(text=reasoning_text, done=False))
            step_name = _friendly_step_name(step_name)
            if step_name in self._active_steps:
                mapped.append(self.step_end(step_name, **extra))
            return mapped

        if event.event_type == "stream_chunk":
            stream_id = str(extra.get("stream_id") or "")
            if stream_id == "status" and self._reasoning_source == "status":
                text = str(extra.get("text") or "")
                if text.endswith("\n"):
                    text = text.rstrip("\n")
                if text:
                    text = f"{text}\n"
                    mapped.extend(self._emit_reasoning_events(text=text, done=False))
            meta = extra.get("meta", {})
            if isinstance(meta, Mapping):
                meta = dict(meta)
                step_name = meta.get("step_name")
                if isinstance(step_name, str) and step_name:
                    if self._current_tool_step != step_name:
                        if self._current_tool_step and self._current_tool_step in self._active_steps:
                            mapped.append(self.step_end(self._current_tool_step))
                        self._current_tool_step = step_name
                        mapped.append(self.step_start(step_name))
            return mapped

        if event.event_type == "llm_stream_chunk":
            channel = extra.get("channel")
            text = str(extra.get("text") or "")
            done = bool(extra.get("done"))

            if channel == "thinking":
                if self._reasoning_source == "thinking" and (text or done):
                    mapped.extend(self._emit_reasoning_events(text=text, done=False))
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
            custom_event = self._artifact_chunk_custom_event(extra)
            if custom_event is not None:
                mapped.append(custom_event)
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

    def _close_active_steps(self) -> list[AGUIEvent]:
        if not self._active_steps:
            return []
        events: list[AGUIEvent] = []
        for step_name in list(self._active_steps):
            events.append(self.step_end(step_name))
        return events

    def _emit_reasoning_events(self, *, text: str, done: bool) -> list[AGUIEvent | dict[str, Any]]:
        events: list[AGUIEvent | dict[str, Any]] = []
        text = _sanitize_reasoning_text(text)
        if text and done and not text.endswith("\n"):
            # Delimit completed reasoning chunks so UIs can render each step on a new line.
            text = f"{text}\n"
        if not text and done and self._reasoning_message_id is None:
            return events
        if self._reasoning_message_id is None:
            self._reasoning_message_id = generate_id("reasoning")
            events.append(
                {
                    "type": "CUSTOM",
                    "name": "REASONING_START",
                    "messageId": self._reasoning_message_id,
                    "value": {"messageId": self._reasoning_message_id},
                }
            )
        if text:
            events.append(
                {
                    "type": "CUSTOM",
                    "name": "REASONING_MESSAGE_CONTENT",
                    "messageId": self._reasoning_message_id,
                    "delta": text,
                    "value": {"text": text, "delta": text},
                }
            )
        if done and self._reasoning_message_id is not None:
            events.append(
                {
                    "type": "CUSTOM",
                    "name": "REASONING_END",
                    "messageId": self._reasoning_message_id,
                    "value": {"done": True, "messageId": self._reasoning_message_id},
                }
            )
            self._reasoning_message_id = None
        return events

    def flush_reasoning(self) -> list[dict[str, Any]]:
        if self._reasoning_message_id is None:
            return []
        message_id = self._reasoning_message_id
        self._reasoning_message_id = None
        return [
            {
                "type": "CUSTOM",
                "name": "REASONING_END",
                "messageId": message_id,
                "value": {"done": True, "messageId": message_id},
            }
        ]

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

    def _artifact_chunk_custom_event(self, extra: Mapping[str, Any]) -> AGUIEvent | None:
        message_id = self._current_message_id
        meta = dict(extra.get("meta") or {}) if isinstance(extra.get("meta"), Mapping) else {}
        if message_id and "message_id" not in meta:
            meta["message_id"] = message_id
        chunk = extra.get("chunk")
        return self.custom(
            "artifact_chunk",
            {
                "stream_id": extra.get("stream_id"),
                "seq": extra.get("seq"),
                "done": extra.get("done", False),
                "artifact_type": extra.get("artifact_type"),
                "chunk": chunk,
                "message_id": message_id,
                "meta": meta,
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
            if content is None:
                content = msg.get("text")
        if role != "user":
            continue
        text = _extract_text_content(content)
        if text:
            return text
    return ""


def _friendly_step_name(step_name: str) -> str:
    if not step_name.startswith("step_"):
        return step_name
    try:
        index = int(step_name.split("_", 1)[1])
    except (ValueError, IndexError):
        return step_name
    labels = [
        "Processing request",
        "Analyzing request",
        "Gathering data",
        "Generating response",
        "Finalizing response",
    ]
    if 0 <= index < len(labels):
        return labels[index]
    return step_name


def _should_skip_step(step_name: str) -> bool:
    return step_name.startswith("step_")


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


def _reasoning_text_from_state_update(state_update: Mapping[str, Any]) -> str | None:
    update_type = str(state_update.get("update_type") or "").upper()
    if update_type and update_type not in {"THINKING"}:
        return None

    content = state_update.get("content")
    if not isinstance(content, Mapping):
        return None

    channel = str(content.get("channel") or "").lower()
    if channel == "answer":
        return None

    text = content.get("text")
    if isinstance(text, str) and text.strip():
        return text

    thought = content.get("thought")
    if isinstance(thought, str) and thought.strip():
        return thought

    message = content.get("message")
    if isinstance(message, str) and message.strip():
        return message

    return None


def _reasoning_text_from_planner_event(event: PlannerEvent) -> str | None:
    thought = (getattr(event, "thought", None) or "").strip()
    if not thought:
        return None
    if thought.lower() in {"planning next step", "finish"}:
        return None
    return thought


def _sanitize_reasoning_text(text: str) -> str:
    if not text:
        return text
    text = text.replace("\\n", "\n")
    text = re.sub(r"\btools\b", "steps", text, flags=re.IGNORECASE)
    text = re.sub(r"\btool\b", "step", text, flags=re.IGNORECASE)
    return text


def _sanitize_state_update_reasoning_payload(state_update: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(state_update)
    content = payload.get("content")
    if not isinstance(content, Mapping):
        return payload

    content_out = dict(content)
    for key in ("text", "thought", "message"):
        value = content_out.get(key)
        if isinstance(value, str) and value:
            content_out[key] = _sanitize_reasoning_text(value)

    payload["content"] = content_out
    return payload
