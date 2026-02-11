from __future__ import annotations

import time

from penguiflow.planner import PlannerEvent

from ai_agent_qbr.transport.agui.adapter import (
    AGUIWebsocketAdapter,
    _reasoning_text_from_state_update,
    _sanitize_reasoning_text,
    _sanitize_state_update_reasoning_payload,
    build_run_input,
    pick_query,
)


def test_agui_adapter_maps_streamed_answer() -> None:
    adapter = AGUIWebsocketAdapter()
    run_input = build_run_input(session_id="s1", message="hello", run_id="run-1")
    adapter.start_run(run_input)

    event = PlannerEvent(
        event_type="llm_stream_chunk",
        ts=time.time(),
        trajectory_step=0,
        extra={"channel": "answer", "text": "Hi", "done": False},
    )
    mapped = adapter.convert_planner_event(event)
    event_types = [item.get("type") if isinstance(item, dict) else item.type for item in mapped]
    assert "TEXT_MESSAGE_START" in event_types
    assert "TEXT_MESSAGE_CONTENT" in event_types
    assert adapter.streamed_answer is True


def test_agui_adapter_maps_steps() -> None:
    adapter = AGUIWebsocketAdapter()
    run_input = build_run_input(session_id="s1", message="hello", run_id="run-2")
    adapter.start_run(run_input)

    event = PlannerEvent(
        event_type="step_start",
        ts=time.time(),
        trajectory_step=1,
        node_name="planner",
        extra={},
    )
    mapped = adapter.convert_planner_event(event)
    event_types = [item.get("type") if isinstance(item, dict) else item.type for item in mapped]
    assert "STEP_START" in event_types or "STEP_STARTED" in event_types


def test_pick_query_supports_text_messages() -> None:
    messages = [
        {"id": "m1", "role": "assistant", "text": "Earlier response"},
        {"id": "m2", "role": "user", "text": "Hello from UI schema"},
    ]
    assert pick_query(messages) == "Hello from UI schema"


def test_reasoning_text_from_state_update_prefers_text_then_thought() -> None:
    text_payload = {"update_type": "THINKING", "content": {"text": "React thinking", "channel": "thinking"}}
    thought_payload = {"update_type": "THINKING", "content": {"thought": "planning next step"}}
    empty_payload = {"update_type": "THINKING", "content": {"text": ""}}
    answer_payload = {"update_type": "THINKING", "content": {"text": "answer chunk", "channel": "answer"}}
    result_payload = {"update_type": "RESULT", "content": {"text": "late summary"}}

    assert _reasoning_text_from_state_update(text_payload) == "React thinking"
    assert _reasoning_text_from_state_update(thought_payload) == "planning next step"
    assert _reasoning_text_from_state_update(empty_payload) is None
    assert _reasoning_text_from_state_update(answer_payload) is None
    assert _reasoning_text_from_state_update(result_payload) is None


def test_agui_adapter_reasoning_content_chunks_end_with_newline() -> None:
    adapter = AGUIWebsocketAdapter(reasoning_source="status")
    run_input = build_run_input(session_id="s1", message="hello", run_id="run-reasoning-newline")
    adapter.start_run(run_input)

    event = PlannerEvent(
        event_type="stream_chunk",
        ts=time.time(),
        trajectory_step=0,
        extra={
            "stream_id": "status",
            "text": "Preparing Genie query",
            "done": True,
        },
    )
    mapped = adapter.convert_planner_event(event)
    deltas = [
        item.get("delta")
        for item in mapped
        if isinstance(item, dict) and item.get("name") == "REASONING_MESSAGE_CONTENT"
    ]
    assert deltas == ["Preparing Genie query\n"]
    assert not any(
        isinstance(item, dict) and item.get("name") == "REASONING_END"
        for item in mapped
    )


def test_agui_adapter_emits_reasoning_start_content_end_with_value_fields() -> None:
    adapter = AGUIWebsocketAdapter(reasoning_source="status")
    run_input = build_run_input(session_id="s1", message="hello", run_id="run-reasoning-shape")
    adapter.start_run(run_input)

    event = PlannerEvent(
        event_type="stream_chunk",
        ts=time.time(),
        trajectory_step=0,
        extra={
            "stream_id": "status",
            "text": "Thinking",
            "done": True,
        },
    )
    mapped = adapter.convert_planner_event(event)
    custom = [item for item in mapped if isinstance(item, dict) and item.get("type") == "CUSTOM"]
    names = [item.get("name") for item in custom]
    assert names == ["REASONING_START", "REASONING_MESSAGE_CONTENT"]
    assert isinstance(custom[0].get("value"), dict)
    assert custom[1].get("value", {}).get("text") == "Thinking\n"
    flushed = adapter.flush_reasoning()
    assert len(flushed) == 1
    assert flushed[0]["name"] == "REASONING_END"
    assert flushed[0].get("value", {}).get("done") is True


def test_sanitize_reasoning_text_unescapes_newlines() -> None:
    assert _sanitize_reasoning_text("line 1\\nline 2") == "line 1\nline 2"


def test_sanitize_reasoning_text_keeps_inline_numbered_steps_unchanged() -> None:
    text = "I need to: 1. Parse query 2. Check window 3. Build Genie query"
    assert _sanitize_reasoning_text(text) == text


def test_sanitize_reasoning_text_keeps_inline_dash_bullets_and_date_ranges_unchanged() -> None:
    text = "Use this range Jan 12 - Feb 10 and include: - Impressions - Clicks - CTR"
    assert _sanitize_reasoning_text(text) == text


def test_sanitize_state_update_reasoning_payload_formats_content_fields() -> None:
    payload = {
        "update_type": "THINKING",
        "content": {
            "text": "I need to: 1. Parse query 2. Build query",
            "thought": "Include: - Impressions - CTR",
            "message": "line 1\\nline 2",
        },
    }
    out = _sanitize_state_update_reasoning_payload(payload)
    assert out["content"]["text"] == "I need to: 1. Parse query 2. Build query"
    assert out["content"]["thought"] == "Include: - Impressions - CTR"
    assert out["content"]["message"] == "line 1\nline 2"
