from ai_agent_qbr.transport.websocket.dto import parse_planner_event
from ai_agent_qbr.transport.websocket.schemas import OutputThinking


def test_parse_planner_event_thinking():
    event = {"type": "thinking", "payload": {"message": "step", "step": 1}}
    dto = parse_planner_event(event)
    assert isinstance(dto, OutputThinking)
    assert dto.message == "step"
    assert dto.step == 1
