from ai_agent_qbr.planner import SYSTEM_PROMPT_EXTRA


def test_planner_prompt_requires_query_metrics_fanout_for_multi_entity_intents():
    assert "you MUST fan out into multiple `query_metrics` calls using `plan` + `join`" in SYSTEM_PROMPT_EXTRA
    assert "Do not issue one broad `query_metrics` call that mixes those values." in SYSTEM_PROMPT_EXTRA
