import pytest

from ai_agent_qbr.planner import ScriptedLLM


@pytest.mark.asyncio
async def test_scripted_llm_reseeds_when_exhausted():
    llm = ScriptedLLM(scripted=[{"thought": "once", "next_node": None, "args": None}])

    first = await llm.complete(messages=[{"content": "hi"}])
    assert "once" in first

    second = await llm.complete(messages=[{"content": "hi again"}])
    assert "search_documents" in second
