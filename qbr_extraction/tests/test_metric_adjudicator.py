from __future__ import annotations

from qbr_intelligence.metrics.adjudicator import AdjudicationRequest, LLMAdjudicator


def test_llm_adjudicator_parses_and_caches(tmp_path) -> None:
    calls = {"count": 0}

    def _call(prompt: str) -> str:
        calls["count"] += 1
        return '{"chosen_index": 1, "normalized_value": 2.5, "unit": "percent", "reasoning_short": "ok"}'

    adjudicator = LLMAdjudicator(call_llm=_call, cache_dir=tmp_path, enabled=True)
    request = AdjudicationRequest(
        deck_hash="abc",
        slide_index=1,
        metric_id="click_through_rate",
        metric_name="Click Through Rate",
        label_text="CTR",
        context_snippet="CTR: 2.5%",
        candidates=[{"raw_value_text": "2%"}, {"raw_value_text": "2.5%"}],
    )

    result = adjudicator.adjudicate(request)
    assert result is not None
    assert result.chosen_index == 1
    assert result.unit == "percent"
    assert calls["count"] == 1

    # Cached
    result2 = adjudicator.adjudicate(request)
    assert result2 is not None
    assert calls["count"] == 1
