from __future__ import annotations

import pytest

from ai_agent_qbr.config import Config


def test_reasoning_defaults() -> None:
    config = Config()
    assert config.use_native_reasoning is True
    assert config.reasoning_effort == "medium"


def test_reasoning_effort_accepts_allowed_values() -> None:
    for effort in (None, "low", "medium", "high"):
        config = Config(reasoning_effort=effort)
        config.validate()


def test_reasoning_effort_rejects_invalid_value() -> None:
    config = Config(reasoning_effort="max")
    with pytest.raises(ValueError, match="REASONING_EFFORT"):
        config.validate()
