from __future__ import annotations

from ai_agent_qbr.observability.mlflow_logger import _normalize_tracking_uri


def test_normalize_tracking_uri_keeps_non_sqlite() -> None:
    assert _normalize_tracking_uri("http://localhost:5000") == "http://localhost:5000"


def test_normalize_tracking_uri_keeps_absolute_sqlite() -> None:
    uri = "sqlite:////tmp/mlflow.db"
    assert _normalize_tracking_uri(uri) == uri


def test_normalize_tracking_uri_expands_relative_sqlite_path() -> None:
    out = _normalize_tracking_uri("sqlite:///mlflow.db")
    assert out is not None
    assert out.startswith("sqlite:////")
    assert out.endswith("/mlflow.db")
