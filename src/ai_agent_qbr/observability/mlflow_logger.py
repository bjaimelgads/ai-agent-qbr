"""MLflow tracing helpers."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

_LOGGER = logging.getLogger(__name__)


def _get_mlflow():
    try:
        import mlflow
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("MLflow unavailable: %s", exc)
        return None
    return mlflow


@dataclass
class MlflowConfig:
    enabled: bool
    tracking_uri: str | None
    experiment: str | None


class MlflowTracer:
    def __init__(self, config: MlflowConfig) -> None:
        self._config = config
        self._mlflow = _get_mlflow() if config.enabled else None

    @contextmanager
    def interaction_run(self, *, run_name: str, tags: dict[str, str]) -> Iterator[object | None]:
        if not self._mlflow:
            yield None
            return
        if self._config.tracking_uri:
            self._mlflow.set_tracking_uri(self._config.tracking_uri)
        if self._config.experiment:
            self._mlflow.set_experiment(self._config.experiment)
        with self._mlflow.start_run(run_name=run_name, tags=tags):
            yield self._mlflow

    @contextmanager
    def nested_run(self, *, run_name: str, tags: dict[str, str] | None = None) -> Iterator[object | None]:
        if not self._mlflow:
            yield None
            return
        with self._mlflow.start_run(run_name=run_name, nested=True, tags=tags or {}):
            yield self._mlflow

    def log_json(self, mlflow_obj: object | None, name: str, payload: Any) -> None:
        if not mlflow_obj:
            return
        text = json.dumps(payload, ensure_ascii=True, indent=2, default=str)
        if hasattr(mlflow_obj, "log_text"):
            mlflow_obj.log_text(text, f"{name}.json")
            return
        self._log_text_fallback(mlflow_obj, name, text)

    def log_text(self, mlflow_obj: object | None, name: str, text: str) -> None:
        if not mlflow_obj:
            return
        if hasattr(mlflow_obj, "log_text"):
            mlflow_obj.log_text(text, f"{name}.txt")
            return
        self._log_text_fallback(mlflow_obj, name, text)

    def _log_text_fallback(self, mlflow_obj: object, name: str, text: str) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / f"{name}.txt"
            path.write_text(text, encoding="utf-8")
            mlflow_obj.log_artifact(str(path))


@dataclass
class MlflowTraceConfig:
    enabled: bool
    tracking_uri: str | None
    experiment: str | None


class MlflowTrace:
    def __init__(self, config: MlflowTraceConfig) -> None:
        self._config = config
        self._mlflow = _get_mlflow() if config.enabled else None
        self._configured = False

    def _configure(self) -> None:
        if self._configured or not self._mlflow:
            return
        if self._config.tracking_uri:
            self._mlflow.set_tracking_uri(self._config.tracking_uri)
        if self._config.experiment:
            self._mlflow.set_experiment(self._config.experiment)
        try:
            self._mlflow.tracing.enable()
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Failed to enable MLflow tracing: %s", exc)
        self._configured = True

    @contextmanager
    def span(
        self,
        *,
        name: str,
        span_type: str = "UNKNOWN",
        attributes: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
    ) -> Iterator[object | None]:
        if not self._mlflow:
            yield None
            return
        self._configure()
        with self._mlflow.start_span(name=name, span_type=span_type, attributes=attributes) as span:
            if inputs:
                span.set_inputs(inputs)
            yield span

    def set_outputs(self, span: object | None, outputs: dict[str, Any]) -> None:
        if not span:
            return
        if hasattr(span, "set_outputs"):
            span.set_outputs(outputs)

    def set_attributes(self, span: object | None, attributes: dict[str, Any]) -> None:
        if not span:
            return
        if hasattr(span, "set_attributes"):
            span.set_attributes(attributes)
