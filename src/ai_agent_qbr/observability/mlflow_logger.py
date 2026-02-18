"""MLflow tracing helpers."""

from __future__ import annotations

import json
import logging
import os
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


def _normalize_tracking_uri(uri: str | None) -> str | None:
    if not uri:
        return uri
    if not uri.startswith("sqlite:///") or uri.startswith("sqlite:////"):
        return uri
    db_path = uri.replace("sqlite:///", "", 1)
    if not os.path.isabs(db_path):
        db_path = os.path.abspath(db_path)
    return f"sqlite:////{db_path.lstrip('/')}"


@dataclass
class MlflowConfig:
    enabled: bool
    tracking_uri: str | None
    experiment: str | None


class MlflowTracer:
    def __init__(self, config: MlflowConfig) -> None:
        self._config = config
        self._mlflow = _get_mlflow() if config.enabled else None
        self._configured = False

    def _configure(self) -> None:
        if self._configured or not self._mlflow:
            return
        tracking_uri = _normalize_tracking_uri(self._config.tracking_uri)
        if tracking_uri:
            self._mlflow.set_tracking_uri(tracking_uri)
            if tracking_uri.startswith("sqlite:////"):
                # Keep local runs stable and avoid noisy progress logs.
                self._mlflow.set_registry_uri(tracking_uri)
                os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
                os.environ.setdefault("MLFLOW_TRACKING_INSECURE_TLS", "true")
        if self._config.experiment:
            self._mlflow.set_experiment(self._config.experiment)
        self._configured = True

    @contextmanager
    def interaction_run(self, *, run_name: str, tags: dict[str, str]) -> Iterator[object | None]:
        if not self._mlflow:
            yield None
            return
        try:
            self._configure()
            parent_run = self._mlflow.active_run()
            with self._mlflow.start_run(
                run_name=run_name,
                tags=tags,
                nested=bool(parent_run),
            ):
                yield self._mlflow
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("MLflow interaction run disabled due to runtime error: %s", exc)
            yield None

    @contextmanager
    def nested_run(self, *, run_name: str, tags: dict[str, str] | None = None) -> Iterator[object | None]:
        if not self._mlflow:
            yield None
            return
        try:
            self._configure()
            with self._mlflow.start_run(run_name=run_name, nested=True, tags=tags or {}):
                yield self._mlflow
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("MLflow nested run disabled due to runtime error: %s", exc)
            yield None

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
        tracking_uri = _normalize_tracking_uri(self._config.tracking_uri)
        if tracking_uri:
            self._mlflow.set_tracking_uri(tracking_uri)
            if tracking_uri.startswith("sqlite:////"):
                self._mlflow.set_registry_uri(tracking_uri)
                os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
                os.environ.setdefault("MLFLOW_TRACKING_INSECURE_TLS", "true")
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
        try:
            self._configure()
            with self._mlflow.start_span(name=name, span_type=span_type, attributes=attributes) as span:
                if inputs:
                    span.set_inputs(inputs)
                yield span
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("MLflow trace span disabled due to runtime error: %s", exc)
            yield None

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
