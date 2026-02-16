"""Main orchestrator for ai-agent-qbr."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import time
from contextlib import suppress
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from penguiflow.errors import FlowError
from penguiflow.planner import PlannerFinish, PlannerPause, ToolContext

from ai_agent_qbr.config import Config
from ai_agent_qbr.infrastructure.memory_store import InMemoryMemoryStore
from ai_agent_qbr.infrastructure.region_filter import (
    build_context_from_items,
    filter_items_by_region,
)
from ai_agent_qbr.infrastructure.region_verifier import RegionVerifier
from ai_agent_qbr.infrastructure.metric_router import MetricQueryRouter
from ai_agent_qbr.observability.mlflow_logger import (
    MlflowConfig,
    MlflowTrace,
    MlflowTraceConfig,
    MlflowTracer,
)
from ai_agent_qbr.planner import PlannerBundle, build_planner
from ai_agent_qbr.telemetry import AgentTelemetry
from ai_agent_qbr.tools.region_verifier import verify_region_filter
from ai_agent_qbr.models import RegionFilterVerificationArgs
from qbr_agent.application.use_cases import AnswerQuestion, HybridSearchKnowledge
from qbr_agent.infrastructure.factory import InfrastructureBundle, build_infrastructure
from qbr_intelligence.metric_qa import MetricQueryEngine
from qbr_intelligence.metric_qa.intent_llm import build_intent_llm
from qbr_intelligence.schemas.metric_qa import AnswerCitation, MetricAnswer

_LOGGER = logging.getLogger(__name__)

_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|hiya|yo|sup|good\s+(morning|afternoon|evening))(\s+there)?\s*[!.?]*\s*$",
    re.IGNORECASE,
)


def _is_greeting(text: str) -> bool:
    if not text:
        return False
    return bool(_GREETING_RE.match(text))


_FOLLOWUP_MARKERS = {
    "and",
    "also",
    "compare",
    "vs",
    "versus",
    "what about",
    "same",
    "that",
    "those",
    "it",
}


def _looks_like_followup(query: str) -> bool:
    if not query:
        return False
    lowered = query.strip().lower()
    if len(lowered.split()) <= 4:
        return True
    return any(marker in lowered for marker in _FOLLOWUP_MARKERS)


def _planner_event_payload(event: Any) -> tuple[str | None, dict[str, Any]]:
    event_type: str | None = None
    payload: dict[str, Any] = {}
    if isinstance(event, dict):
        event_type = event.get("event_type") or event.get("type")
        raw_payload = event.get("payload") or event.get("extra")
        if isinstance(raw_payload, dict):
            payload = dict(raw_payload)
        return event_type, payload

    if hasattr(event, "event_type"):
        event_type = getattr(event, "event_type")
    if hasattr(event, "extra"):
        raw_extra = getattr(event, "extra", None)
        if isinstance(raw_extra, dict):
            payload.update(raw_extra)
    if hasattr(event, "to_payload"):
        try:
            raw_payload = event.to_payload()
        except Exception:  # noqa: BLE001
            raw_payload = None
        if isinstance(raw_payload, dict):
            merged = dict(raw_payload)
            merged.update(payload)
            payload = merged
    return event_type, payload


def _decode_json_like(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:  # noqa: BLE001
            return value
    return value


def _decode_json_like_recursive(value: Any) -> Any:
    decoded = _decode_json_like(value)
    if isinstance(decoded, str):
        second = _decode_json_like(decoded)
        return second
    return decoded


def _extract_rag_scope_payload(payload: Any) -> dict[str, Any] | None:
    candidate = _decode_json_like_recursive(payload)
    if not isinstance(candidate, dict):
        return None
    debug_payload = candidate.get("debug")
    if isinstance(debug_payload, dict):
        rag_scope = debug_payload.get("rag_scope")
        if isinstance(rag_scope, dict):
            return rag_scope
    direct_scope = candidate.get("rag_scope")
    if isinstance(direct_scope, dict):
        return direct_scope
    for key in ("result", "output", "value", "data"):
        nested = candidate.get(key)
        rag_scope = _extract_rag_scope_payload(nested)
        if rag_scope is not None:
            return rag_scope
    return None


class _ToolIoCollector:
    def __init__(self) -> None:
        self._by_id: dict[str, dict[str, Any]] = {}
        self._ordered: list[dict[str, Any]] = []
        self._sequence = 0

    @property
    def records(self) -> list[dict[str, Any]]:
        return [dict(record) for record in self._ordered]

    def on_event(self, event: Any) -> None:
        event_type, payload = _planner_event_payload(event)
        if event_type not in {"tool_call_start", "tool_call_result", "tool_call_end"}:
            return
        raw_id = payload.get("tool_call_id")
        if raw_id is None:
            return
        tool_call_id = str(raw_id)

        record = self._by_id.get(tool_call_id)
        if record is None:
            record = {
                "index": self._sequence,
                "tool_call_id": tool_call_id,
                "tool_name": str(payload.get("tool_name") or "unknown"),
                "input": None,
                "output": None,
                "status": "started",
            }
            self._sequence += 1
            self._by_id[tool_call_id] = record
            self._ordered.append(record)

        if event_type == "tool_call_start":
            if payload.get("tool_name"):
                record["tool_name"] = str(payload.get("tool_name"))
            if "args_json" in payload:
                record["input"] = _decode_json_like(payload.get("args_json"))
            return

        if event_type == "tool_call_result":
            if "result_json" in payload:
                record["output"] = _decode_json_like(payload.get("result_json"))
            record["status"] = "result"
            return

        if event_type == "tool_call_end":
            if record.get("status") == "started":
                record["status"] = "completed_no_result"
            else:
                record["status"] = "completed"


class _PlannerMlflowEventCollector:
    def __init__(self, trace: MlflowTrace, *, interaction_metadata: dict[str, Any] | None = None) -> None:
        self._trace = trace
        self._tool_io = _ToolIoCollector()
        self._interaction_metadata = interaction_metadata if isinstance(interaction_metadata, dict) else None
        self._tool_spans: dict[str, dict[str, Any]] = {}
        self._llm_span_cm: Any = None
        self._llm_span: object | None = None
        self._llm_call_index = 0
        self._llm_outputs: dict[str, str] = {}
        self._llm_calls: list[dict[str, Any]] = []
        self._llm_chunk_count = 0
        self._llm_last_chunk: dict[str, Any] | None = None
        self._query_metrics_rag_emit_count = 0
        self._streamed_answer_text = ""

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return self._tool_io.records

    @property
    def llm_calls(self) -> list[dict[str, Any]]:
        return list(self._llm_calls)

    @property
    def query_metrics_rag_emitted(self) -> bool:
        return self._query_metrics_rag_emit_count > 0

    @property
    def streamed_answer_text(self) -> str | None:
        text = self._streamed_answer_text.strip()
        return text or None

    def on_event(self, event: Any) -> None:
        event_type, payload = _planner_event_payload(event)
        if not event_type:
            return
        self._tool_io.on_event(event)
        event_type = str(event_type)

        if event_type == "llm_stream_chunk":
            self._on_llm_stream(payload)
            return

        if event_type == "tool_call_start":
            self._close_llm_span()
            self._on_tool_start(payload)
            return

        if event_type == "tool_call_result":
            self._on_tool_result(payload)
            return

        if event_type == "tool_call_end":
            self._on_tool_end(payload)
            return

        if event_type in {"step_complete", "node_end", "iteration_end"}:
            self._close_llm_span()

    def close(self) -> None:
        self._close_llm_span()
        for tool_call_id in list(self._tool_spans):
            self._close_tool_span(tool_call_id)

    def _on_llm_stream(self, payload: dict[str, Any]) -> None:
        if self._llm_span is None:
            self._llm_call_index += 1
            channel = str(payload.get("channel") or "unknown")
            phase = str(payload.get("phase") or "unknown")
            cm = self._trace.span(
                name=f"llm_call_{self._llm_call_index}",
                span_type="LLM",
                attributes={"channel": channel, "phase": phase},
            )
            span = cm.__enter__()
            self._llm_span_cm = cm
            self._llm_span = span
            self._llm_outputs = {}
            self._llm_chunk_count = 0
            self._llm_last_chunk = None
            self._llm_calls.append({"index": self._llm_call_index, "channel": channel, "phase": phase})

        channel = str(payload.get("channel") or "unknown")
        text = payload.get("text") or payload.get("content") or payload.get("delta") or payload.get("message")
        self._llm_chunk_count += 1
        self._llm_last_chunk = {
            "channel": channel,
            "phase": payload.get("phase"),
            "done": bool(payload.get("done")),
            "keys": sorted(payload.keys()),
        }
        if text:
            existing = self._llm_outputs.get(channel, "")
            self._llm_outputs[channel] = f"{existing}{text}"
            if channel == "answer":
                self._streamed_answer_text = f"{self._streamed_answer_text}{text}"

        if bool(payload.get("done")):
            self._close_llm_span()

    def _close_llm_span(self) -> None:
        if self._llm_span_cm is None:
            return
        outputs: dict[str, Any] = dict(self._llm_outputs)
        if not outputs:
            outputs = {"content": "", "note": "No text-like payload found in llm_stream_chunk"}
        outputs["chunk_count"] = self._llm_chunk_count
        if self._llm_last_chunk is not None:
            outputs["last_chunk"] = self._llm_last_chunk
        self._trace.set_outputs(self._llm_span, outputs)
        cm = self._llm_span_cm
        self._llm_span_cm = None
        self._llm_span = None
        self._llm_outputs = {}
        self._llm_chunk_count = 0
        self._llm_last_chunk = None
        with suppress(Exception):
            cm.__exit__(None, None, None)

    def _on_tool_start(self, payload: dict[str, Any]) -> None:
        raw_id = payload.get("tool_call_id")
        if raw_id is None:
            return
        tool_call_id = str(raw_id)
        tool_name = str(payload.get("tool_name") or "unknown")
        tool_input = _decode_json_like(payload.get("args_json"))
        if tool_name == "query_metrics":
            tool_input = self._enrich_query_metrics_input(tool_input)
        cm = self._trace.span(
            name=f"tool:{tool_name}",
            span_type="TOOL",
            attributes={"tool_call_id": tool_call_id, "tool_name": tool_name},
            inputs={"args": tool_input} if tool_input is not None else None,
        )
        span = cm.__enter__()
        self._tool_spans[tool_call_id] = {
            "cm": cm,
            "span": span,
            "result": None,
            "ended": False,
            "rag_emitted": False,
            "tool_name": tool_name,
            "input": tool_input,
        }

    def _on_tool_result(self, payload: dict[str, Any]) -> None:
        raw_id = payload.get("tool_call_id")
        if raw_id is None:
            return
        tool_call_id = str(raw_id)
        span_state = self._tool_spans.get(tool_call_id)
        if not span_state:
            return
        span = span_state.get("span")
        if span is None:
            return
        result_payload: Any = None
        for key in ("result_json", "result", "output", "value"):
            if key in payload:
                result_payload = _decode_json_like_recursive(payload.get(key))
                break
        span_state["result"] = result_payload
        self._trace.set_outputs(
            span,
            {"result": result_payload, "status": "result"},
        )
        if span_state.get("tool_name") == "query_metrics":
            self._emit_query_metrics_rag_span(span_state, result_payload)
        if span_state.get("ended"):
            self._close_tool_span(tool_call_id)

    def _on_tool_end(self, payload: dict[str, Any]) -> None:
        raw_id = payload.get("tool_call_id")
        if raw_id is None:
            return
        tool_call_id = str(raw_id)
        span_state = self._tool_spans.get(tool_call_id)
        if not span_state:
            return
        span_state["ended"] = True
        if span_state.get("tool_name") == "query_metrics" and not span_state.get("rag_emitted"):
            self._emit_query_metrics_rag_span(span_state, span_state.get("result"))
        if span_state.get("result") is not None:
            self._close_tool_span(tool_call_id)

    def _close_tool_span(self, tool_call_id: str) -> None:
        span_state = self._tool_spans.pop(tool_call_id, None)
        if not span_state:
            return
        span = span_state.get("span")
        if span is not None and span_state.get("result") is None:
            self._trace.set_outputs(span, {"status": "ended_without_result"})
        cm = span_state.get("cm")
        if cm is None:
            return
        with suppress(Exception):
            cm.__exit__(None, None, None)

    def _emit_query_metrics_rag_span(self, span_state: dict[str, Any], result_payload: Any) -> None:
        parent_span = span_state.get("span")
        if parent_span is None:
            return
        rag_scope = _extract_rag_scope_payload(result_payload)
        if rag_scope is None and isinstance(self._interaction_metadata, dict):
            fallback_scope = self._interaction_metadata.get("metric_rag_scope")
            if isinstance(fallback_scope, dict):
                rag_scope = fallback_scope
        if rag_scope is None:
            return
        tool_input = span_state.get("input")
        query_value = None
        if isinstance(tool_input, dict):
            query_value = tool_input.get("question")
        with self._trace.span(
            name="query_metrics.rag",
            span_type="RETRIEVER",
            attributes={
                "source_tool": "query_metrics",
                "enabled": bool(rag_scope.get("enabled", False)),
            },
            inputs={
                "query": query_value,
                "prefiltered_doc_ids": rag_scope.get("prefiltered_doc_ids"),
            },
        ) as rag_span:
            self._trace.set_outputs(
                rag_span,
                {
                    "prefiltered_docs": rag_scope.get("prefiltered_docs"),
                    "document_ids": rag_scope.get("document_ids"),
                    "selected_docs": rag_scope.get("selected_docs"),
                    "per_doc_hit_count": rag_scope.get("per_doc_hit_count"),
                    "per_doc_max_score": rag_scope.get("per_doc_max_score"),
                    "retrieved_hit_chunks": rag_scope.get("retrieved_hit_chunks"),
                    "post_rerank_chunks_full": rag_scope.get("post_rerank_chunks_full"),
                    "slide_ids_count": rag_scope.get("slide_ids_count"),
                    "slide_range_count": rag_scope.get("slide_range_count"),
                    "retrieved_hits": rag_scope.get("retrieved_hits"),
                    "post_rerank_hit_count": rag_scope.get("post_rerank_hit_count"),
                    "pre_rerank_hit_count": rag_scope.get("pre_rerank_hit_count"),
                },
            )
        span_state["rag_emitted"] = True
        self._query_metrics_rag_emit_count += 1

    def _enrich_query_metrics_input(self, tool_input: Any) -> Any:
        if not isinstance(tool_input, dict):
            return tool_input
        if tool_input.get("intent"):
            return tool_input

        latest_intent = self._latest_intent_from_tool_outputs()
        if latest_intent is None:
            return tool_input

        enriched = dict(tool_input)
        enriched["intent"] = latest_intent
        enriched["_intent_source"] = "mlflow_enriched_from_prior_tool_output"
        return enriched

    def _latest_intent_from_tool_outputs(self) -> dict[str, Any] | None:
        for record in reversed(self._tool_io.records):
            if not isinstance(record, dict):
                continue
            if record.get("tool_name") not in {"refine_metric_intent", "resolve_metric_intent"}:
                continue
            output = record.get("output")
            if not isinstance(output, dict):
                continue
            intent = output.get("intent")
            if isinstance(intent, dict):
                return intent
        return None


def _emit_query_metrics_rag_fallback_span(
    trace: MlflowTrace,
    *,
    query: str | None,
    rag_scope: dict[str, Any],
) -> None:
    with trace.span(
        name="query_metrics.rag",
        span_type="RETRIEVER",
        attributes={
            "source_tool": "query_metrics",
            "enabled": bool(rag_scope.get("enabled", False)),
            "fallback_emission": True,
        },
        inputs={
            "query": query,
            "prefiltered_doc_ids": rag_scope.get("prefiltered_doc_ids"),
        },
    ) as rag_span:
        trace.set_outputs(
            rag_span,
            {
                "prefiltered_docs": rag_scope.get("prefiltered_docs"),
                "document_ids": rag_scope.get("document_ids"),
                "selected_docs": rag_scope.get("selected_docs"),
                "per_doc_hit_count": rag_scope.get("per_doc_hit_count"),
                "per_doc_max_score": rag_scope.get("per_doc_max_score"),
                "retrieved_hit_chunks": rag_scope.get("retrieved_hit_chunks"),
                "post_rerank_chunks_full": rag_scope.get("post_rerank_chunks_full"),
                "slide_ids_count": rag_scope.get("slide_ids_count"),
                "slide_range_count": rag_scope.get("slide_range_count"),
                "retrieved_hits": rag_scope.get("retrieved_hits"),
                "post_rerank_hit_count": rag_scope.get("post_rerank_hit_count"),
                "pre_rerank_hit_count": rag_scope.get("pre_rerank_hit_count"),
            },
        )


def _extract_rag_scope_from_interaction_metadata(
    interaction_metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(interaction_metadata, dict):
        return None
    direct_scope = interaction_metadata.get("metric_rag_scope")
    if isinstance(direct_scope, dict):
        return direct_scope
    metric_answer = interaction_metadata.get("metric_answer")
    rag_scope = _extract_rag_scope_payload(metric_answer)
    if isinstance(rag_scope, dict):
        return rag_scope
    return None



class AiAgentQbrFlowError(RuntimeError):
    """Raised when planner execution fails."""

    def __init__(self, flow_error: FlowError | str) -> None:
        message = flow_error.message if isinstance(flow_error, FlowError) else str(flow_error)
        super().__init__(message)
        self.flow_error = flow_error


@dataclass
class AgentResponse:
    """Response envelope returned by the orchestrator."""

    answer: str | None
    trace_id: str
    metadata: dict[str, Any] | None = None
    artifacts: dict[str, Any] | None = None


def _coerce_metric_answer(payload: Any) -> MetricAnswer | None:
    if payload is None:
        return None
    if isinstance(payload, MetricAnswer):
        return payload
    if isinstance(payload, Mapping):
        try:
            return MetricAnswer.model_validate(payload)
        except Exception:
            return None
    return None


def _format_metric_value(value: float | None, unit: str | None) -> str:
    if value is None:
        return "unknown"
    if unit == "percent":
        return f"{value:.2f}%"
    if unit == "currency":
        return f"${value:,.2f}"
    return f"{value:,.2f}"


def _build_metric_grid_artifact(answer: MetricAnswer | None) -> dict[str, Any] | None:
    if answer is None:
        return None
    rows = answer.table_data or []
    if not rows and answer.data:
        rows = [row.model_dump() if hasattr(row, "model_dump") else dict(row) for row in answer.data]
    if len(rows) <= 1:
        return None
    grid_rows: list[dict[str, Any]] = []
    metric_title = None
    for row in rows:
        metric = row.get("metric") or "metric"
        if metric_title is None and metric:
            metric_title = str(metric)
        value = _format_metric_value(row.get("value"), row.get("unit"))
        period = row.get("period")
        region = row.get("region")
        client = row.get("client")
        metric_url = row.get("slide_url") or row.get("document_url")
        grid_rows.append(
            {
                "value": value,
                "period": period,
                "region": region,
                "client": client,
                "metric_url": metric_url,
            }
        )
    return {
        "type": "datagrid",
        "title": metric_title or "Metric Results",
        "columns": [
            {"field": "value", "header": "Value", "format": "text"},
            {"field": "period", "header": "Time Period", "format": "text"},
            {"field": "region", "header": "Region", "format": "text"},
            {"field": "client", "header": "Client", "format": "text"},
        ],
        "rows": grid_rows,
    }


def _extract_answer(payload: Any) -> str | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        return payload.strip() or None
    if isinstance(payload, Mapping):
        candidate = _extract_answer_from_mapping(payload)
        if candidate is not None:
            return str(candidate)
        return str(payload)

    for attr in (
        "raw_answer",
        "answer",
        "text",
        "content",
        "message",
        "greeting",
        "response",
        "result",
        "final_answer",
    ):
        if hasattr(payload, attr):
            value = getattr(payload, attr)
            if isinstance(value, str) and value.strip():
                return value
            if value is not None and not isinstance(value, str):
                return str(value)

    if hasattr(payload, "args"):
        nested_args = getattr(payload, "args")
        if isinstance(nested_args, Mapping):
            candidate = _extract_answer_from_mapping(nested_args)
            if candidate is not None:
                return str(candidate)
    return str(payload)


def _extract_answer_from_mapping(payload: Mapping[str, Any]) -> str | None:
    raw_answer = payload.get("raw_answer")
    if isinstance(raw_answer, str) and raw_answer.strip():
        return raw_answer
    if raw_answer is not None and not isinstance(raw_answer, str):
        return str(raw_answer)

    nested_args = payload.get("args")
    if isinstance(nested_args, Mapping):
        nested_raw = nested_args.get("raw_answer")
        if isinstance(nested_raw, str) and nested_raw.strip():
            return nested_raw
        if nested_raw is not None and not isinstance(nested_raw, str):
            return str(nested_raw)

    for key in (
        "answer",
        "text",
        "content",
        "message",
        "greeting",
        "response",
        "result",
        "final_answer",
    ):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if value is not None and not isinstance(value, str):
            return str(value)

    if isinstance(nested_args, Mapping):
        for key in (
            "answer",
            "text",
            "content",
            "message",
            "greeting",
            "response",
            "result",
            "final_answer",
        ):
            value = nested_args.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if value is not None and not isinstance(value, str):
                return str(value)
    return None


def _format_metric_answer(answer: MetricAnswer) -> str:
    def _format_source_from_row(row: Mapping[str, Any]) -> str | None:
        doc_name = row.get("document_name")
        doc_id = row.get("document_id")
        slide_title = row.get("slide_title")
        slide_number = row.get("slide_number")
        slide_id = row.get("slide_id")
        source_label = doc_name or (f"doc {doc_id}" if doc_id is not None else None)
        if source_label is None:
            return None
        if slide_title:
            source_label = f"{source_label} - {slide_title}"
        elif slide_number is not None:
            source_label = f"{source_label} slide {slide_number}"
        elif slide_id is not None:
            source_label = f"{source_label} slide {slide_id}"
        source_url = row.get("slide_url") or row.get("document_url")
        if source_url:
            return f"[{source_label}]({source_url})"
        return source_label

    text = answer.summary_text.strip()
    details_rows = answer.table_data
    if not details_rows and answer.data:
        details_rows = [row.model_dump() if hasattr(row, "model_dump") else dict(row) for row in answer.data]
    if details_rows:
        has_doc_context = any(row.get("document_name") or row.get("document_url") for row in details_rows)
        if has_doc_context:
            metric_name = details_rows[0].get("metric") or "Metric"
            lines = ["", f"Metric: {metric_name}"]
            grouped: dict[tuple[str | None, str | None, int | None, int | None, str | None], list[dict]] = {}
            for row in details_rows:
                key = (
                    row.get("document_name"),
                    row.get("document_url"),
                    row.get("slide_number"),
                    row.get("slide_id"),
                    row.get("slide_title"),
                )
                grouped.setdefault(key, []).append(row)

            doc_order: list[tuple[str | None, str | None]] = []
            for row in details_rows:
                doc_key = (row.get("document_name"), row.get("document_url"))
                if doc_key not in doc_order:
                    doc_order.append(doc_key)

            def _slide_sort_key(item: tuple) -> tuple:
                _, _, slide_number, slide_id, _ = item[0]
                return (slide_number or 0, slide_id or 0)

            for doc_name, doc_url in doc_order:
                doc_label = doc_name or "Document"
                doc_line = f"Document: {doc_label}"
                if doc_url:
                    doc_line = f"{doc_line} (`{doc_url}`)"
                lines.append(doc_line)

                slide_items = [
                    (key, rows)
                    for key, rows in grouped.items()
                    if key[0] == doc_name and key[1] == doc_url
                ]
                for (key, rows) in sorted(slide_items, key=_slide_sort_key):
                    _, _, slide_number, slide_id, slide_title = key
                    slide_label = "Slide"
                    if slide_title:
                        slide_label = f"{slide_label}: {slide_title}"
                    elif slide_number:
                        slide_label = f"{slide_label} {slide_number}"
                    elif slide_id:
                        slide_label = f"{slide_label} {slide_id}"
                    lines.append(f"- {slide_label}")

                    for row in rows[:4]:
                        value = row.get("value")
                        unit = row.get("unit") or ""
                        value_text = f"{value} {unit}".strip()
                        context = row.get("snippet") or row.get("llm_context_label")
                        context_text = f" — {context}" if context else ""
                        source_text = _format_source_from_row(row)
                        source_suffix = f" | {source_text}" if source_text else ""
                        lines.append(f"  - {value_text}{context_text}{source_suffix}")
            text = f"{text}\n" + "\n".join(lines)
        else:
            lines = ["", "Details:"]
            for row in details_rows[:5]:
                metric = row.get("metric") or "metric"
                period = row.get("period") or "period"
                value = row.get("value")
                unit = row.get("unit") or ""
                context_label = row.get("llm_context_label")
                context_text = f" — {context_label}" if context_label else ""
                lines.append(f"- {metric}: {value} {unit} ({period}){context_text}")
            text = f"{text}\n" + "\n".join(lines)
    if answer.citations:
        doc_sources: list[str] = []
        seen_docs: set[tuple[str, str | None]] = set()
        for citation in answer.citations:
            doc_label = citation.document_name or f"doc {citation.document_id}"
            doc_url = citation.document_url
            key = (doc_label, doc_url)
            if key in seen_docs:
                continue
            seen_docs.add(key)
            if doc_url:
                doc_sources.append(f"[{doc_label}]({doc_url})")
            else:
                doc_sources.append(doc_label)
        sources = ", ".join(doc_sources)
        text = f"{text}\n\nSources: {sources}"
    return text


def _tool_calls_include_search(tool_calls: list[dict[str, Any]]) -> bool:
    for call in tool_calls:
        tool_name = str(call.get("tool_name") or "").strip()
        if tool_name in {"search_documents", "search_qbr"}:
            return True
    return False


def _format_rag_slide_refs(qbr_citations: list[dict[str, Any]]) -> list[str]:
    refs: list[str] = []
    seen: set[tuple[int | None, int | None, int | None]] = set()
    for citation in qbr_citations:
        if not isinstance(citation, dict):
            continue
        doc_id_raw = citation.get("document_id")
        try:
            doc_id = int(doc_id_raw) if doc_id_raw is not None else None
        except (TypeError, ValueError):
            doc_id = None
        start_raw = citation.get("start_slide")
        end_raw = citation.get("end_slide")
        try:
            start = int(start_raw) if start_raw is not None else None
        except (TypeError, ValueError):
            start = None
        try:
            end = int(end_raw) if end_raw is not None else None
        except (TypeError, ValueError):
            end = None
        key = (doc_id, start, end)
        if key in seen:
            continue
        seen.add(key)

        doc_label = f"doc {doc_id}" if doc_id is not None else "doc ?"
        if start is None and end is None:
            slide_label = "slide ?"
        elif start is None:
            slide_label = f"slides ?-{end}"
        elif end is None or end == start:
            slide_label = f"slide {start}"
        else:
            slide_label = f"slides {start}-{end}"
        refs.append(f"{doc_label} {slide_label}")
    return refs


def _append_search_slide_refs_to_answer(
    answer_text: str,
    *,
    tool_calls: list[dict[str, Any]],
    qbr_citations: list[dict[str, Any]],
) -> str:
    if not answer_text.strip():
        return answer_text
    if not _tool_calls_include_search(tool_calls):
        return answer_text
    slide_refs = _format_rag_slide_refs(qbr_citations)
    if not slide_refs:
        return answer_text
    if "slides:" in answer_text.lower():
        return answer_text
    return f"{answer_text}\n\nSlides: {', '.join(slide_refs)}"


def _strip_trailing_sources_block(text: str) -> str:
    if not text:
        return text
    lowered = text.lower()
    markers = ("\n**sources:**", "\nsources:")
    cut_at = -1
    for marker in markers:
        idx = lowered.rfind(marker)
        if idx > cut_at:
            cut_at = idx
    if cut_at >= 0:
        return text[:cut_at].rstrip()
    return text


def _query_metrics_doc_sources(tool_calls: list[dict[str, Any]]) -> list[str]:
    seen: set[tuple[str, str | None]] = set()
    ordered: list[tuple[str, str | None]] = []
    for call in tool_calls:
        if str(call.get("tool_name") or "").strip() != "query_metrics":
            continue
        output = call.get("output")
        if not isinstance(output, Mapping):
            continue
        citations = output.get("citations")
        if isinstance(citations, list):
            for citation in citations:
                if not isinstance(citation, Mapping):
                    continue
                doc_name = citation.get("document_name")
                doc_id = citation.get("document_id")
                doc_url = citation.get("document_url")
                label = str(doc_name).strip() if doc_name else (f"doc {doc_id}" if doc_id is not None else "")
                if not label:
                    continue
                key = (label, str(doc_url).strip() if isinstance(doc_url, str) and doc_url.strip() else None)
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(key)
        table_data = output.get("table_data")
        if isinstance(table_data, list):
            for row in table_data:
                if not isinstance(row, Mapping):
                    continue
                doc_name = row.get("document_name")
                doc_id = row.get("document_id")
                doc_url = row.get("document_url")
                label = str(doc_name).strip() if doc_name else (f"doc {doc_id}" if doc_id is not None else "")
                if not label:
                    continue
                key = (label, str(doc_url).strip() if isinstance(doc_url, str) and doc_url.strip() else None)
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(key)
    rendered: list[str] = []
    for label, url in ordered:
        if url:
            rendered.append(f"[{label}]({url})")
        else:
            rendered.append(label)
    return rendered


def _replace_slide_number_refs_with_titles(
    text: str,
    *,
    metric_answer: MetricAnswer | None,
) -> str:
    if not text or metric_answer is None:
        return text
    rows = metric_answer.table_data or []
    if not rows:
        return text

    updated = text
    replacements: list[tuple[str, str]] = []
    for row in rows:
        doc_name = row.get("document_name")
        slide_number = row.get("slide_number")
        slide_title = row.get("slide_title")
        if not doc_name or slide_number is None or not slide_title:
            continue
        source_pattern = rf"{re.escape(str(doc_name))}\s+slide\s+{re.escape(str(slide_number))}\b"
        source_label = f"{doc_name} - {slide_title}"
        replacements.append((source_pattern, source_label))

    if not replacements:
        return text

    # Replace longer patterns first to avoid partial collisions for similarly named decks.
    replacements.sort(key=lambda item: len(item[0]), reverse=True)
    for pattern, replacement in replacements:
        updated = re.sub(pattern, replacement, updated)
    return updated


def _finalize_answer_text(
    *,
    output_protocol: str,
    extracted_answer_text: str | None,
    metric_answer: MetricAnswer | None,
    streamed_answer_text: str | None,
    tool_calls: list[dict[str, Any]],
    qbr_citations: list[dict[str, Any]],
) -> str | None:
    query_metrics_call_count = sum(
        1 for call in tool_calls if str(call.get("tool_name") or "").strip() == "query_metrics"
    )
    if output_protocol == "agui" and streamed_answer_text:
        if metric_answer is not None and query_metrics_call_count == 1:
            return _format_metric_answer(metric_answer)
        text = _replace_slide_number_refs_with_titles(
            streamed_answer_text,
            metric_answer=metric_answer,
        )
        if query_metrics_call_count > 1:
            sources = _query_metrics_doc_sources(tool_calls)
            if sources:
                base = _strip_trailing_sources_block(text)
                return f"{base}\n\nSources: {', '.join(sources)}"
        return text
    if metric_answer is not None:
        return _format_metric_answer(metric_answer)
    base_text = extracted_answer_text or ""
    if not base_text.strip():
        return None
    return _append_search_slide_refs_to_answer(
        base_text,
        tool_calls=tool_calls,
        qbr_citations=qbr_citations,
    )


def _make_tool_context(payload: dict[str, Any]) -> Any:
    """Create a tool context compatible with PenguiFlow Protocols."""
    try:
        return ToolContext(tool_context=payload)
    except TypeError:
        return type("ToolContext", (), {"tool_context": payload})()


class AiAgentQbrOrchestrator:
    """Orchestrator that coordinates planner execution and QBR retrieval."""

    def __init__(
        self,
        config: Config,
        *,
        memory_store: Any | None = None,
        telemetry: AgentTelemetry | None = None,
        planner: Any | None = None,
        infrastructure: InfrastructureBundle | None = None,
    ) -> None:
        self._config = config
        self._memory = memory_store or InMemoryMemoryStore(
            max_turns=config.short_term_memory_full_zone_turns,
            retrieval_turns=config.short_term_memory_full_zone_turns,
        )
        self._telemetry = telemetry or AgentTelemetry(
            flow_name="ai-agent-qbr",
            logger=_LOGGER,
        )
        if planner is None:
            planner_bundle: PlannerBundle = build_planner(
                config,
                event_callback=self._telemetry.record_planner_event,
            )
            self._planner = planner_bundle.planner
        else:
            self._planner = planner

        self._started = True
        self._session_cache: dict[tuple[str, str, str], Mapping[str, Any]] = {}
        self._session_started: set[tuple[str, str, str]] = set()
        self._recent_turns: dict[tuple[str, str, str], list[dict[str, str]]] = {}
        self._recent_turns_limit = config.short_term_memory_full_zone_turns
        self._infra_bundle = infrastructure
        self._infra_lock = asyncio.Lock()
        self._region_verifier: RegionVerifier | None = None
        self._metric_router = MetricQueryRouter()
        self._metric_query_engine: MetricQueryEngine | None = None
        self._mlflow_tracer = MlflowTracer(
            MlflowConfig(
                enabled=config.mlflow_enabled,
                tracking_uri=config.mlflow_tracking_uri,
                experiment=config.mlflow_experiment,
            )
        )
        self._mlflow_trace = MlflowTrace(
            MlflowTraceConfig(
                enabled=config.mlflow_tracing_enabled,
                tracking_uri=config.mlflow_tracking_uri,
                experiment=config.mlflow_experiment,
            )
        )

    async def _get_infrastructure(self) -> InfrastructureBundle:
        if self._infra_bundle is not None:
            return self._infra_bundle
        async with self._infra_lock:
            if self._infra_bundle is None:
                self._infra_bundle = await build_infrastructure(
                    database_url=self._config.database_url,
                    storage_backend=self._config.storage_backend,
                    vector_backend=self._config.vector_backend,
                    embeddings_backend=self._config.embeddings_backend,
                    embeddings_model=self._config.embeddings_model,
                    embeddings_normalize=self._config.embeddings_normalize,
                    text_search_backend=self._config.text_search_backend,
                    rerank_backend=self._config.rerank_backend,
                    rerank_model=self._config.rerank_model,
                    rerank_max_length=self._config.rerank_max_length,
                    faiss_dir=self._config.faiss_dir,
                    faiss_normalize=self._config.faiss_normalize,
                )
        return self._infra_bundle

    async def _get_metric_query_engine(self) -> MetricQueryEngine:
        if self._metric_query_engine is not None:
            return self._metric_query_engine
        infra = await self._get_infrastructure()
        intent_llm = None
        if self._config.llm_intent_enabled and not self._config.use_stub_llm:
            model_name = self._config.llm_model
            if not model_name or model_name == "stub-llm":
                if self._config.llm_model_name:
                    model_name = f"databricks/{self._config.llm_model_name}"
                else:
                    model_name = None
            if model_name and model_name != "stub-llm":
                intent_llm = build_intent_llm(
                    model=model_name,
                    max_tokens=min(512, self._config.llm_max_tokens),
                    temperature=0.0,
                )
        self._metric_query_engine = MetricQueryEngine(
            database_url=self._config.database_url,
            embeddings_provider=infra.embeddings,
            llm_intent_enabled=self._config.llm_intent_enabled,
            llm_answer_enabled=self._config.llm_answer_enabled,
            llm_max_calls_per_query=self._config.llm_max_calls_per_query,
            intent_llm=intent_llm,
        )
        return self._metric_query_engine

    def _append_recent_turn(
        self,
        *,
        tenant_id: str,
        user_id: str,
        session_id: str,
        user_prompt: str,
        agent_response: str,
    ) -> None:
        session_key = (tenant_id, user_id, session_id)
        turns = self._recent_turns.setdefault(session_key, [])
        turns.append({"user": user_prompt, "assistant": agent_response})
        if self._recent_turns_limit and len(turns) > self._recent_turns_limit:
            del turns[: len(turns) - self._recent_turns_limit]

    async def execute(
        self,
        query: str,
        *,
        tenant_id: str,
        user_id: str,
        session_id: str,
    ) -> AgentResponse:
        trace_id = secrets.token_hex(8)
        session_key = (tenant_id, user_id, session_id)
        mlflow_tags = {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "session_id": session_id,
            "trace_id": trace_id,
            "output_protocol": self._config.output_protocol,
        }

        with self._mlflow_tracer.interaction_run(
            run_name=f"interaction-{trace_id}",
            tags=mlflow_tags,
        ) as mlflow_run, self._mlflow_trace.span(
            name="interaction",
            span_type="CHAIN",
            attributes=mlflow_tags,
            inputs={"query": query},
        ) as interaction_span:
            if mlflow_run:
                mlflow_run.log_param("query", query)
                mlflow_run.log_param("retrieval_top_k", self._config.retrieval_top_k)
                mlflow_run.log_param("text_search_backend", self._config.text_search_backend)
                mlflow_run.log_param("retrieval_text_weight", self._config.retrieval_text_weight)
                mlflow_run.log_param("retrieval_vector_weight", self._config.retrieval_vector_weight)
                mlflow_run.log_param(
                    "retrieval_candidate_multiplier",
                    self._config.retrieval_candidate_multiplier,
                )
                mlflow_run.log_param("rerank_backend", self._config.rerank_backend)
                mlflow_run.log_param("rerank_model", self._config.rerank_model)
                mlflow_run.log_param("rerank_top_n", self._config.rerank_top_n)
                mlflow_run.log_param("retrieval_mmr_lambda", self._config.retrieval_mmr_lambda)
                mlflow_run.log_param(
                    "retrieval_max_chunks_per_doc",
                    self._config.retrieval_max_chunks_per_doc,
                )
                self._mlflow_tracer.log_text(mlflow_run, "query", query)

            if session_key not in self._session_started:
                _LOGGER.info("Session not started for %s; starting now", session_id)
                await self.start_session(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    session_id=session_id,
                )

            if _is_greeting(query):
                answer_text = "Hi!"
                await self._memory.ingest_interaction(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    session_id=session_id,
                    user_prompt=query,
                    agent_response=answer_text,
                    metadata={},
                )
                self._append_recent_turn(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    session_id=session_id,
                    user_prompt=query,
                    agent_response=answer_text,
                )
                self._mlflow_trace.set_outputs(
                    interaction_span,
                    {
                        "answer": answer_text,
                        "trace_id": trace_id,
                    },
                )
                if mlflow_run:
                    self._mlflow_tracer.log_text(mlflow_run, "final_answer", answer_text)
                return AgentResponse(
                    answer=answer_text,
                    trace_id=trace_id,
                    metadata={},
                )


            region_result = None
            region_focus = None
            if self._config.region_verifier_enabled:
                if self._region_verifier is None:
                    self._region_verifier = RegionVerifier(self._config)
                region_result = await verify_region_filter(
                    RegionFilterVerificationArgs(question=query),
                    _make_tool_context(
                        {
                            "status_publisher": self._telemetry.publish_status,
                            "region_verifier": self._region_verifier,
                        }
                    ),
                )
                region_focus = region_result.region_focus

            infra = await self._get_infrastructure()
            search_use_case = HybridSearchKnowledge(
                repository=infra.repository,
                vector_index=infra.vector_index,
                embeddings=infra.embeddings,
                reranker=infra.reranker,
                text_weight=self._config.retrieval_text_weight,
                vector_weight=self._config.retrieval_vector_weight,
                candidate_multiplier=self._config.retrieval_candidate_multiplier,
                rerank_top_n=self._config.rerank_top_n,
                mmr_lambda=self._config.retrieval_mmr_lambda,
                max_chunks_per_doc=self._config.retrieval_max_chunks_per_doc,
            )
            filtered_citations: list = []
            filtered_context = ""
            if self._config.retrieval_enabled:
                answer_use_case = AnswerQuestion(
                    repository=infra.repository,
                    vector_index=infra.vector_index,
                    embeddings=infra.embeddings,
                    reranker=infra.reranker,
                    use_hybrid=True,
                    text_weight=self._config.retrieval_text_weight,
                    vector_weight=self._config.retrieval_vector_weight,
                    candidate_multiplier=self._config.retrieval_candidate_multiplier,
                    include_document_path=self._config.retrieval_include_document_path,
                    rerank_top_n=self._config.rerank_top_n,
                    mmr_lambda=self._config.retrieval_mmr_lambda,
                    max_chunks_per_doc=self._config.retrieval_max_chunks_per_doc,
                )
                with self._mlflow_trace.span(
                    name="retrieval",
                    span_type="RETRIEVER",
                    attributes={
                        "top_k": self._config.retrieval_top_k,
                        "min_score": self._config.retrieval_min_score,
                        "text_search_backend": self._config.text_search_backend,
                        "text_weight": self._config.retrieval_text_weight,
                        "vector_weight": self._config.retrieval_vector_weight,
                        "rerank_top_n": self._config.rerank_top_n,
                        "mmr_lambda": self._config.retrieval_mmr_lambda,
                        "max_chunks_per_doc": self._config.retrieval_max_chunks_per_doc,
                    },
                    inputs={"query": query},
                ) as retrieval_span:
                    answer_context = await answer_use_case.execute(
                        query=query,
                        top_k=self._config.retrieval_top_k,
                        min_score=self._config.retrieval_min_score,
                    )

                    if region_focus in {"us", "emea"}:
                        filtered_citations = filter_items_by_region(
                            answer_context.citations, region_focus
                        )
                        filtered_context = (
                            build_context_from_items(filtered_citations)
                            if filtered_citations
                            else ""
                        )
                    else:
                        filtered_citations = list(answer_context.citations)
                        filtered_context = answer_context.context or ""

                    self._mlflow_trace.set_outputs(
                        retrieval_span,
                        {
                            "qbr_context": filtered_context,
                            "citations": [
                                {
                                    "chunk_id": result.chunk.chunk_id.value,
                                    "document_id": result.chunk.document_id.value,
                                    "score": result.score.value,
                                    "start_slide": result.chunk.start_slide,
                                    "end_slide": result.chunk.end_slide,
                                    "content": result.chunk.content,
                                }
                                for result in filtered_citations
                            ],
                            "retrieval_debug": answer_use_case.last_debug or {},
                        },
                    )

                with self._mlflow_tracer.nested_run(
                    run_name="retrieval",
                    tags={"trace_id": trace_id},
                ) as retrieval_run:
                    if retrieval_run:
                        retrieval_run.log_param("citation_count", len(filtered_citations))
                        self._mlflow_tracer.log_text(
                            retrieval_run,
                            "qbr_context",
                            filtered_context or "",
                        )
                        self._mlflow_tracer.log_json(
                            retrieval_run,
                            "citations",
                            [
                                {
                                    "chunk_id": result.chunk.chunk_id.value,
                                    "document_id": result.chunk.document_id.value,
                                    "score": result.score.value,
                                    "start_slide": result.chunk.start_slide,
                                    "end_slide": result.chunk.end_slide,
                                    "content": result.chunk.content,
                                }
                                for result in filtered_citations
                            ],
                        )
                        if answer_use_case.last_debug:
                            self._mlflow_tracer.log_json(
                                retrieval_run,
                                "retrieval_debug",
                                answer_use_case.last_debug,
                            )

            conscious = self._session_cache.get(
                session_key, {"conscious": [], "token_estimate": 0}
            )
            last_metric_intent = None
            if hasattr(self._memory, "get_last_metric_intent"):
                try:
                    candidate = self._memory.get_last_metric_intent(
                        tenant_id=tenant_id,
                        user_id=user_id,
                        session_id=session_id,
                    )
                except Exception:  # noqa: BLE001
                    candidate = None
                if isinstance(candidate, dict):
                    last_metric_intent = candidate
            llm_context = {
                "conscious_memories": list(conscious.get("conscious", [])),
                "conversation_memory": {
                    "recent_turns": list(self._recent_turns.get(session_key, []))
                },
                "last_metric_intent": last_metric_intent,
                "qbr_context": filtered_context,
                "region_verification": (
                    region_result.model_dump() if region_result is not None else None
                ),
                "qbr_citations": [
                    {
                        "chunk_id": result.chunk.chunk_id.value,
                        "document_id": result.chunk.document_id.value,
                        "score": result.score.value,
                        "start_slide": result.chunk.start_slide,
                        "end_slide": result.chunk.end_slide,
                    }
                    for result in filtered_citations
                ],
            }
            tool_context = {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "session_id": session_id,
                "original_query": query,
                "trace_id": trace_id,
                "mlflow_trace": self._mlflow_trace,
                "status_publisher": self._telemetry.publish_status,
                "output_protocol": self._config.output_protocol,
                "qbr_search_use_case": search_use_case,
                "qbr_answer_context": filtered_context,
                "region_focus": region_focus,
                "region_verifier": self._region_verifier,
                "retrieval_top_k": self._config.retrieval_top_k,
                "retrieval_min_score": self._config.retrieval_min_score,
                "retrieval_include_document_path": self._config.retrieval_include_document_path,
                "comparison_top_docs": self._config.comparison_top_docs,
                "comparison_per_doc_k": self._config.comparison_per_doc_k,
            }
            interaction_metadata: dict[str, Any] = {}
            tool_context["interaction_metadata"] = interaction_metadata
            tool_context["metric_query_engine"] = await self._get_metric_query_engine()
            if self._config.comparison_stage1_top_k is not None:
                tool_context["comparison_stage1_top_k"] = self._config.comparison_stage1_top_k
            planner_event_collector = _PlannerMlflowEventCollector(
                self._mlflow_trace,
                interaction_metadata=interaction_metadata,
            )

            with self._mlflow_trace.span(
                name="planner",
                span_type="LLM",
                inputs={
                    "query": query,
                    "qbr_context": filtered_context,
                    "qbr_citations": llm_context["qbr_citations"],
                },
            ) as planner_span, self._telemetry.subscribe(
                event_callback=planner_event_collector.on_event
            ):
                planner_started = time.perf_counter()
                _LOGGER.info("Planner start trace_id=%s session_id=%s", trace_id, session_id)
                try:
                    result = await self._planner.run(
                        query=query,
                        llm_context=llm_context,
                        tool_context=tool_context,
                    )
                    _LOGGER.info(
                        "Planner finished trace_id=%s session_id=%s elapsed_s=%.2f",
                        trace_id,
                        session_id,
                        time.perf_counter() - planner_started,
                    )
                finally:
                    planner_event_collector.close()

            if isinstance(result, PlannerPause):
                raise AiAgentQbrFlowError("Planner paused unexpectedly")

            if not isinstance(result, PlannerFinish):
                raise AiAgentQbrFlowError("Planner did not finish successfully")

            payload: Any = result.payload
            answer_text = _extract_answer(payload)
            tool_calls = planner_event_collector.tool_calls
            llm_calls = planner_event_collector.llm_calls
            metric_answer = None
            if isinstance(interaction_metadata, dict):
                metric_payload = interaction_metadata.get("metric_answer")
                metric_answer = _coerce_metric_answer(metric_payload)
            if metric_answer is None:
                metric_answer = _coerce_metric_answer(payload)
            answer_text = _finalize_answer_text(
                output_protocol=self._config.output_protocol,
                extracted_answer_text=answer_text,
                metric_answer=metric_answer,
                streamed_answer_text=planner_event_collector.streamed_answer_text,
                tool_calls=tool_calls,
                qbr_citations=llm_context.get("qbr_citations", []),
            )
            rag_scope_payload = _extract_rag_scope_from_interaction_metadata(interaction_metadata)
            if (
                isinstance(rag_scope_payload, dict)
                and not planner_event_collector.query_metrics_rag_emitted
            ):
                _LOGGER.info(
                    "MLFLOW_RAG_FALLBACK_EMIT trace_id=%s source=interaction_metadata",
                    trace_id,
                )
                _emit_query_metrics_rag_fallback_span(
                    self._mlflow_trace,
                    query=query,
                    rag_scope=rag_scope_payload,
                )
            elif not planner_event_collector.query_metrics_rag_emitted:
                _LOGGER.warning(
                    "MLFLOW_RAG_FALLBACK_MISSING trace_id=%s keys=%s",
                    trace_id,
                    sorted(interaction_metadata.keys()) if isinstance(interaction_metadata, dict) else [],
                )

            if answer_text:
                await self._memory.ingest_interaction(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    session_id=session_id,
                    user_prompt=query,
                    agent_response=answer_text,
                    metadata=interaction_metadata,
                )
                self._append_recent_turn(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    session_id=session_id,
                    user_prompt=query,
                    agent_response=answer_text,
                )
                self._mlflow_trace.set_outputs(
                    planner_span,
                    {
                        "answer": answer_text,
                        "payload": payload,
                        "tool_calls": tool_calls,
                        "llm_calls": llm_calls,
                    },
                )
                self._mlflow_trace.set_outputs(
                    interaction_span,
                    {
                        "answer": answer_text,
                        "trace_id": trace_id,
                    },
                )
            else:
                self._mlflow_trace.set_outputs(
                    planner_span,
                    {
                        "answer": "",
                        "payload": payload,
                        "tool_calls": tool_calls,
                        "llm_calls": llm_calls,
                    },
                )
            with self._mlflow_tracer.nested_run(
                run_name="planner",
                tags={"trace_id": trace_id},
            ) as planner_run:
                if answer_text:
                    self._mlflow_tracer.log_text(
                        planner_run,
                        "final_answer",
                        answer_text,
                    )
                self._mlflow_tracer.log_json(
                    planner_run,
                    "planner_payload",
                    payload,
                )
                self._mlflow_tracer.log_json(
                    planner_run,
                    "tool_calls",
                    tool_calls,
                )
                self._mlflow_tracer.log_json(
                    planner_run,
                    "llm_calls",
                    llm_calls,
                )

            if metric_answer is None:
                _LOGGER.info("metric_grid trace_id=%s status=missing_metric_answer", trace_id)
            else:
                table_len = len(metric_answer.table_data or [])
                data_len = len(metric_answer.data or [])
                _LOGGER.info(
                    "metric_grid trace_id=%s status=answer_loaded table_len=%s data_len=%s",
                    trace_id,
                    table_len,
                    data_len,
                )
            artifacts = None
            if self._config.rich_output_enabled and (
                "datagrid" in (self._config.rich_output_allowlist or [])
            ):
                artifacts = _build_metric_grid_artifact(metric_answer)
            if artifacts is None:
                _LOGGER.info("metric_grid trace_id=%s status=no_artifact", trace_id)
            else:
                _LOGGER.info(
                    "metric_grid trace_id=%s status=artifact type=%s rows=%s",
                    trace_id,
                    artifacts.get("type") if isinstance(artifacts, dict) else None,
                    len(artifacts.get("rows") or []) if isinstance(artifacts, dict) else None,
                )

            return AgentResponse(
                answer=answer_text,
                trace_id=trace_id,
                metadata=dict(result.metadata),
                artifacts=artifacts,
            )

    async def start_session(
        self,
        *,
        tenant_id: str,
        user_id: str,
        session_id: str,
    ) -> None:
        session_key = (tenant_id, user_id, session_id)
        if session_key in self._session_started:
            return
        conscious = await self._memory.start_session(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
        )
        self._session_cache[session_key] = conscious
        self._session_started.add(session_key)

    async def stop(self) -> None:
        if self._started:
            self._started = False
            _LOGGER.info("ai-agent-qbr orchestrator stopped")

    def clear_session(self, *, session_id: str) -> None:
        _LOGGER.info("Clearing session state and memory for %s", session_id)
        if hasattr(self._memory, "clear_session_by_id"):
            self._memory.clear_session_by_id(session_id=session_id)
        self._session_cache = {
            key: value for key, value in self._session_cache.items() if key[2] != session_id
        }
        self._session_started = {key for key in self._session_started if key[2] != session_id}
        self._recent_turns = {
            key: value for key, value in self._recent_turns.items() if key[2] != session_id
        }
