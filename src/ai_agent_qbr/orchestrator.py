"""Main orchestrator for ai-agent-qbr."""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import time
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
    text = answer.summary_text.strip()
    details_rows = answer.table_data
    if not details_rows and answer.data:
        details_rows = [row.model_dump() if hasattr(row, "model_dump") else dict(row) for row in answer.data]
    if details_rows:
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
        def _format_source(citation: AnswerCitation) -> str:
            doc_label = citation.document_name or f"doc {citation.document_id}"
            if citation.slide_number is not None:
                doc_label = f"{doc_label} slide {citation.slide_number}"
            elif citation.slide_id is not None:
                doc_label = f"{doc_label} slide {citation.slide_id}"
            doc_url = citation.slide_url or citation.document_url
            if doc_url:
                return f"{doc_label} ({doc_url})"
            return doc_label

        sources = ", ".join(_format_source(c) for c in answer.citations)
        text = f"{text}\n\nSources: {sources}"
    return text


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
            llm_context = {
                "conscious_memories": list(conscious.get("conscious", [])),
                "conversation_memory": {
                    "recent_turns": list(self._recent_turns.get(session_key, []))
                },
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
                "trace_id": trace_id,
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

            with self._mlflow_trace.span(
                name="planner",
                span_type="LLM",
                inputs={
                    "query": query,
                    "qbr_context": filtered_context,
                    "qbr_citations": llm_context["qbr_citations"],
                },
            ) as planner_span:
                planner_started = time.perf_counter()
                _LOGGER.info("Planner start trace_id=%s session_id=%s", trace_id, session_id)
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

            if isinstance(result, PlannerPause):
                raise AiAgentQbrFlowError("Planner paused unexpectedly")

            if not isinstance(result, PlannerFinish):
                raise AiAgentQbrFlowError("Planner did not finish successfully")

            payload: Any = result.payload
            answer_text = _extract_answer(payload)

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
                    },
                )
                self._mlflow_trace.set_outputs(
                    interaction_span,
                    {
                        "answer": answer_text,
                        "trace_id": trace_id,
                    },
                )
                with self._mlflow_tracer.nested_run(
                    run_name="planner",
                    tags={"trace_id": trace_id},
                ) as planner_run:
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

            return AgentResponse(
                answer=answer_text,
                trace_id=trace_id,
                metadata=dict(result.metadata),
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
