"""Main orchestrator for ai-agent-qbr."""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from penguiflow.errors import FlowError
from penguiflow.planner import PlannerFinish, PlannerPause

from ai_agent_qbr.config import Config
from ai_agent_qbr.infrastructure.memory_store import InMemoryMemoryStore
from ai_agent_qbr.planner import PlannerBundle, build_planner
from ai_agent_qbr.telemetry import AgentTelemetry
from qbr_agent.application.use_cases import AnswerQuestion, HybridSearchKnowledge
from qbr_agent.infrastructure.factory import InfrastructureBundle, build_infrastructure

_LOGGER = logging.getLogger(__name__)


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
    if isinstance(payload, Mapping):
        for key in ("raw_answer", "answer", "text", "content", "message", "greeting", "response", "result"):
            if key in payload:
                value = payload.get(key)
                return None if value is None else str(value)
        return str(payload)

    for attr in ("raw_answer", "answer", "text", "content", "message", "greeting", "response", "result"):
        if hasattr(payload, attr):
            value = getattr(payload, attr)
            return None if value is None else str(value)

    return str(payload)


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
                    faiss_dir=self._config.faiss_dir,
                    faiss_normalize=self._config.faiss_normalize,
                )
        return self._infra_bundle

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

        if session_key not in self._session_started:
            _LOGGER.info("Session not started for %s; starting now", session_id)
            await self.start_session(
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=session_id,
            )

        infra = await self._get_infrastructure()
        search_use_case = HybridSearchKnowledge(
            repository=infra.repository,
            vector_index=infra.vector_index,
            embeddings=infra.embeddings,
            text_weight=self._config.retrieval_text_weight,
            vector_weight=self._config.retrieval_vector_weight,
            candidate_multiplier=self._config.retrieval_candidate_multiplier,
        )
        answer_use_case = AnswerQuestion(
            repository=infra.repository,
            vector_index=infra.vector_index,
            embeddings=infra.embeddings,
            use_hybrid=True,
            text_weight=self._config.retrieval_text_weight,
            vector_weight=self._config.retrieval_vector_weight,
            candidate_multiplier=self._config.retrieval_candidate_multiplier,
        )
        answer_context = await answer_use_case.execute(
            query=query,
            top_k=self._config.retrieval_top_k,
            min_score=self._config.retrieval_min_score,
        )

        conscious = self._session_cache.get(session_key, {"conscious": [], "token_estimate": 0})
        llm_context = {
            "conscious_memories": list(conscious.get("conscious", [])),
            "conversation_memory": {
                "recent_turns": list(self._recent_turns.get(session_key, []))
            },
            "qbr_context": answer_context.context,
            "qbr_citations": [
                {
                    "chunk_id": result.chunk.chunk_id.value,
                    "document_id": result.chunk.document_id.value,
                    "score": result.score.value,
                    "start_slide": result.chunk.start_slide,
                    "end_slide": result.chunk.end_slide,
                }
                for result in answer_context.citations
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
            "qbr_answer_context": answer_context.context,
            "retrieval_top_k": self._config.retrieval_top_k,
            "retrieval_min_score": self._config.retrieval_min_score,
        }

        result = await self._planner.run(
            query=query,
            llm_context=llm_context,
            tool_context=tool_context,
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
            )
            self._append_recent_turn(
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=session_id,
                user_prompt=query,
                agent_response=answer_text,
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
