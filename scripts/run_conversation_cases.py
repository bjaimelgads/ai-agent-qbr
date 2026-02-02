"""Run multi-turn conversation cases through the full agent flow."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv


def _load_cases(repo_root: Path) -> list[dict]:
    cases_path = repo_root / "tests" / "conversation_cases.json"
    if not cases_path.exists():
        raise SystemExit(f"Missing cases file: {cases_path}")
    return json.loads(cases_path.read_text())


async def _build_orchestrator(config, *, event_callback=None):
    from qbr_agent.infrastructure.factory import build_infrastructure
    from ai_agent_qbr.infrastructure.memory_store import InMemoryMemoryStore
    from ai_agent_qbr.orchestrator import AiAgentQbrOrchestrator
    from ai_agent_qbr.telemetry import AgentTelemetry

    infra = await build_infrastructure(
        database_url=config.database_url,
        storage_backend=config.storage_backend,
        vector_backend=config.vector_backend,
        embeddings_backend=config.embeddings_backend,
        embeddings_model=config.embeddings_model,
        embeddings_normalize=config.embeddings_normalize,
        text_search_backend=config.text_search_backend,
        rerank_backend=config.rerank_backend,
        rerank_model=config.rerank_model,
        rerank_max_length=config.rerank_max_length,
        faiss_dir=config.faiss_dir,
        faiss_normalize=config.faiss_normalize,
    )
    return AiAgentQbrOrchestrator(
        config=config,
        infrastructure=infra,
        memory_store=InMemoryMemoryStore(max_turns=6, retrieval_turns=6),
        telemetry=AgentTelemetry(
            flow_name="ai-agent-qbr",
            logger=__import__("logging").getLogger("ai_agent_qbr.run_conversation_cases"),
            event_callback=event_callback,
        ),
    )


async def _run_cases(
    cases: list[dict],
    config,
    *,
    offset: int = 0,
    limit: int | None = None,
    out_path: Path,
    planner_events_path: Path,
) -> dict:
    from ai_agent_qbr.infrastructure.metric_router import MetricQueryRouter

    planner_events: list[dict] = []
    current_context: dict[str, str | int | None] = {
        "case_id": None,
        "turn": None,
        "session_id": None,
        "query": None,
    }

    def _safe_jsonable(value):
        try:
            json.dumps(value)
            return value
        except TypeError:
            return str(value)

    def _event_to_dict(event) -> dict:
        payload = None
        event_type = None
        if isinstance(event, dict):
            event_type = event.get("event_type") or event.get("type")
            payload = event.get("payload") or event.get("extra") or event
        else:
            if hasattr(event, "event_type"):
                event_type = getattr(event, "event_type")
            if hasattr(event, "extra"):
                payload = getattr(event, "extra")
            elif hasattr(event, "to_payload"):
                try:
                    payload = event.to_payload()
                except TypeError:
                    payload = event.to_payload
            elif hasattr(event, "model_dump"):
                payload = event.model_dump()
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": _safe_jsonable(event_type),
            "payload": _safe_jsonable(payload),
            "context": dict(current_context),
        }

    def _event_callback(event) -> None:
        planner_events.append(_event_to_dict(event))
        planner_events_path.write_text(json.dumps(planner_events, indent=2))

    orchestrator = await _build_orchestrator(config, event_callback=_event_callback)
    router = MetricQueryRouter()
    results: list[dict] = []
    failures = 0

    subset = cases[offset:] if limit is None else cases[offset : offset + limit]
    if out_path.exists():
        existing = json.loads(out_path.read_text())
    else:
        existing = {"total": 0, "results": []}
    merged_results = existing.get("results", [])

    for idx, case in enumerate(subset, start=offset + 1):
        session_id = f"conv-{idx}"
        case_result = {
            "id": case.get("id"),
            "topic": case.get("topic"),
            "turns": [],
        }
        for turn_idx, query in enumerate(case.get("turns", []), start=1):
            current_context.update(
                {
                    "case_id": case.get("id"),
                    "turn": turn_idx,
                    "session_id": session_id,
                    "query": query,
                }
            )
            router_match = bool(router.is_metric_query(query))
            response = await orchestrator.execute(
                query=query,
                tenant_id="test",
                user_id="runner",
                session_id=session_id,
            )
            used_metric_tool = bool(
                response.metadata
                and isinstance(response.metadata, dict)
                and response.metadata.get("metric_answer") is not None
            )
            planner_metadata = (
                response.metadata if isinstance(response.metadata, dict) else None
            )
            case_result["turns"].append(
                {
                    "turn": turn_idx,
                    "query": query,
                    "router_matched": router_match,
                    "metric_tool_used": used_metric_tool,
                    "answer": response.answer,
                    "planner_metadata": planner_metadata,
                }
            )
        results.append(case_result)
        merged_results.append(case_result)
        out_path.write_text(
            json.dumps(
                {
                    "total": len(merged_results),
                    "results": merged_results,
                },
                indent=2,
            )
        )
        planner_events_path.write_text(json.dumps(planner_events, indent=2))

    return {
        "total": len(results),
        "offset": offset,
        "limit": limit,
        "results": results,
        "failed": failures,
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root / "qbr_extraction"))

    load_dotenv(repo_root / ".env")
    os.environ.setdefault("QBR_INTELLIGENCE_LIGHT_IMPORT", "1")
    os.environ.setdefault("LOCAL_DEVELOPMENT", "true")
    os.environ.setdefault("PLATFORM_URL", "")
    os.environ.setdefault("MLFLOW_ENABLED", "false")
    os.environ.setdefault("MLFLOW_TRACING_ENABLED", "false")
    cache_dir = "/tmp/litellm-cache"
    dspy_cache_dir = "/tmp/dspy-cache"
    os.makedirs(dspy_cache_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("DSPY_CACHEDIR", dspy_cache_dir)
    os.environ.setdefault("LITELLM_CACHE", "false")
    os.environ.setdefault("LITELLM_CACHE_TYPE", "memory")
    os.environ.setdefault("DISK_CACHE", "false")
    os.environ.setdefault("DISK_CACHE_DIR", cache_dir)

    from ai_agent_qbr.config import Config

    base_config = Config.from_env()
    config = Config(
        output_protocol="legacy",
        use_stub_llm=False,
        database_url=base_config.database_url,
        guardrails_enabled=False,
        embeddings_backend="hash",
        embeddings_model="ignored",
        embeddings_normalize=True,
        storage_backend="sqlite",
        vector_backend="sqlite_embeddings",
        rerank_backend="none",
        text_search_backend=base_config.text_search_backend,
        llm_intent_enabled=base_config.llm_intent_enabled,
        llm_answer_enabled=base_config.llm_answer_enabled,
        llm_max_calls_per_query=base_config.llm_max_calls_per_query,
        llm_model=base_config.llm_model,
        llm_model_name=base_config.llm_model_name,
        llm_max_tokens=base_config.llm_max_tokens,
        databricks_host=base_config.databricks_host,
        databricks_token=base_config.databricks_token,
        databricks_api_base=base_config.databricks_api_base,
        databricks_api_key=base_config.databricks_api_key,
        databricks_client_id=base_config.databricks_client_id,
        databricks_client_secret=base_config.databricks_client_secret,
        metric_router_enabled=True,
    )

    cases = _load_cases(repo_root)
    offset = int(os.getenv("CASE_OFFSET", "0"))
    limit_raw = os.getenv("CASE_LIMIT")
    limit = int(limit_raw) if limit_raw else None
    runs_root = repo_root / "artifacts" / "conversation_runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_id = os.getenv("CONVERSATION_RUN_ID")
    if not run_id:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_name = os.getenv("CONVERSATION_RESULTS_FILE", "conversation_case_results.json")
    planner_events_name = os.getenv("PLANNER_EVENTS_FILE", "planner_events.json")
    out_path = run_dir / output_name
    planner_events_path = run_dir / planner_events_name

    report = __import__("asyncio").run(
        _run_cases(
            cases,
            config,
            offset=offset,
            limit=limit,
            out_path=out_path,
            planner_events_path=planner_events_path,
        )
    )

    print(f"Cases run: {len(report['results'])} (offset={offset}, limit={limit})")
    if out_path.exists():
        saved = json.loads(out_path.read_text())
        print(f"Total saved: {saved.get('total', 0)}")
    print(f"Results saved to: {out_path}")
    if planner_events_path.exists():
        print(f"Planner events saved to: {planner_events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
