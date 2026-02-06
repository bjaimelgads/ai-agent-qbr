"""Run full agent flow for multi-turn conversation cases and save results."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def _load_cases(repo_root: Path) -> list[dict]:
    cases_path = repo_root / "tests" / "metric_conversation_cases.json"
    if not cases_path.exists():
        raise SystemExit(f"Missing cases file: {cases_path}")
    return json.loads(cases_path.read_text())


async def _build_orchestrator(config):
    from qbr_agent.infrastructure.factory import build_infrastructure
    from ai_agent_qbr.infrastructure.memory_store import InMemoryMemoryStore
    from ai_agent_qbr.orchestrator import AiAgentQbrOrchestrator

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
        memory_store=InMemoryMemoryStore(max_turns=5, retrieval_turns=5),
    )


async def _run_conversations(cases: list[dict], config) -> dict:
    orchestrator = await _build_orchestrator(config)
    results: list[dict] = []
    total_turns = 0
    failed_turns = 0

    for idx, case in enumerate(cases, start=1):
        convo_id = case.get("id")
        turns = case.get("turns", [])
        convo_results: list[dict] = []
        session_id = f"conv-{idx}"
        convo_failed = 0

        for turn_idx, turn in enumerate(turns, start=1):
            query = turn.get("query", "")
            expected = bool(turn.get("expected_metric_route", False))
            total_turns += 1

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
            passed = expected == used_metric_tool
            if not passed:
                failed_turns += 1
                convo_failed += 1

            convo_results.append(
                {
                    "turn_index": turn_idx,
                    "query": query,
                    "expected_metric_route": expected,
                    "metric_tool_used": used_metric_tool,
                    "passed": passed,
                    "answer": response.answer,
                }
            )

        results.append(
            {
                "id": convo_id,
                "description": case.get("description"),
                "failed_turns": convo_failed,
                "turns": convo_results,
            }
        )

    return {
        "total_conversations": len(results),
        "total_turns": total_turns,
        "passed_turns": total_turns - failed_turns,
        "failed_turns": failed_turns,
        "results": results,
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
    cache_dir = repo_root / "artifacts" / "litellm_cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    os.environ.setdefault("LITELLM_CACHE", "false")
    os.environ.setdefault("LITELLM_CACHE_TYPE", "none")
    os.environ.setdefault("LITELLM_CACHE_DIR", str(cache_dir))
    os.environ.setdefault("DISK_CACHE_DIR", str(cache_dir))

    from ai_agent_qbr.config import Config

    base_config = Config.from_env()
    config = Config(
        output_protocol="legacy",
        use_stub_llm=False,
        database_url=base_config.database_url,
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
    report = __import__("asyncio").run(_run_conversations(cases, config))

    out_dir = repo_root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "full_agent_conversation_results.json"
    try:
        out_path.write_text(json.dumps(report, indent=2))
    except OSError:
        fallback = Path("/tmp/full_agent_conversation_results.json")
        fallback.write_text(json.dumps(report, indent=2))
        out_path = fallback

    print(
        f"Conversations: {report['total_conversations']}  "
        f"Turns: {report['total_turns']}  "
        f"Passed: {report['passed_turns']}  Failed: {report['failed_turns']}"
    )
    print(f"Results saved to: {out_path}")
    if report["failed_turns"]:
        print("Failed turns:")
        for convo in report["results"]:
            for turn in convo["turns"]:
                if not turn["passed"]:
                    print(
                        f"- {convo['id']} turn {turn['turn_index']}: "
                        f"expected={turn['expected_metric_route']} "
                        f"metric_tool_used={turn['metric_tool_used']}"
                    )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
