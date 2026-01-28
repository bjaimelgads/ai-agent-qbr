#!/usr/bin/env python3
"""End-to-end region filter validation for ai-agent-qbr."""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
DOTENV_PATH = ROOT_DIR / ".env"

US_PATTERNS_CASE_SENSITIVE = (
    r"\bUS\b",
    r"\bUSA\b",
    r"\bU\.S\.\b",
    r"\bU\.S\.A\.\b",
)

US_PATTERNS_CASE_INSENSITIVE = (
    r"\bUnited States\b",
    r"\bDomestic\b",
    r"\bNorth America\b",
)

EMEA_PATTERNS_CASE_SENSITIVE = (
    r"\bEMEA\b",
    r"\bEU5\b",
    r"\bEU-?5\b",
    r"\bEU\b",
    r"\bUK\b",
    r"\bU\.K\.\b",
    r"\bDE\b",
    r"\bFR\b",
    r"\bIT\b",
    r"\bES\b",
    r"\bTR\b",
    r"\bPL\b",
    r"\bSE\b",
    r"\bGR\b",
)

EMEA_PATTERNS_CASE_INSENSITIVE = (
    r"\bEurope(?:an)?\b",
    r"\bGermany\b",
    r"\bFrance\b",
    r"\bItaly\b",
    r"\bSpain\b",
    r"\bTurkey\b",
    r"\bPoland\b",
    r"\bSweden\b",
    r"\bGreece\b",
)


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _matches_any(text: str, patterns: Iterable[str], flags: int = 0) -> bool:
    return any(re.search(pattern, text, flags) for pattern in patterns)


def _validate_output(text: str, region: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if region == "us":
        if _matches_any(text, EMEA_PATTERNS_CASE_SENSITIVE):
            errors.append("Found EMEA short codes in US response.")
        if _matches_any(text, EMEA_PATTERNS_CASE_INSENSITIVE, re.IGNORECASE):
            errors.append("Found EMEA region terms in US response.")
    elif region == "emea":
        if _matches_any(text, US_PATTERNS_CASE_SENSITIVE):
            errors.append("Found US short codes in EMEA response.")
        if _matches_any(text, US_PATTERNS_CASE_INSENSITIVE, re.IGNORECASE):
            errors.append("Found US region terms in EMEA response.")
    ok = not errors
    return ok, errors


def _update_summary(script_path: Path, summary_lines: list[str]) -> None:
    content = script_path.read_text().splitlines()
    start = "# === Last Run Summary ==="
    end = "# === End Summary ==="
    if start not in content or end not in content:
        return
    start_idx = content.index(start)
    end_idx = content.index(end)
    updated = (
        content[: start_idx + 1]
        + [f"# {line}" for line in summary_lines]
        + content[end_idx:]
    )
    script_path.write_text("\n".join(updated) + "\n")


async def _run() -> int:
    _load_dotenv(DOTENV_PATH)
    os.environ.setdefault("REGION_VERIFY_MODEL_NAME", "databricks-gpt-5-mini")
    os.environ.setdefault("REGION_VERIFY_MAX_TOKENS", "256")
    os.environ.setdefault("OUTPUT_PROTOCOL", "websocket")
    os.environ.setdefault("PLANNER_STREAM_FINAL_RESPONSE", "false")
    os.environ.setdefault("FAISS_AUTO_BUILD", "false")
    os.environ.setdefault("FAISS_REBUILD_ON_STARTUP", "false")
    os.environ.setdefault("MLFLOW_ENABLED", "false")
    os.environ.setdefault("MLFLOW_TRACING_ENABLED", "false")
    os.environ.setdefault("LLM_MODEL_NAME", "databricks-claude-haiku-4-5")
    os.environ.setdefault("LLM_MAX_TOKENS", "800")
    os.environ.setdefault("GUARDRAILS_ENABLED", "false")

    sys.path.insert(0, str(ROOT_DIR))

    from ai_agent_qbr.config import Config
    from ai_agent_qbr.infrastructure.region_filter import (
        build_context_from_items,
        filter_items_by_region,
    )
    from ai_agent_qbr.infrastructure.region_verifier import RegionVerifier
    from qbr_agent.application.use_cases import AnswerQuestion
    from qbr_agent.infrastructure.factory import build_infrastructure

    config = Config.from_env()
    config.validate()

    region_verifier = RegionVerifier(config)

    cases = [
        ("Summarize the latest QBR for Disney US.", "us"),
        ("Summarize the latest QBR for Disney EMEA.", "emea"),
    ]

    results: list[str] = []
    failures = 0

    for query, region in cases:
        print(f"Running region verification for: {query}", flush=True)
        verification = await region_verifier.verify(query)
        print(f"Region verification result: {verification}", flush=True)

        print("Running retrieval...", flush=True)
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
        answer_use_case = AnswerQuestion(
            repository=infra.repository,
            vector_index=infra.vector_index,
            embeddings=infra.embeddings,
            reranker=infra.reranker,
            use_hybrid=True,
            text_weight=config.retrieval_text_weight,
            vector_weight=config.retrieval_vector_weight,
            candidate_multiplier=config.retrieval_candidate_multiplier,
            include_document_path=config.retrieval_include_document_path,
            rerank_top_n=config.rerank_top_n,
            mmr_lambda=config.retrieval_mmr_lambda,
            max_chunks_per_doc=config.retrieval_max_chunks_per_doc,
        )
        answer_context = await answer_use_case.execute(
            query=query,
            top_k=config.retrieval_top_k,
            min_score=config.retrieval_min_score,
        )
        filtered_citations = filter_items_by_region(
            answer_context.citations,
            verification.region_focus,
        )
        filtered_context = build_context_from_items(filtered_citations)
        print(
            f"Filtered context chars: {len(filtered_context)} (citations={len(filtered_citations)})",
            flush=True,
        )

        print("Running LLM summary...", flush=True)
        import dspy
        from ai_agent_qbr.planner import _resolve_databricks_host, _resolve_databricks_token

        resolved_host = _resolve_databricks_host(config)
        api_base = f"{resolved_host.rstrip('/')}/serving-endpoints"
        model_id = f"databricks/{config.llm_model_name}"
        api_key = _resolve_databricks_token(config)
        lm = dspy.LM(
            model_id,
            api_key=api_key,
            api_base=api_base,
            max_tokens=config.llm_max_tokens,
            cache=False,
        )

        class SummarySig(dspy.Signature):
            context: str = dspy.InputField()
            question: str = dspy.InputField()
            answer: str = dspy.OutputField(
                desc=(
                    "Answer grounded in the provided context only. "
                    "Do not mix regions or introduce other markets."
                )
            )

        predictor = dspy.Predict(SummarySig)

        def _run_summary():
            with dspy.context(lm=lm):
                return predictor(context=filtered_context, question=query)

        try:
            summary = await asyncio.wait_for(asyncio.to_thread(_run_summary), timeout=120)
        except asyncio.TimeoutError:
            failures += 1
            results.append(f"{region.upper()} response: FAIL - summary timeout")
            continue

        answer_text = str(getattr(summary, "answer", "") or "")
        print(f"\n--- Response ({region}) ---\n{answer_text}\n")

        ok, errors = _validate_output(answer_text, region)
        if ok:
            results.append(f"{region.upper()} response: OK")
        else:
            failures += 1
            results.append(f"{region.upper()} response: FAIL - {'; '.join(errors)}")

    summary_lines = [
        "Region filter E2E run completed.",
        f"Total cases: {len(cases)}",
        f"Failures: {failures}",
    ] + results
    _update_summary(Path(__file__), summary_lines)

    if failures:
        print("\nE2E validation failed.")
        return 1
    print("\nE2E validation passed.")
    return 0


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())

# === Last Run Summary ===
# Region filter E2E run completed.
# Total cases: 2
# Failures: 1
# US response: OK
# EMEA response: FAIL - Found US short codes in EMEA response.
# === End Summary ===
