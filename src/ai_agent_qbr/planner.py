"""Planner configuration for ai-agent-qbr."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Union, cast

from penguiflow.catalog import build_catalog
from penguiflow.planner import PlannerEventCallback, ReactPlanner
from penguiflow.planner.memory import MemoryBudget, MemoryIsolation, ShortTermMemoryConfig
from penguiflow.rich_output import DEFAULT_ALLOWLIST, RichOutputConfig, attach_rich_output_nodes, get_runtime
from .config import Config
from .tools import build_catalog_bundle

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_EXTRA = """You are the QBR agent. Answer questions using retrieved QBR context.

- Use the `qbr_context` provided in the LLM context whenever available.
- Cite slide ranges when possible.
- If no context is provided, say that you could not find relevant QBR content.
"""


def _build_system_prompt(rich_output_prompt: str) -> str:
    prompt = SYSTEM_PROMPT_EXTRA
    if rich_output_prompt:
        prompt = f"{prompt}\n\n{rich_output_prompt}" if prompt else rich_output_prompt
    return prompt


@dataclass
class PlannerBundle:
    """Container for planner and LLM client."""

    planner: ReactPlanner
    llm_client: Any  # ScriptedLLM or DSPyLLMClient


class ScriptedLLM:
    """Deterministic LLM client that returns pre-baked planner actions."""

    def __init__(self, scripted: Sequence[Mapping[str, Any]] | None = None) -> None:
        self._scripted = [json.dumps(item, ensure_ascii=False) for item in scripted] if scripted else None

    async def complete(
        self,
        *,
        messages: list[Mapping[str, str]],
        response_format: Mapping[str, Any] | None = None,
        stream: bool = False,
        on_stream_chunk: object = None,
    ) -> str:
        del response_format, stream, on_stream_chunk
        if not self._scripted:
            query = messages[-1].get("content", "")
            scripted = [
                {
                    "thought": "gather evidence",
                    "next_node": "search_documents",
                    "args": {"question": query},
                },
                {
                    "thought": "summarise context",
                    "next_node": "analyze_results",
                    "args": {
                        "results": [
                            {
                                "title": "context",
                                "snippet": f"PenguiFlow answer plan for '{query}'",
                            }
                        ]
                    },
                },
                {
                    "thought": "finish",
                    "next_node": None,
                    "args": {"raw_answer": "Hi! What can I help you with today?"},
                },
            ]
            self._scripted = [json.dumps(item, ensure_ascii=False) for item in scripted]

        return self._scripted.pop(0)


class DatabricksDSPyClient:
    """DSPy-based LLM client for Databricks that implements the full JSONLLMClient protocol.

    This client directly uses a pre-configured dspy.LM instance, which allows us to
    pass Databricks credentials (api_key, api_base) that the standard DSPyLLMClient
    doesn't support.
    """

    expects_json_schema = True

    def __init__(self, lm: Any) -> None:
        self._lm = lm

    def _messages_to_text(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Convert OpenAI-style messages to a single text prompt."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(content)
        return "\n\n".join(parts)

    async def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        response_format: Mapping[str, Any] | None = None,
        stream: bool = False,
        on_stream_chunk: Any = None,
    ) -> str | tuple[str, float]:
        """Generate completion using DSPy with structured output."""
        import asyncio
        import dspy
        from pydantic import BaseModel

        # Import PlannerAction for structured output
        from penguiflow.planner.react import PlannerAction

        # Ignore streaming (DSPy doesn't support it natively)
        del stream, on_stream_chunk

        # Create signature for structured output
        attrs = {
            "__doc__": "Generate a structured PlannerAction output.",
            "__annotations__": {"messages": str, "response": PlannerAction},
            "messages": dspy.InputField(desc="Conversation and query"),
            "response": dspy.OutputField(desc="Structured PlannerAction"),
        }
        signature_class = type("PlannerActionSignature", (dspy.Signature,), attrs)

        # Create predictor
        predictor = dspy.Predict(signature_class)

        # Convert messages to text
        input_text = self._messages_to_text(messages)

        # Run DSPy in executor (it's synchronous)
        loop = asyncio.get_running_loop()

        def _run_dspy() -> Any:
            with dspy.context(lm=self._lm):
                return predictor(messages=input_text)

        result = await loop.run_in_executor(None, _run_dspy)

        # Extract response
        if hasattr(result, "response"):
            response_obj = result.response
            if isinstance(response_obj, BaseModel):
                return response_obj.model_dump_json(), 0.0
            elif isinstance(response_obj, dict):
                return json.dumps(response_obj), 0.0
            else:
                # Try to parse as JSON
                response_str = str(response_obj)
                try:
                    json.loads(response_str)
                    return response_str, 0.0
                except json.JSONDecodeError:
                    # Extract JSON from response if wrapped in other text
                    if "{" in response_str and "}" in response_str:
                        start = response_str.find("{")
                        end = response_str.rfind("}") + 1
                        candidate = response_str[start:end]
                        try:
                            json.loads(candidate)
                            return candidate, 0.0
                        except json.JSONDecodeError:
                            pass
                    raise RuntimeError(f"DSPy returned non-JSON response: {response_str[:200]}")
        else:
            raise RuntimeError("DSPy returned no response field")


def _create_llm_client(config: Config) -> Any:
    """Create LLM client - real DSPy-backed or stub based on config.

    Returns:
        ScriptedLLM if USE_STUB_LLM=true (for testing without Databricks).
        DatabricksDSPyClient wrapping a dspy.LM if USE_STUB_LLM=false.
    """
    if config.use_stub_llm:
        logger.info("Using stub LLM (ScriptedLLM) for planner")
        return ScriptedLLM()

    # Native streaming path for Playground AG-UI
    if config.output_protocol == "agui" and config.planner_stream_final_response:
        api_base = config.databricks_api_base or ""
        if not api_base and config.databricks_host:
            api_base = f"{config.databricks_host.rstrip('/')}/serving-endpoints"
        api_key = config.databricks_api_key or config.databricks_token
        if not api_base or not api_key:
            raise ValueError(
                "Databricks native streaming requires DATABRICKS_API_BASE (or DATABRICKS_HOST) "
                "and DATABRICKS_API_KEY (or DATABRICKS_TOKEN)."
            )
        model_id = f"databricks/{config.llm_model_name}"
        logger.info("Using native Databricks streaming for model=%s", model_id)
        return {"model": model_id, "api_base": api_base, "api_key": api_key}

    # Validate required config
    if not config.databricks_host:
        raise ValueError(
            "DATABRICKS_HOST is required when USE_STUB_LLM=false. "
            "Set it to your Databricks workspace URL (e.g., https://your-workspace.cloud.databricks.com)"
        )
    if not config.databricks_token:
        raise ValueError(
            "DATABRICKS_TOKEN is required when USE_STUB_LLM=false. "
            "Set it to your Databricks personal access token."
        )

    # Import DSPy
    try:
        import dspy
    except ImportError as e:
        raise ImportError(
            "dspy is required for real LLM. Install with: pip install dspy"
        ) from e

    # Create DSPy LM with Databricks endpoint
    serving_host = f"{config.databricks_host}/serving-endpoints"
    model_id = f"databricks/{config.llm_model_name}"

    logger.info(
        f"Creating DSPy LM: model={model_id}, "
        f"api_base={serving_host}, "
        f"max_tokens={config.llm_max_tokens}, "
        f"cache={config.llm_cache_enabled}"
    )

    lm = dspy.LM(
        model_id,
        api_key=config.databricks_token,
        api_base=serving_host,
        max_tokens=config.llm_max_tokens,
        cache=config.llm_cache_enabled,
    )

    # Create our custom Databricks DSPy client that uses the pre-configured LM
    logger.info("Creating DatabricksDSPyClient for ReactPlanner compatibility")
    return DatabricksDSPyClient(lm=lm)


def _build_rich_output_config(config: Config) -> RichOutputConfig:
    allowlist = config.rich_output_allowlist or list(DEFAULT_ALLOWLIST)
    return RichOutputConfig(
        enabled=config.rich_output_enabled,
        allowlist=allowlist,
        include_prompt_catalog=config.rich_output_include_prompt_catalog,
        include_prompt_examples=config.rich_output_include_prompt_examples,
        max_payload_bytes=config.rich_output_max_payload_bytes,
        max_total_bytes=config.rich_output_max_total_bytes,
    )


def _build_short_term_memory(config: Config) -> ShortTermMemoryConfig | None:
    """Build built-in short-term memory configuration."""
    if not config.short_term_memory_enabled:
        return None

    budget = MemoryBudget(
        full_zone_turns=config.short_term_memory_full_zone_turns,
        summary_max_tokens=config.short_term_memory_summary_max_tokens,
        total_max_tokens=config.short_term_memory_total_max_tokens,
        overflow_policy=cast(
            Literal["truncate_summary", "truncate_oldest", "error"],
            config.short_term_memory_overflow_policy,
        ),
    )
    isolation = MemoryIsolation(
        tenant_key=config.short_term_memory_tenant_key,
        user_key=config.short_term_memory_user_key,
        session_key=config.short_term_memory_session_key,
        require_explicit_key=config.short_term_memory_require_explicit_key,
    )

    return ShortTermMemoryConfig(
        strategy=cast(
            Literal["truncation", "rolling_summary", "none"],
            config.short_term_memory_strategy,
        ),
        budget=budget,
        isolation=isolation,
        summarizer_model=config.short_term_memory_summarizer_model,
        include_trajectory_digest=config.short_term_memory_include_trajectory_digest,
        recovery_backlog_limit=config.short_term_memory_recovery_backlog_limit,
        retry_attempts=config.short_term_memory_retry_attempts,
        retry_backoff_base_s=config.short_term_memory_retry_backoff_base_s,
        degraded_retry_interval_s=config.short_term_memory_degraded_retry_interval_s,
    )


def build_planner(
    config: Config,
    *,
    event_callback: PlannerEventCallback | None = None,
) -> PlannerBundle:
    """Create a ReactPlanner with real or stub LLM based on config.

    LLM selection is controlled by USE_STUB_LLM environment variable:
    - USE_STUB_LLM=true (default): Uses ScriptedLLM for testing without Databricks
    - USE_STUB_LLM=false: Uses DSPyLLMClient with Databricks Sonnet 4.5

    When using real LLM, requires:
    - DATABRICKS_HOST: Databricks workspace URL
    - DATABRICKS_TOKEN: Databricks personal access token

    Optional overrides:
    - LLM_MODEL_NAME: Model name (default: databricks-claude-sonnet-4-5)
    - LLM_MAX_TOKENS: Max tokens (default: 4000)
    - LLM_CACHE_ENABLED: Enable response caching (default: true)
    """
    nodes, registry = build_catalog_bundle()
    rich_output_config = _build_rich_output_config(config)
    if rich_output_config.enabled:
        nodes.extend(attach_rich_output_nodes(registry, config=rich_output_config))
    catalog = build_catalog(nodes, registry)
    rich_output_prompt = get_runtime().prompt_section()

    # Create LLM client based on config (stub or real)
    llm_client = _create_llm_client(config)

    if isinstance(llm_client, ScriptedLLM) or isinstance(llm_client, DatabricksDSPyClient):
        planner = ReactPlanner(
            llm_client=llm_client,
            catalog=catalog,
            registry=registry,
            system_prompt_extra=_build_system_prompt(rich_output_prompt),
            event_callback=event_callback,
            stream_final_response=config.planner_stream_final_response,
            short_term_memory=_build_short_term_memory(config),
        )
        return PlannerBundle(planner=planner, llm_client=llm_client)

    planner = ReactPlanner(
        llm=llm_client,
        catalog=catalog,
        registry=registry,
        system_prompt_extra=_build_system_prompt(rich_output_prompt),
        event_callback=event_callback,
        stream_final_response=config.planner_stream_final_response,
        short_term_memory=_build_short_term_memory(config),
    )
    return PlannerBundle(planner=planner, llm_client=llm_client)
