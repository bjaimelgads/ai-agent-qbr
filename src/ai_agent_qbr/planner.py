"""Planner configuration for ai-agent-qbr."""

from __future__ import annotations

import base64
import json
import urllib.parse
import urllib.request
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Union, cast

from penguiflow.catalog import build_catalog
from penguiflow.planner import PlannerEventCallback, ReactPlanner
from penguiflow.planner.memory import MemoryBudget, MemoryIsolation, ShortTermMemoryConfig
from penguiflow.rich_output import DEFAULT_ALLOWLIST, RichOutputConfig, attach_rich_output_nodes, get_runtime
from .config import Config
from .infrastructure.guardrails import build_guardrail_gateway
from .infrastructure.metric_router import MetricQueryRouter
from .tools import build_catalog_bundle

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_EXTRA = """You are the LG Ads QBR agent focused on Quarterly Business Reviews.

- Answer only questions that relate to LG Ads QBRs: advertiser and business insights, performance
  trends, inventory dynamics (CTV, OTT, FAST), pacing risks, and strategic recommendations.
- If the user message is only a greeting (e.g., "hi", "hello"), reply with a short greeting only.
- If the question is out of scope or general knowledge, say you can only help with LG Ads QBR topics
  and ask a brief clarifying question; do not answer the off-topic question.
- Use the `qbr_context` provided in the LLM context whenever available.
- Cite slide ranges when possible.
- If no context is provided, say that you could not find relevant QBR content.
- If the user specifies a region (e.g., US or EMEA), only use data explicitly tied to that region.
  Do not blend regions. If the available context spans multiple regions or is ambiguous, ask the user
  to confirm the desired source (US vs EMEA vs global). If they want aggregated decks, separate the
  response by region.
- Use `region_verification` in context when available to resolve regional scope.
- If the user asks about capabilities, what you can do, or how you can help, call `agent_capabilities`.
- For questions that ask for specific metrics, KPI values, or period comparisons, you MUST call
  `resolve_metric_intent` first. If fields are missing or ambiguous, call `refine_metric_intent`
  with candidate values; if still unresolved, ask a clarifying question. When ready, call
  `query_metrics` using a concise canonical query string that preserves the user’s intent but
  replaces only the missing/ambiguous entities with the resolved values. Avoid verbose sentences.
- For follow-up turns that omit scope (e.g., "what about installs?"), infer missing scope from
  `conversation_memory.recent_turns` and `last_metric_intent` in the LLM context. Preserve the
  user's latest metric change, but carry forward prior client/region/period unless the user
  explicitly overrides them.
- Tool argument contract: for tools with `args.question` (`resolve_metric_intent`, `query_metrics`,
  `search_documents`, `refine_metric_intent`), pass only the latest user utterance or a concise
  canonical rewrite. Never pass planner internals such as `observation`, `context`, serialized
  JSON payloads, prior tool outputs, or citations inside `args.question`.
- Treat `raw_context` and `llm_context_label` as supporting context only. Focus the answer on the
  user’s requested metric(s) and entities; do not introduce additional metrics or KPIs unless the
  user explicitly asked for them. If you include context, tie it directly to the requested metric.
- When citations include `document_url`, include those links in the Sources section.
- When finishing (next_node=null), always include a non-empty `args.raw_answer`.
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


class DeterministicMetricToolLLM:
    """LLM wrapper that deterministically routes metric queries to query_metrics."""

    def __init__(self, inner: Any, router: MetricQueryRouter) -> None:
        self._inner = inner
        self._router = router
        self._last_query: str | None = None
        self._forced_for_last_query = False

    def _extract_user_query(self, messages: Sequence[Mapping[str, str]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return messages[-1].get("content", "") if messages else ""

    async def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        response_format: Mapping[str, Any] | None = None,
        stream: bool = False,
        on_stream_chunk: Any = None,
    ) -> str | tuple[str, float]:
        query = self._extract_user_query(messages)
        if query and query != self._last_query:
            self._last_query = query
            self._forced_for_last_query = False

        if query and not self._forced_for_last_query and self._router.is_metric_query(query):
            self._forced_for_last_query = True
            payload = {
                "thought": "Resolve metric intent before querying structured metrics.",
                "next_node": "resolve_metric_intent",
                "args": {"question": query},
            }
            return json.dumps(payload, ensure_ascii=False)

        return await self._inner.complete(
            messages=messages,
            response_format=response_format,
            stream=stream,
            on_stream_chunk=on_stream_chunk,
        )


class DatabricksDSPyClient:
    """DSPy-based LLM client for Databricks compatible with ReactPlanner."""

    expects_json_schema = True

    def __init__(self, lm: Any) -> None:
        self._lm = lm

    def _messages_to_text(self, messages: Sequence[Mapping[str, str]]) -> str:
        parts: list[str] = []
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
        import json as jsonlib
        import dspy
        from pydantic import BaseModel
        from penguiflow.planner.react import PlannerAction

        del response_format, stream, on_stream_chunk

        input_text = self._messages_to_text(messages)
        loop = asyncio.get_running_loop()

        def _run_dspy(predictor: Any) -> Any:
            with dspy.context(lm=self._lm):
                return predictor(messages=input_text)

        try:
            attrs = {
                "__doc__": "Generate a structured PlannerAction output.",
                "__annotations__": {"messages": str, "response": PlannerAction},
                "messages": dspy.InputField(desc="Conversation and query"),
                "response": dspy.OutputField(desc="Structured PlannerAction"),
            }
            signature_class = type("PlannerActionSignature", (dspy.Signature,), attrs)
            predictor = dspy.Predict(signature_class)
            result = await loop.run_in_executor(None, _run_dspy, predictor)

            if hasattr(result, "response"):
                response_obj = result.response
                if isinstance(response_obj, BaseModel):
                    return response_obj.model_dump_json(), 0.0
                if isinstance(response_obj, dict):
                    return jsonlib.dumps(response_obj), 0.0
                response_str = str(response_obj)
                try:
                    jsonlib.loads(response_str)
                    return response_str, 0.0
                except jsonlib.JSONDecodeError:
                    pass
        except Exception:
            pass

        attrs = {
            "__doc__": "Generate a planner action JSON output.",
            "__annotations__": {
                "messages": str,
                "thought": str,
                "next_node": str,
                "args": dict,
                "plan": list | None,
                "join": dict | None,
            },
            "messages": dspy.InputField(desc="Conversation and query"),
            "thought": dspy.OutputField(desc="Planner reasoning"),
            "next_node": dspy.OutputField(desc="Next node name or null"),
            "args": dspy.OutputField(desc="Planner action arguments"),
            "plan": dspy.OutputField(desc="Parallel plan actions (optional)"),
            "join": dspy.OutputField(desc="Parallel join configuration (optional)"),
        }
        signature_class = type("PlannerActionSignature", (dspy.Signature,), attrs)
        predictor = dspy.Predict(signature_class)
        result = await loop.run_in_executor(None, _run_dspy, predictor)

        thought = getattr(result, "thought", None)
        next_node = getattr(result, "next_node", None)
        args = getattr(result, "args", None)
        plan = getattr(result, "plan", None)
        join = getattr(result, "join", None)
        if thought is None or args is None:
            raise RuntimeError("DSPy returned no planner action fields")
        response_payload = {
            "thought": thought,
            "next_node": next_node,
            "args": args,
            "plan": plan,
            "join": join,
        }
        return jsonlib.dumps(response_payload), 0.0


def _fetch_service_principal_token(config: Config) -> str | None:
    if not config.databricks_client_id or not config.databricks_client_secret:
        return None
    token_url = config.databricks_oauth_token_url
    if not token_url:
        if not config.databricks_host:
            return None
        token_url = f"{config.databricks_host.rstrip('/')}/oidc/v1/token"
    payload = urllib.parse.urlencode(
        {
            "grant_type": "client_credentials",
            "scope": config.databricks_oauth_scope or "all-apis",
        }
    ).encode("utf-8")
    basic = base64.b64encode(
        f"{config.databricks_client_id}:{config.databricks_client_secret}".encode("utf-8")
    ).decode("ascii")
    request = urllib.request.Request(
        token_url,
        data=payload,
        headers={
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch Databricks OAuth token: %s", exc)
        return None
    token = data.get("access_token")
    if not token:
        logger.warning("Databricks OAuth response missing access_token")
        return None
    return str(token)


def _resolve_workspace_client_credentials() -> tuple[str | None, str | None]:
    """Resolve Databricks host/token from implicit workspace client auth."""
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError:
        return None, None

    try:
        workspace_client = WorkspaceClient()
        return workspace_client.config.host, workspace_client.config.token
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to resolve Databricks credentials from workspace client: %s", exc)
        return None, None


def _resolve_databricks_token(config: Config) -> str | None:
    if config.databricks_api_key:
        return config.databricks_api_key
    if config.databricks_token:
        return config.databricks_token

    token = _fetch_service_principal_token(config)
    if token:
        return token

    _, implicit_token = _resolve_workspace_client_credentials()
    return implicit_token


def _resolve_databricks_host(config: Config) -> str | None:
    if config.databricks_host:
        return config.databricks_host
    implicit_host, _ = _resolve_workspace_client_credentials()
    return implicit_host


def _create_llm_client(config: Config) -> Any:
    """Create LLM client - real DSPy-backed or stub based on config.

    Returns:
        ScriptedLLM if USE_STUB_LLM=true (for testing without Databricks).
        DatabricksDSPyClient wrapping a dspy.LM if USE_STUB_LLM=false.
    """
    if config.use_stub_llm:
        logger.info("Using stub LLM (ScriptedLLM) for planner")
        return ScriptedLLM()

    # Native streaming path for WebSocket/AG-UI when streaming final response
    if config.output_protocol in {"agui", "legacy"} and config.planner_stream_final_response:
        api_base = config.databricks_api_base or ""
        resolved_host = _resolve_databricks_host(config)
        if not api_base and resolved_host:
            api_base = f"{resolved_host.rstrip('/')}/serving-endpoints"
        api_key = _resolve_databricks_token(config)
        if not api_base or not api_key:
            raise ValueError(
                "Databricks native streaming requires DATABRICKS_API_BASE (or DATABRICKS_HOST) "
                "and DATABRICKS_API_KEY (or DATABRICKS_TOKEN)."
            )
        model_id = f"databricks/{config.llm_model_name}"
        logger.info("Using native Databricks streaming for model=%s", model_id)
        return {"model": model_id, "api_base": api_base, "api_key": api_key}

    # Validate required config
    resolved_host = _resolve_databricks_host(config)
    if not resolved_host:
        raise ValueError(
            "DATABRICKS_HOST is required when USE_STUB_LLM=false. "
            "Set it to your Databricks workspace URL (e.g., https://your-workspace.cloud.databricks.com)"
        )
    if not _resolve_databricks_token(config):
        raise ValueError(
            "Databricks authentication is required when USE_STUB_LLM=false. "
            "Set DATABRICKS_TOKEN or DATABRICKS_API_KEY, or configure service principal "
            "credentials (DATABRICKS_CLIENT_ID/SECRET)."
        )

    # Import DSPy
    try:
        import dspy
    except ImportError as e:
        raise ImportError(
            "dspy is required for real LLM. Install with: pip install dspy"
        ) from e

    # Create DSPy LM with Databricks endpoint
    serving_host = f"{resolved_host}/serving-endpoints"
    model_id = f"databricks/{config.llm_model_name}"
    api_key = _resolve_databricks_token(config)
    if not api_key:
        raise ValueError("Databricks authentication failed to produce an API token.")

    logger.info(
        f"Creating DSPy LM: model={model_id}, "
        f"api_base={serving_host}, "
        f"max_tokens={config.llm_max_tokens}, "
        f"cache={config.llm_cache_enabled}"
    )

    lm = dspy.LM(
        model_id,
        api_key=api_key,
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
    - REGION_VERIFY_MODEL_NAME: Small model for region verification (default: gpt-5-2-mini)
    - REGION_VERIFY_MAX_TOKENS: Max tokens for region verification (default: 256)
    """
    nodes, registry = build_catalog_bundle()
    rich_output_config = _build_rich_output_config(config)
    if rich_output_config.enabled:
        nodes.extend(attach_rich_output_nodes(registry, config=rich_output_config))
    catalog = build_catalog(nodes, registry)
    logger.info("Planner tools: %s", [node.name for node in nodes])
    rich_output_prompt = get_runtime().prompt_section()

    # Create LLM client based on config (stub or real)
    llm_client = _create_llm_client(config)
    if config.metric_router_enabled and hasattr(llm_client, "complete"):
        llm_client = DeterministicMetricToolLLM(llm_client, MetricQueryRouter())
    guardrail_gateway = build_guardrail_gateway(config)

    if hasattr(llm_client, "complete"):
        planner = ReactPlanner(
            llm_client=llm_client,
            catalog=catalog,
            registry=registry,
            system_prompt_extra=_build_system_prompt(rich_output_prompt),
            event_callback=event_callback,
            stream_final_response=config.planner_stream_final_response,
            short_term_memory=_build_short_term_memory(config),
            guardrail_gateway=guardrail_gateway,
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
        guardrail_gateway=guardrail_gateway,
    )
    return PlannerBundle(planner=planner, llm_client=llm_client)
