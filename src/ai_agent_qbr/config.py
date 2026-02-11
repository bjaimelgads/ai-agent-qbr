"""Configuration for ai-agent-qbr."""

from __future__ import annotations

import logging
import os
import re
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse
from dataclasses import dataclass, field

_DEFAULT_RICH_OUTPUT_ALLOWLIST = [
    "markdown",
    "json",
    "echarts",
    "mermaid",
    "plotly",
    "datagrid",
    "metric",
    "report",
    "grid",
    "tabs",
    "accordion",
    "code",
    "latex",
    "callout",
    "image",
    "video",
]

_LOGGER = logging.getLogger(__name__)
_ENV_REF_RE = re.compile(r"^\$\{([A-Z0-9_]+)\}$")


def _env_flag(name: str, default: bool) -> bool:
    """Parse boolean from environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Parse integer from environment variable."""
    raw = os.getenv(name)
    return int(raw) if raw is not None else default


def _env_float(name: str, default: float) -> float:
    """Parse float from environment variable."""
    raw = os.getenv(name)
    return float(raw) if raw is not None else default


def _env_str(name: str, default: str) -> str:
    """Parse string from environment variable."""
    raw = os.getenv(name)
    return raw if raw is not None else default


def _normalize_output_protocol(raw: str) -> str:
    normalized = raw.strip().lower()
    if normalized == "websocket":
        return "legacy"
    return normalized


def _env_optional_str(name: str, default: str | None) -> str | None:
    """Parse optional string from environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw or None


def _env_csv(name: str, default: list[str]) -> list[str]:
    """Parse comma-separated list from environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items


def _resolve_env_reference(raw: str) -> str:
    value = (raw or "").strip()
    match = _ENV_REF_RE.match(value)
    if not match:
        return value
    return os.getenv(match.group(1), "").strip()


def _is_unresolved_placeholder(value: str) -> bool:
    raw = (value or "").strip()
    return bool(_ENV_REF_RE.match(raw))


def _build_lakebase_database_url() -> str:
    host = os.getenv("QBR_LAKEBASE_DB_INSTANCE", "")
    if host and "." not in host:
        host = f"{host}.database.cloud.databricks.com"
    db_name = os.getenv("QBR_LAKEBASE_DB_NAME", "")
    port = os.getenv("QBR_LAKEBASE_DB_PORT", "5432")
    username = os.getenv("QBR_LAKEBASE_DB_USERNAME", "")
    has_oauth = bool(os.getenv("DATABRICKS_CLIENT_ID", "").strip()) and bool(
        os.getenv("DATABRICKS_CLIENT_SECRET", "").strip()
    )
    token = _resolve_env_reference(os.getenv("QBR_LAKEBASE_TOKEN", ""))
    if not token:
        token = _resolve_env_reference(os.getenv("DATABRICKS_TOKEN", ""))
    if not token:
        token = _resolve_env_reference(os.getenv("DATABRICKS_API_KEY", ""))
    if token in {"DATABRICKS_TOKEN_REDACTED"} or _is_unresolved_placeholder(token):
        token = ""
    has_pat = bool(token)
    if not host or not db_name or not username or not (has_oauth or has_pat):
        _LOGGER.error(
            "Lakebase settings missing: host=%s db_name=%s username=%s oauth=%s token_set=%s",
            bool(host),
            bool(db_name),
            bool(username),
            has_oauth,
            has_pat,
        )
        raise ValueError(
            "Lakebase settings missing. Require QBR_LAKEBASE_DB_INSTANCE, "
            "QBR_LAKEBASE_DB_NAME, QBR_LAKEBASE_DB_USERNAME, and Databricks auth: "
            "OAuth (DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET) or token "
            "(QBR_LAKEBASE_TOKEN, DATABRICKS_TOKEN, or DATABRICKS_API_KEY)."
        )
    encoded_username = quote(username, safe="")
    # Password is injected dynamically by DatabaseGateway for Lakebase.
    safe_url = f"postgresql+asyncpg://{encoded_username}:***@{host}:{port}/{db_name}"
    _LOGGER.info("Lakebase DATABASE_URL resolved: %s", safe_url)
    return f"postgresql+asyncpg://{encoded_username}:@{host}:{port}/{db_name}"


def _normalize_database_url(database_url: str) -> str:
    if not database_url.startswith("postgresql+asyncpg://"):
        return database_url
    parsed = urlparse(database_url)
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in {"sslmode", "ssl"}
    ]
    cleaned = parsed._replace(query=urlencode(query))
    return urlunparse(cleaned)


def _is_placeholder_database_url(database_url: str) -> bool:
    raw = (database_url or "").strip()
    if not raw:
        return True
    lowered = raw.lower()
    if "${" in raw:
        return True
    if "placeholder-host" in lowered:
        return True
    if "databricks_token_redacted" in lowered:
        return True
    return False


def _parse_optional_float(raw: str | None) -> float | None:
    if raw is None or raw.strip() == "":
        return None
    return float(raw)


def _parse_optional_int(raw: str | None) -> int | None:
    if raw is None or raw.strip() == "":
        return None
    return int(raw)


@dataclass
class Config:
    """Environment-driven configuration."""

    memory_base_url: str = "http://localhost:8000"
    llm_model: str = "stub-llm"
    output_protocol: str = "legacy"
    agui_reasoning_source: str = "status"
    use_native_reasoning: bool = True
    reasoning_effort: str | None = "medium"
    planner_stream_final_response: bool = False
    keepalive_interval_seconds: float = 20.0
    receive_timeout_seconds: float = 60.0
    rich_output_enabled: bool = False
    rich_output_allowlist: list[str] = field(
        default_factory=lambda: list(_DEFAULT_RICH_OUTPUT_ALLOWLIST)
    )
    rich_output_include_prompt_catalog: bool = True
    rich_output_include_prompt_examples: bool = False
    rich_output_max_payload_bytes: int = 250000
    rich_output_max_total_bytes: int = 2000000

    # Built-in Short-Term Memory (ReactPlanner)
    short_term_memory_enabled: bool = True
    short_term_memory_strategy: str = "rolling_summary"
    short_term_memory_full_zone_turns: int = 5
    short_term_memory_summary_max_tokens: int = 1000
    short_term_memory_total_max_tokens: int = 10000
    short_term_memory_overflow_policy: str = "truncate_oldest"
    short_term_memory_tenant_key: str = "tenant_id"
    short_term_memory_user_key: str = "user_id"
    short_term_memory_session_key: str = "session_id"
    short_term_memory_require_explicit_key: bool = True
    short_term_memory_include_trajectory_digest: bool = True
    short_term_memory_summarizer_model: str | None = None
    short_term_memory_recovery_backlog_limit: int = 20
    short_term_memory_retry_attempts: int = 3
    short_term_memory_retry_backoff_base_s: float = 2.0
    short_term_memory_degraded_retry_interval_s: float = 30.0

    # Databricks LLM configuration
    databricks_host: str = ""
    databricks_token: str = ""
    databricks_api_base: str = ""
    databricks_api_key: str = ""
    databricks_client_id: str = ""
    databricks_client_secret: str = ""
    databricks_oauth_token_url: str = ""
    databricks_oauth_scope: str = "all-apis"
    llm_model_name: str = "databricks-claude-sonnet-4-5"
    llm_max_tokens: int = 4000
    llm_cache_enabled: bool = True
    region_verify_model_name: str = "databricks-gpt-5-mini"
    region_verify_max_tokens: int = 256
    llm_intent_enabled: bool = True
    llm_answer_enabled: bool = False
    llm_max_calls_per_query: int = 1
    metric_router_enabled: bool = True
    region_verifier_enabled: bool = False
    reflection_enabled: bool = True
    reflection_quality_threshold: float = 0.8
    reflection_max_revisions: int = 2
    reflection_use_separate_llm: bool = False
    reflection_require_search_documents_on_low_confidence: bool = True

    # Flag to use stub LLM (for testing)
    use_stub_llm: bool = True

    # MLflow tracing
    mlflow_enabled: bool = False
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str | None = None
    mlflow_tracing_enabled: bool = False

    # QBR knowledge retrieval configuration
    database_url: str = "sqlite+aiosqlite:///qbr_intelligence.db"
    log_sqlite_status: bool = True
    storage_backend: str = "sqlite"
    vector_backend: str = "sqlite_embeddings"
    embeddings_backend: str = "sentence_transformers"
    embeddings_model: str = "all-MiniLM-L6-v2"
    embeddings_normalize: bool = True
    retrieval_top_k: int = 5
    retrieval_min_score: float | None = None
    retrieval_enabled: bool = True
    text_search_backend: str = "fts5"
    retrieval_text_weight: float = 0.6
    retrieval_vector_weight: float = 0.4
    retrieval_candidate_multiplier: int = 4
    retrieval_include_document_path: bool = False
    retrieval_max_chunks_per_doc: int = 3
    retrieval_mmr_lambda: float = 0.5
    comparison_top_docs: int = 3
    comparison_per_doc_k: int = 3
    comparison_stage1_top_k: int | None = None
    rerank_backend: str = "cross_encoder"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_max_length: int | None = None
    rerank_top_n: int = 20
    faiss_dir: str = "./data/faiss"
    faiss_normalize: bool = True
    faiss_index_type: str = "Flat"
    faiss_metric: str = "ip"
    faiss_auto_build: bool = False
    faiss_rebuild_on_startup: bool = False

    # Guardrails
    guardrails_enabled: bool = True
    guardrails_mode: str = "enforce"
    guardrails_scope_model_path: str | None = None
    guardrails_jailbreak_endpoint: str | None = None
    guardrails_jailbreak_token: str | None = None
    guardrails_jailbreak_threshold: float = 0.6
    guardrails_jailbreak_payload_style: str = "inputs_list"
    guardrails_jailbreak_input_field: str | None = None
    guardrails_sync_timeout_ms: float = 2000.0
    guardrails_router_endpoint: str | None = (
        "https://dbc-3b4bc42a-bf11.cloud.databricks.com/ml/endpoints/slm-router/metrics?o=185296477739056"
    )
    guardrails_router_token: str | None = None
    guardrails_router_conversation_turns: int = 5


    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        output_protocol = _normalize_output_protocol(
            os.getenv("OUTPUT_PROTOCOL", "legacy")
        )
        storage_backend = os.getenv("STORAGE_BACKEND", "sqlite")
        text_search_backend = os.getenv("TEXT_SEARCH_BACKEND", "fts5")
        database_url = os.getenv("QBR_DB_URL")
        if database_url and _is_placeholder_database_url(database_url):
            _LOGGER.info("Ignoring placeholder DB URL; using Lakebase env vars.")
            database_url = None
        if not database_url and _env_flag("QBR_LAKEBASE_ENABLED", False):
            try:
                database_url = _build_lakebase_database_url()
            except ValueError as exc:
                _LOGGER.warning(
                    "Lakebase configuration incomplete; falling back to SQLite. %s",
                    exc,
                )
                database_url = "sqlite+aiosqlite:///qbr_intelligence.db"
                storage_backend = "sqlite"
                if text_search_backend == "postgres_fts":
                    text_search_backend = "fts5"
        if database_url:
            database_url = _normalize_database_url(database_url)
        return cls(
            memory_base_url=os.getenv("MEMORY_BASE_URL", "http://localhost:8000"),
            llm_model=os.getenv("LLM_MODEL", "stub-llm"),
            output_protocol=output_protocol,
            agui_reasoning_source=_env_str("AGUI_REASONING_SOURCE", "status").lower(),
            use_native_reasoning=_env_flag("USE_NATIVE_REASONING", True),
            reasoning_effort=_env_optional_str("REASONING_EFFORT", "medium"),
            planner_stream_final_response=_env_flag(
                "PLANNER_STREAM_FINAL_RESPONSE",
                output_protocol == "agui",
            ),
            keepalive_interval_seconds=_env_float("WS_KEEPALIVE_SECONDS", 20.0),
            receive_timeout_seconds=_env_float("WS_RECEIVE_TIMEOUT_SECONDS", 60.0),
            rich_output_enabled=_env_flag(
                "RICH_OUTPUT_ENABLED",
                output_protocol == "agui",
            ),
            rich_output_allowlist=_env_csv(
                "RICH_OUTPUT_ALLOWLIST",
                _DEFAULT_RICH_OUTPUT_ALLOWLIST,
            ),
            rich_output_include_prompt_catalog=_env_flag("RICH_OUTPUT_INCLUDE_PROMPT_CATALOG", True),
            rich_output_include_prompt_examples=_env_flag("RICH_OUTPUT_INCLUDE_PROMPT_EXAMPLES", False),
            rich_output_max_payload_bytes=_env_int("RICH_OUTPUT_MAX_PAYLOAD_BYTES", 250000),
            rich_output_max_total_bytes=_env_int("RICH_OUTPUT_MAX_TOTAL_BYTES", 2000000),
            short_term_memory_enabled=_env_flag("SHORT_TERM_MEMORY_ENABLED", True),
            short_term_memory_strategy=os.getenv("SHORT_TERM_MEMORY_STRATEGY", "rolling_summary"),
            short_term_memory_full_zone_turns=_env_int("SHORT_TERM_MEMORY_FULL_ZONE_TURNS", 5),
            short_term_memory_summary_max_tokens=_env_int("SHORT_TERM_MEMORY_SUMMARY_MAX_TOKENS", 1000),
            short_term_memory_total_max_tokens=_env_int("SHORT_TERM_MEMORY_TOTAL_MAX_TOKENS", 10000),
            short_term_memory_overflow_policy=os.getenv(
                "SHORT_TERM_MEMORY_OVERFLOW_POLICY",
                "truncate_oldest",
            ),
            short_term_memory_tenant_key=os.getenv("SHORT_TERM_MEMORY_TENANT_KEY", "tenant_id"),
            short_term_memory_user_key=os.getenv("SHORT_TERM_MEMORY_USER_KEY", "user_id"),
            short_term_memory_session_key=os.getenv("SHORT_TERM_MEMORY_SESSION_KEY", "session_id"),
            short_term_memory_require_explicit_key=_env_flag(
                "SHORT_TERM_MEMORY_REQUIRE_EXPLICIT_KEY",
                True,
            ),
            short_term_memory_include_trajectory_digest=_env_flag(
                "SHORT_TERM_MEMORY_INCLUDE_TRAJECTORY_DIGEST",
                True,
            ),
            short_term_memory_summarizer_model=os.getenv("SHORT_TERM_MEMORY_SUMMARIZER_MODEL"),
            short_term_memory_recovery_backlog_limit=_env_int(
                "SHORT_TERM_MEMORY_RECOVERY_BACKLOG_LIMIT",
                20,
            ),
            short_term_memory_retry_attempts=_env_int("SHORT_TERM_MEMORY_RETRY_ATTEMPTS", 3),
            short_term_memory_retry_backoff_base_s=_env_float(
                "SHORT_TERM_MEMORY_RETRY_BACKOFF_BASE_S",
                2.0,
            ),
            short_term_memory_degraded_retry_interval_s=_env_float(
                "SHORT_TERM_MEMORY_DEGRADED_RETRY_INTERVAL_S",
                30.0,
            ),
            # Databricks LLM configuration
            databricks_host=os.getenv("DATABRICKS_HOST", ""),
            databricks_token=os.getenv("DATABRICKS_TOKEN", ""),
            databricks_api_base=os.getenv("DATABRICKS_API_BASE", ""),
            databricks_api_key=os.getenv("DATABRICKS_API_KEY", ""),
            databricks_client_id=os.getenv("DATABRICKS_CLIENT_ID", ""),
            databricks_client_secret=os.getenv("DATABRICKS_CLIENT_SECRET", ""),
            databricks_oauth_token_url=os.getenv("DATABRICKS_OAUTH_TOKEN_URL", ""),
            databricks_oauth_scope=os.getenv("DATABRICKS_OAUTH_SCOPE", "all-apis"),
            llm_model_name=os.getenv("LLM_MODEL_NAME", "databricks-claude-sonnet-4-5"),
            llm_max_tokens=_env_int("LLM_MAX_TOKENS", 4000),
            llm_cache_enabled=_env_flag("LLM_CACHE_ENABLED", True),
            region_verify_model_name=os.getenv(
                "REGION_VERIFY_MODEL_NAME",
                "databricks-gpt-5-mini",
            ),
            region_verify_max_tokens=_env_int("REGION_VERIFY_MAX_TOKENS", 256),
            llm_intent_enabled=_env_flag("LLM_INTENT_ENABLED", True),
            llm_answer_enabled=_env_flag("LLM_ANSWER_ENABLED", False),
            llm_max_calls_per_query=_env_int("LLM_MAX_CALLS_PER_QUERY", 1),
            metric_router_enabled=_env_flag("METRIC_ROUTER_ENABLED", True),
            region_verifier_enabled=_env_flag("REGION_VERIFIER_ENABLED", False),
            reflection_enabled=_env_flag("REFLECTION_ENABLED", True),
            reflection_quality_threshold=_env_float("REFLECTION_QUALITY_THRESHOLD", 0.8),
            reflection_max_revisions=_env_int("REFLECTION_MAX_REVISIONS", 2),
            reflection_use_separate_llm=_env_flag("REFLECTION_USE_SEPARATE_LLM", False),
            reflection_require_search_documents_on_low_confidence=_env_flag(
                "REFLECTION_REQUIRE_SEARCH_DOCUMENTS_ON_LOW_CONFIDENCE",
                True,
            ),
            use_stub_llm=_env_flag("USE_STUB_LLM", True),
            mlflow_enabled=_env_flag("MLFLOW_ENABLED", False),
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
            mlflow_experiment=os.getenv("MLFLOW_EXPERIMENT")
            or os.getenv("MLFLOW_EXPERIMENT_NAME"),
            mlflow_tracing_enabled=_env_flag("MLFLOW_TRACING_ENABLED", False),
            database_url=database_url or "sqlite+aiosqlite:///qbr_intelligence.db",
            log_sqlite_status=_env_flag("LOG_SQLITE_STATUS", True),
            storage_backend=storage_backend,
            vector_backend=os.getenv("VECTOR_BACKEND", "sqlite_embeddings"),
            embeddings_backend=os.getenv("EMBEDDINGS_BACKEND", "sentence_transformers"),
            embeddings_model=os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2"),
            embeddings_normalize=_env_flag("EMBEDDINGS_NORMALIZE", True),
            retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 5),
            retrieval_min_score=_parse_optional_float(os.getenv("RETRIEVAL_MIN_SCORE")),
            retrieval_enabled=_env_flag("RETRIEVAL_ENABLED", True),
            text_search_backend=text_search_backend,
            retrieval_text_weight=_env_float("RETRIEVAL_TEXT_WEIGHT", 0.6),
            retrieval_vector_weight=_env_float("RETRIEVAL_VECTOR_WEIGHT", 0.4),
            retrieval_candidate_multiplier=_env_int("RETRIEVAL_CANDIDATE_MULTIPLIER", 4),
            retrieval_include_document_path=_env_flag(
                "RETRIEVAL_INCLUDE_DOCUMENT_PATH",
                False,
            ),
            retrieval_max_chunks_per_doc=_env_int("RETRIEVAL_MAX_CHUNKS_PER_DOC", 3),
            retrieval_mmr_lambda=_env_float("RETRIEVAL_MMR_LAMBDA", 0.5),
            comparison_top_docs=_env_int("COMPARISON_TOP_DOCS", 3),
            comparison_per_doc_k=_env_int("COMPARISON_PER_DOC_K", 3),
            comparison_stage1_top_k=_parse_optional_int(os.getenv("COMPARISON_STAGE1_TOP_K")),
            rerank_backend=os.getenv("RERANK_BACKEND", "cross_encoder"),
            rerank_model=os.getenv(
                "RERANK_MODEL",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
            ),
            rerank_max_length=_parse_optional_int(os.getenv("RERANK_MAX_LENGTH")),
            rerank_top_n=_env_int("RERANK_TOP_N", 20),
            faiss_dir=os.getenv("FAISS_DIR", "./data/faiss"),
            faiss_normalize=_env_flag("FAISS_NORMALIZE", True),
            faiss_index_type=os.getenv("FAISS_INDEX_TYPE", "Flat"),
            faiss_metric=os.getenv("FAISS_METRIC", "ip"),
            faiss_auto_build=_env_flag("FAISS_AUTO_BUILD", False),
            faiss_rebuild_on_startup=_env_flag("FAISS_REBUILD_ON_STARTUP", False),
            guardrails_enabled=_env_flag("GUARDRAILS_ENABLED", True),
            guardrails_mode=_env_str("GUARDRAILS_MODE", "enforce"),
            guardrails_scope_model_path=_env_optional_str("GUARDRAILS_SCOPE_MODEL_PATH", None),
            guardrails_jailbreak_endpoint=_env_optional_str("GUARDRAILS_JAILBREAK_ENDPOINT", None),
            guardrails_jailbreak_token=_env_optional_str("GUARDRAILS_JAILBREAK_TOKEN", None),
            guardrails_jailbreak_threshold=_env_float("GUARDRAILS_JAILBREAK_THRESHOLD", 0.6),
            guardrails_jailbreak_payload_style=_env_str(
                "GUARDRAILS_JAILBREAK_PAYLOAD_STYLE",
                "inputs_list",
            ),
            guardrails_jailbreak_input_field=_env_optional_str("GUARDRAILS_JAILBREAK_INPUT_FIELD", None),
            guardrails_sync_timeout_ms=_env_float("GUARDRAILS_SYNC_TIMEOUT_MS", 2000.0),
            guardrails_router_endpoint=_env_optional_str(
                "GUARDRAILS_ROUTER_ENDPOINT",
                "https://dbc-3b4bc42a-bf11.cloud.databricks.com/ml/endpoints/slm-router/metrics?o=185296477739056",
            ),
            guardrails_router_token=_env_optional_str("GUARDRAILS_ROUTER_TOKEN", None),
            guardrails_router_conversation_turns=_env_int(
                "GUARDRAILS_ROUTER_CONVERSATION_TURNS",
                5,
            ),
        )

    def validate(self) -> None:
        """Validate required configuration and raise ValueError when missing."""
        if self.output_protocol not in {"legacy", "agui"}:
            raise ValueError("OUTPUT_PROTOCOL must be one of: legacy, agui")
        if self.agui_reasoning_source not in {"status", "thinking"}:
            raise ValueError("AGUI_REASONING_SOURCE must be one of: status, thinking")
        if self.reasoning_effort is not None and self.reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError("REASONING_EFFORT must be one of: low, medium, high")
        if self.storage_backend not in {"sqlite", "postgres"}:
            raise ValueError("STORAGE_BACKEND must be one of: sqlite, postgres")
        if self.vector_backend not in {"sqlite_embeddings", "faiss", "pgvector"}:
            raise ValueError("VECTOR_BACKEND must be one of: sqlite_embeddings, faiss, pgvector")
        if self.rerank_backend not in {"none", "cross_encoder"}:
            raise ValueError("RERANK_BACKEND must be one of: none, cross_encoder")
        if self.text_search_backend not in {"fts5", "auto", "like", "postgres_fts"}:
            raise ValueError(
                "TEXT_SEARCH_BACKEND must be one of: fts5, auto, like, postgres_fts"
            )
        if self.guardrails_mode not in {"shadow", "enforce"}:
            raise ValueError("GUARDRAILS_MODE must be one of: shadow, enforce")
        if self.guardrails_router_conversation_turns < 1:
            raise ValueError("GUARDRAILS_ROUTER_CONVERSATION_TURNS must be >= 1")
        if not 0.0 <= self.reflection_quality_threshold <= 1.0:
            raise ValueError("REFLECTION_QUALITY_THRESHOLD must be between 0.0 and 1.0")
        if self.reflection_max_revisions < 1:
            raise ValueError("REFLECTION_MAX_REVISIONS must be >= 1")
