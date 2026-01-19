"""Configuration for ai-agent-qbr."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


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


def _env_csv(name: str, default: list[str]) -> list[str]:
    """Parse comma-separated list from environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items


def _parse_optional_float(raw: str | None) -> float | None:
    if raw is None or raw.strip() == "":
        return None
    return float(raw)


@dataclass
class Config:
    """Environment-driven configuration."""

    memory_base_url: str = "http://localhost:8000"
    llm_model: str = "stub-llm"
    output_protocol: str = "websocket"
    planner_stream_final_response: bool = False
    keepalive_interval_seconds: float = 20.0
    receive_timeout_seconds: float = 60.0
    rich_output_enabled: bool = False
    rich_output_allowlist: list[str] = field(default_factory=lambda: [])
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
    llm_model_name: str = "databricks-claude-sonnet-4-5"
    llm_max_tokens: int = 4000
    llm_cache_enabled: bool = True

    # Flag to use stub LLM (for testing)
    use_stub_llm: bool = True

    # QBR knowledge retrieval configuration
    database_url: str = "sqlite+aiosqlite:///qbr_intelligence.db"
    storage_backend: str = "sqlite"
    vector_backend: str = "sqlite_embeddings"
    embeddings_backend: str = "sentence_transformers"
    embeddings_model: str = "all-MiniLM-L6-v2"
    embeddings_normalize: bool = True
    retrieval_top_k: int = 5
    retrieval_min_score: float | None = None
    retrieval_text_weight: float = 0.6
    retrieval_vector_weight: float = 0.4
    retrieval_candidate_multiplier: int = 4
    faiss_dir: str = "./data/faiss"
    faiss_normalize: bool = True
    faiss_index_type: str = "Flat"
    faiss_metric: str = "ip"
    faiss_auto_build: bool = False
    faiss_rebuild_on_startup: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            memory_base_url=os.getenv("MEMORY_BASE_URL", "http://localhost:8000"),
            llm_model=os.getenv("LLM_MODEL", "stub-llm"),
            output_protocol=os.getenv("OUTPUT_PROTOCOL", "websocket").lower(),
            planner_stream_final_response=_env_flag("PLANNER_STREAM_FINAL_RESPONSE", False),
            keepalive_interval_seconds=_env_float("WS_KEEPALIVE_SECONDS", 20.0),
            receive_timeout_seconds=_env_float("WS_RECEIVE_TIMEOUT_SECONDS", 60.0),
            rich_output_enabled=_env_flag("RICH_OUTPUT_ENABLED", False),
            rich_output_allowlist=_env_csv(
                "RICH_OUTPUT_ALLOWLIST",
                [],
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
            llm_model_name=os.getenv("LLM_MODEL_NAME", "databricks-claude-sonnet-4-5"),
            llm_max_tokens=_env_int("LLM_MAX_TOKENS", 4000),
            llm_cache_enabled=_env_flag("LLM_CACHE_ENABLED", True),
            use_stub_llm=_env_flag("USE_STUB_LLM", True),
            database_url=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///qbr_intelligence.db"),
            storage_backend=os.getenv("STORAGE_BACKEND", "sqlite"),
            vector_backend=os.getenv("VECTOR_BACKEND", "sqlite_embeddings"),
            embeddings_backend=os.getenv("EMBEDDINGS_BACKEND", "sentence_transformers"),
            embeddings_model=os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2"),
            embeddings_normalize=_env_flag("EMBEDDINGS_NORMALIZE", True),
            retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 5),
            retrieval_min_score=_parse_optional_float(os.getenv("RETRIEVAL_MIN_SCORE")),
            retrieval_text_weight=_env_float("RETRIEVAL_TEXT_WEIGHT", 0.6),
            retrieval_vector_weight=_env_float("RETRIEVAL_VECTOR_WEIGHT", 0.4),
            retrieval_candidate_multiplier=_env_int("RETRIEVAL_CANDIDATE_MULTIPLIER", 4),
            faiss_dir=os.getenv("FAISS_DIR", "./data/faiss"),
            faiss_normalize=_env_flag("FAISS_NORMALIZE", True),
            faiss_index_type=os.getenv("FAISS_INDEX_TYPE", "Flat"),
            faiss_metric=os.getenv("FAISS_METRIC", "ip"),
            faiss_auto_build=_env_flag("FAISS_AUTO_BUILD", False),
            faiss_rebuild_on_startup=_env_flag("FAISS_REBUILD_ON_STARTUP", False),
        )

    def validate(self) -> None:
        """Validate required configuration and raise ValueError when missing."""
        if self.output_protocol not in {"websocket", "agui"}:
            raise ValueError("OUTPUT_PROTOCOL must be one of: websocket, agui")
        if self.storage_backend not in {"sqlite"}:
            raise ValueError("STORAGE_BACKEND must be one of: sqlite")
        if self.vector_backend not in {"sqlite_embeddings", "faiss"}:
            raise ValueError("VECTOR_BACKEND must be one of: sqlite_embeddings, faiss")
