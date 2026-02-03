# Local Run Guide (Agent + CLI)

This guide is written for non-technical users. Follow it in order.

## 1) Copy the environment file

In the project folder, run:
```
cp .env.example .env
```

## 2) Put these exact values in `.env`

Copy and paste the block below into your `.env`.  
These values are **sensitive** (secrets). Keep the file private.

```
# =============================================================================
# Databricks LLM Configuration (for PenguiFlow planner)
# =============================================================================

# Set to "false" to use real Databricks LLM (Sonnet 4.5)
# Set to "true" (default) to use stub LLM for testing without Databricks
USE_STUB_LLM=false

# Output protocol for WebSocket streaming
# websocket (legacy) or agui (AG-UI events)
OUTPUT_PROTOCOL=websocket
PLANNER_STREAM_FINAL_RESPONSE=true
LOG_LEVEL=DEBUG
PLANNER_DEBUG_EVENTS=true
QBR_INTELLIGENCE_LIGHT_IMPORT=1
WS_RECEIVE_TIMEOUT_SECONDS=180
WS_KEEPALIVE_SECONDS=20
PLATFORM_URL=https://dev-ai-platform-backend-185296477739056.aws.databricksapps.com
GUARDRAILS_ENABLED=false

# Databricks workspace URL (required when USE_STUB_LLM=false)
# Example: https://your-workspace.cloud.databricks.com
DATABRICKS_HOST=https://dbc-3b4bc42a-bf11.cloud.databricks.com

# Databricks personal access token (required when USE_STUB_LLM=false)
# Generate at: Settings > User > Developer > Access tokens

DATABRICKS_API_BASE=https://dbc-3b4bc42a-bf11.cloud.databricks.com/serving-endpoints
DATABRICKS_API_KEY=<YOUR DATABRICKS PAT>
# Service principal auth (used when client_id + client_secret are set)
DATABRICKS_CLIENT_ID=9d99a9b3-324f-4d5e-b712-3cd7a01e953e
DATABRICKS_CLIENT_SECRET=doseb93b92d0445ffbd35e3104d29a82c235

# Model configuration (optional, these are the defaults)
LLM_MODEL_NAME=databricks-claude-sonnet-4-5
LLM_MAX_TOKENS=4000
LLM_CACHE_ENABLED=true

# =============================================================================
# Built-in Short-Term Memory (ReactPlanner)
# =============================================================================
SHORT_TERM_MEMORY_ENABLED=true
SHORT_TERM_MEMORY_STRATEGY=rolling_summary
SHORT_TERM_MEMORY_FULL_ZONE_TURNS=5
SHORT_TERM_MEMORY_SUMMARY_MAX_TOKENS=1000
SHORT_TERM_MEMORY_TOTAL_MAX_TOKENS=10000
SHORT_TERM_MEMORY_OVERFLOW_POLICY=truncate_oldest
# Key paths used to extract identifiers from tool_context (advanced; most apps keep defaults)
# SHORT_TERM_MEMORY_TENANT_KEY=tenant_id
# SHORT_TERM_MEMORY_USER_KEY=user_id
# SHORT_TERM_MEMORY_SESSION_KEY=session_id
SHORT_TERM_MEMORY_REQUIRE_EXPLICIT_KEY=true
SHORT_TERM_MEMORY_INCLUDE_TRAJECTORY_DIGEST=true
# SHORT_TERM_MEMORY_SUMMARIZER_MODEL=
SHORT_TERM_MEMORY_RECOVERY_BACKLOG_LIMIT=20
SHORT_TERM_MEMORY_RETRY_ATTEMPTS=3
SHORT_TERM_MEMORY_RETRY_BACKOFF_BASE_S=2.0
SHORT_TERM_MEMORY_DEGRADED_RETRY_INTERVAL_S=30.0

# =============================================================================
# Rich Output (Playground UI components)
# =============================================================================
# RICH_OUTPUT_ENABLED=true

# =============================================================================
# Alternative LLM API Keys (for extraction/other use cases)
# =============================================================================

# LLM API Key (choose one)
OPENAI_API_KEY=sk-...                    # For OpenAI
ANTHROPIC_API_KEY=sk-ant-...             # For Anthropic
OPENROUTER_API_KEY=sk-or-...             # For OpenRouter

# LLM Model (format: provider/model-name) - used by extraction, not planner
LLM_MODEL=databricks/databricks-claude-sonnet-4-5  # Use Databricks endpoint for extraction
# LLM_MODEL=openai/gpt-4o                # More capable
# LLM_MODEL=anthropic/claude-3-5-sonnet-20241022
# LLM_MODEL=openrouter/anthropic/claude-3-haiku
LLM_SLIDE_LIMIT=20

# =============================================================================
# Google Slides API (optional)
# =============================================================================
CLIENT_ID=<YOUR_GOOGLE_OAUTH_CLIENT_ID>
CLIENT_SECRET=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
# GOOGLE_REFRESH_TOKEN=
# GOOGLE_SLIDES_PRESENTATION_ID=
# GOOGLE_SLIDES_TOKEN_PATH=.google_slides_token.json

# Database URL
DATABASE_URL=sqlite+aiosqlite:////workspaces/ai-agent-qbr/qbr_intelligence.db

LOG_SQLITE_STATUS=false

# =============================================================================
# QBR Retrieval Configuration
# =============================================================================
STORAGE_BACKEND=sqlite
VECTOR_BACKEND=faiss
EMBEDDINGS_BACKEND=sentence_transformers
EMBEDDINGS_MODEL=all-MiniLM-L6-v2
EMBEDDINGS_NORMALIZE=true
RETRIEVAL_TOP_K=5
RETRIEVAL_INCLUDE_DOCUMENT_PATH=false
TEXT_SEARCH_BACKEND=fts5
RETRIEVAL_TEXT_WEIGHT=0.6
RETRIEVAL_VECTOR_WEIGHT=0.4
RETRIEVAL_CANDIDATE_MULTIPLIER=4
RERANK_BACKEND=cross_encoder
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_TOP_N=20
RETRIEVAL_MMR_LAMBDA=0.5
RETRIEVAL_MAX_CHUNKS_PER_DOC=3
FAISS_DIR=./data/faiss
FAISS_NORMALIZE=true
FAISS_INDEX_TYPE=Flat
FAISS_METRIC=ip
FAISS_AUTO_BUILD=true
FAISS_REBUILD_ON_STARTUP=true
# RETRIEVAL_MIN_SCORE=0.0
COMPARISON_TOP_DOCS=3
COMPARISON_PER_DOC_K=3
# COMPARISON_STAGE1_TOP_K=

# Tesseract OCR data path (macOS with Homebrew)
TESSDATA_PREFIX=/opt/homebrew/share/tessdata

# Kreuzberg Embeddings (ONNX Runtime)
KREUZBERG_EMBEDDINGS_ENABLED=false
KREUZBERG_EMBEDDINGS_PRESET=fast
KREUZBERG_EMBEDDINGS_NORMALIZE=true
KREUZBERG_EMBEDDINGS_BATCH_SIZE=32
KREUZBERG_EMBEDDINGS_SHOW_DOWNLOAD_PROGRESS=false
KREUZBERG_EMBEDDINGS_CACHE_DIR=
QBR_POST_EMBEDDINGS_ENABLED=true
QBR_POST_EMBEDDINGS_MODEL=all-MiniLM-L6-v2
QBR_POST_EMBEDDINGS_DEVICE=cpu
QBR_POST_EMBEDDINGS_BATCH_SIZE=32
QBR_POST_EMBEDDINGS_NORMALIZE=true


MLFLOW_ENABLED=false
MLFLOW_TRACING_ENABLED=false
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT=qbr
```

## 3) Start the agent (server)

Open a terminal in the project folder and run:
```
uvicorn ai_agent_qbr.api.app:app --reload
```

Keep this running.

## 4) Start the CLI

After the agent is fully up, open a **second** terminal and run:
```
python scripts/ws_cli.py --protocol legacy --base-url ws://localhost:8000 --repl --timeout 120
```

Type your questions after `user>`.  
Type `exit` to quit.


