# QBR Intelligence

A comprehensive document intelligence system for parsing, enhancing, and querying Quarterly Business Review (QBR) presentations.

## Overview

QBR Intelligence extracts structured data from PowerPoint presentations using [Kreuzberg](https://github.com/flatline-ai/kreuzberg), enhances it with LLM-powered analysis via [DSPY](https://github.com/stanfordnlp/dspy), and provides a queryable interface for AI agents.

### Key Features

- **Document Extraction**: Parse PPTX files to extract text, slides, metrics, charts, and images
- **LLM Enhancement**: Automatic slide classification, metric normalization, chart reconstruction, entity extraction, and executive summary generation
- **Kreuzberg Embeddings**: Built-in ONNX Runtime embeddings stored per chunk for RAG and semantic search
- **Faceted Search**: Query documents by market, campaign, time period, and other dimensions
- **Agent-Ready API**: Tool functions designed for AI agent integration

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Tesseract OCR (optional, for image text extraction)

### Setup

```bash
# Clone and enter the project
cd test_kreuzberg

# Install dependencies
uv sync

# Install Tesseract OCR (macOS)
brew install tesseract tesseract-lang

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### Environment Configuration

Create a `.env` file with the following variables:

```bash
# LLM API Key (choose one)
OPENAI_API_KEY=sk-...                    # For OpenAI
ANTHROPIC_API_KEY=sk-ant-...             # For Anthropic
OPENROUTER_API_KEY=sk-or-...             # For OpenRouter

# LLM Model (format: provider/model-name)
LLM_MODEL=openai/gpt-4o-mini             # Fast and cheap
# LLM_MODEL=openai/gpt-4o                # More capable
# LLM_MODEL=anthropic/claude-3-5-sonnet-20241022
# LLM_MODEL=openrouter/anthropic/claude-3-haiku

# Database URL
DATABASE_URL=sqlite+aiosqlite:///qbr_intelligence.db

# Tesseract OCR data path (macOS with Homebrew)
TESSDATA_PREFIX=/opt/homebrew/share/tessdata

# Kreuzberg Embeddings (ONNX Runtime)
KREUZBERG_EMBEDDINGS_ENABLED=true
KREUZBERG_EMBEDDINGS_PRESET=fast
KREUZBERG_EMBEDDINGS_NORMALIZE=true
KREUZBERG_EMBEDDINGS_BATCH_SIZE=32
KREUZBERG_EMBEDDINGS_SHOW_DOWNLOAD_PROGRESS=false
KREUZBERG_EMBEDDINGS_CACHE_DIR=

# Post-Extraction Embeddings (SentenceTransformers)
QBR_POST_EMBEDDINGS_ENABLED=true
QBR_POST_EMBEDDINGS_MODEL=all-MiniLM-L6-v2
QBR_POST_EMBEDDINGS_DEVICE=cpu
QBR_POST_EMBEDDINGS_BATCH_SIZE=32
QBR_POST_EMBEDDINGS_NORMALIZE=true
```

## Usage

### Processing Documents

```bash
# Extraction only (no LLM, default)
uv run python run_pipeline.py document.pptx

# Full pipeline: extraction + LLM enhancement
uv run python run_pipeline.py document.pptx --llm

# LLM adjudicator only (low-confidence metric adjudication)
uv run python run_pipeline.py document.pptx --llm-adjudicator

# Extraction outputs to extraction_output/<pptx-stem>/
uv run python run_pipeline.py document.pptx --export-extraction

# Process all PPTX files in a folder (top-level only)
uv run python run_pipeline.py --folder decks --export-extraction

# Query existing data only
uv run python run_pipeline.py --query-only

# Query specific document
uv run python run_pipeline.py --query-only --document-id 1

# Export data to JSON files
uv run python run_pipeline.py --query-only --document-id 1 --export ./output

# Custom database
uv run python run_pipeline.py document.pptx --db sqlite:///custom.db

# Overwrite existing document rows with the same filename
uv run python run_pipeline.py document.pptx --override
```

### Metrics Extraction CLI

```bash
# Extract metrics with deterministic scoring (no LLM)
metrics extract qbr_extraction/decks/Disney+\\ US\\ FY24\\ H2.pptx --out /tmp/metrics.json

# Include debug artifact and enable LLM adjudication
metrics extract qbr_extraction/decks/Disney+\\ US\\ FY24\\ H2.pptx --out /tmp/metrics.json --debug /tmp/metrics_debug.json --llm
```

Notes:
- LLM adjudication is gated and cached; set `LEGACY_LLM_METRICS=true` to re-enable the legacy full LLM metric refinement steps.
- For local runs without the `metrics` console script, use `uv run python scripts/metrics.py extract ...`.

### Command Line Options

| Option | Description |
|--------|-------------|
| `file` | Path to PPTX file to process |
| `--no-llm` | Disable all LLM steps (default) |
| `--llm` | Enable all LLM steps (metrics + enhancement + summary + adjudicator) |
| `--llm-metrics` | Enable LLM metric refinement (legacy DSPy pipeline) |
| `--llm-enhancement` | Enable LLM enhancement pipeline (slides/charts/entities) |
| `--llm-summary` | Enable LLM executive summary only |
| `--llm-adjudicator` | Enable LLM adjudication for low-confidence metrics only |
| `--query-only` | Only run queries, skip processing |
| `--document-id ID` | Specify document ID for queries |
| `--export DIR` | Export document data to JSON files |
| `--export-extraction` | Write extraction artifacts to extraction output directory |
| `--extraction-output-dir DIR` | Override extraction output directory |
| `--folder DIR` | Process all PPTX files in a folder (top-level only) |
| `--override` | Delete existing documents with the same filename before processing |
| `--db URL` | Custom database URL |

## Architecture

```
qbr_intelligence/
├── db/                 # Database layer
│   ├── models.py       # SQLAlchemy ORM models
│   └── __init__.py     # Database initialization
├── schemas/            # Pydantic schemas
│   ├── documents.py    # Document/slide/metric schemas
│   ├── llm_outputs.py  # DSPY output schemas
│   └── queries.py      # Query request/response schemas
├── llm/                # LLM enhancement
│   └── modules.py      # DSPY modules for analysis
├── pipeline/           # Processing pipeline
│   └── processor.py    # Main QBRProcessor class
└── query/              # Query interface
    ├── interface.py    # QBRQueryInterface class
    └── tools.py        # Agent tool functions
```

## Query Interface

### Python API

```python
from qbr_intelligence import init_db, QBRQueryInterface
from sqlalchemy.ext.asyncio import AsyncSession

# Initialize database
engine = await init_db("sqlite+aiosqlite:///qbr_intelligence.db")

async with AsyncSession(engine) as session:
    interface = QBRQueryInterface(session)

    # List documents
    docs = await interface.list_documents()

    # Get document summary
    summary = await interface.get_document_summary(document_id=1)

    # Query metrics by category
    metrics = await interface.get_metrics_by_category(document_id=1)

    # Search content
    results = await interface.search_content("revenue growth")
```

### Available Query Methods

#### Document Queries

| Method | Description |
|--------|-------------|
| `list_documents()` | List all documents with optional filtering |
| `get_document(id)` | Get full document details |
| `get_document_summary(id)` | Get executive summary and key insights |

#### Metric Queries

| Method | Description |
|--------|-------------|
| `get_metrics()` | Query metrics with flexible filtering |
| `get_metrics_by_category(id)` | Get metrics grouped by category |
| `get_top_metrics(id, limit)` | Get most significant metrics |
| `compare_metrics_across_documents(name)` | Compare metric across documents |

#### Slide Queries

| Method | Description |
|--------|-------------|
| `get_slides(id)` | Get slides with optional filtering |
| `get_slides_by_type(id)` | Get slides grouped by type |
| `get_slide_detail(slide_id)` | Get full slide content |

#### Entity Queries

| Method | Description |
|--------|-------------|
| `get_entities()` | Query entities (companies, people, etc.) |
| `get_entity_relationships(id)` | Get entity with relationships |

#### Search & Facets

| Method | Description |
|--------|-------------|
| `search_content(query)` | Full-text search in document content |
| `get_available_facets()` | Get all facets for filtering |
| `filter_documents_by_facets(filters)` | Filter documents by facet values |
| `get_keywords(id)` | Get extracted keywords |

#### Aggregations

| Method | Description |
|--------|-------------|
| `get_insights_summary(id)` | Aggregated insights from all slides |
| `get_charts_summary(id)` | All charts with reconstructed data |

## Agent Tool Functions

Pre-built tool functions in `qbr_intelligence.query.tools` for AI agent integration:

```python
from qbr_intelligence.query.tools import (
    # Document tools
    list_documents,           # List all QBR documents
    get_document_summary,     # Get executive summary

    # Metric tools
    get_metrics_by_category,  # Metrics grouped by category
    get_top_metrics,          # Most significant KPIs
    search_metrics,           # Search metrics by name

    # Slide tools
    get_slides_by_type,       # Slides grouped by type
    get_slide_content,        # Full slide details

    # Insight tools
    get_recommendations,      # Strategic recommendations
    get_action_items,         # Concrete next steps
    get_key_insights,         # Aggregated insights

    # Entity tools
    get_entities,             # Companies, people, products
    search_entities,          # Search entities by name

    # Search tools
    search_content,           # Full-text search
    get_charts,               # All charts with data

    # Facet tools
    get_available_facets,     # Available filtering options
    filter_by_facets,         # Filter by facet values

    # Comparison tools
    compare_metric_across_documents,  # Cross-document comparison
)
```

### Tool Usage Example

```python
from sqlalchemy.ext.asyncio import AsyncSession
from qbr_intelligence.query.tools import get_document_summary, get_top_metrics

async with AsyncSession(engine) as session:
    # Get summary
    summary = await get_document_summary(session, document_id=1)
    print(summary["executive_summary"])

    # Get top metrics
    metrics = await get_top_metrics(session, document_id=1, limit=5)
    for m in metrics["top_metrics"]:
        print(f"{m['name']}: {m['raw_value']}")
```

## Data Models

### Document

The root entity representing a QBR presentation:

- `filename`, `file_path`, `title`
- `status`: pending, extracting, extracted, enhancing, enhanced, failed
- `slide_count`, `page_count`, `image_count`
- `executive_summary`, `key_wins`, `areas_for_improvement`
- `recommendations`, `next_steps`
- `client_name`, `report_period`

### Slide

Individual slides with analysis:

- `slide_number`, `raw_text`, `speaker_notes`
- `slide_type`: title, agenda, data, chart, comparison, summary, recommendation, etc.
- `title`, `key_message`, `insights`, `action_items`

### Metric

Extracted and normalized business metrics:

- `raw_value`, `normalized_value`, `unit`
- `category`: performance, cost, reach, engagement, revenue, efficiency
- `trend`: up, down, stable, unknown
- `comparison_type`, `comparison_value`
- `significance_score`, `context`

### Entity

Named entities extracted from content:

- `name`, `entity_type`: company, person, product, market, campaign, date
- `normalized_name`, `description`
- `mention_count`

### Chart

Reconstructed chart data:

- `chart_type`: bar, line, pie, table, comparison, funnel, etc.
- `title`, `data_series`, `data_table`
- `insights`

## LLM Enhancement Pipeline

The DSPY-powered enhancement pipeline runs these modules:

1. **SlideAnalyzer**: Classifies slides, extracts key messages and action items
2. **MetricNormalizer**: Normalizes metrics, assigns categories and significance
3. **ChartReconstructor**: Rebuilds chart data from text elements
4. **EntityExtractor**: Identifies companies, people, products, markets
5. **ExecutiveSummarizer**: Generates executive summary with recommendations
6. **SectionDetector**: Identifies logical document sections

### LLM Call Count

For a typical 80-slide presentation with 70 charts:

| Step | Calls |
|------|-------|
| Slide analysis | ~80 |
| Metric normalization | 1 (batch) |
| Chart reconstruction | ~70 |
| Entity extraction | 1 |
| Executive summary | 1 |
| Section detection | 1 |
| **Total** | **~155 calls** |

## Embeddings (Kreuzberg ONNX Runtime)

This project uses Kreuzberg's built-in ONNX Runtime embeddings to generate vectors for each chunk and persist them in `chunks.embedding`.

### Presets

Kreuzberg presets map to specific models and dimensions:

| Preset | Model | Dimensions | Notes |
|--------|-------|------------|-------|
| `fast` | AllMiniLML6V2Q | 384 | Fastest; best for development and low compute |
| `balanced` | BGEBaseENV15 | 768 | Default production balance |
| `quality` | BGELargeENV15 | 1024 | Highest quality, heavier compute |
| `multilingual` | MultilingualE5Base | 768 | Best for non-English or mixed language |

### Requirements

- ONNX Runtime must be installed for embeddings to work.
- Embeddings are generated locally (CPU) by default.

### How It Works

- `ChunkingConfig` includes an `embedding` configuration.
- Kreuzberg returns `chunk["embedding"]` for each chunk.
- The pipeline stores the vectors in `chunks.embedding` and the model label in `chunks.embedding_model`.

## Embeddings (Post-Extraction SentenceTransformers)

When Kreuzberg does not emit embeddings (e.g., PPTX on some environments), the
pipeline can generate embeddings after extraction using SentenceTransformers.

### Requirements

- `sentence-transformers` must be installed.
- Models are downloaded from HuggingFace on first use.

### How It Works

- Missing `chunk["embedding"]` values are filled after extraction.
- The model label is stored as `sentence-transformers:<model>`.

## Extraction Script (Standalone)

For quick extraction without the full pipeline:

```bash
# Run the standalone extraction script
uv run python extract_qbr.py document.pptx

# Outputs to extraction_output/<pptx-stem>/:
#   - full_content.json
#   - slides.json
#   - metrics.json
#   - charts.json
#   - images/
```

## Development

### Running Tests

```bash
uv run pytest
```

### Linting

```bash
uv run ruff check qbr_intelligence/
uv run ruff check --fix qbr_intelligence/  # Auto-fix
```

### Type Checking

```bash
uv run mypy qbr_intelligence/
```

## Troubleshooting

### OCR Not Working

```bash
# Install Tesseract
brew install tesseract tesseract-lang

# Set TESSDATA_PREFIX in .env
TESSDATA_PREFIX=/opt/homebrew/share/tessdata
```

### Slow Processing

The LLM enhancement processes each slide and chart individually. For faster processing:

```bash
# Skip LLM enhancement
uv run python run_pipeline.py document.pptx --no-llm
```

### Database Schema Changes

If you modify models, delete the database to recreate:

```bash
rm qbr_intelligence.db
uv run python run_pipeline.py document.pptx
```

### Import Errors

Ensure you're using uv to run Python:

```bash
uv run python -c "from qbr_intelligence import *; print('OK')"
```

## License

MIT
