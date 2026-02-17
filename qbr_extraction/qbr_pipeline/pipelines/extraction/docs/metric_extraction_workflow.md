# Metric Extraction Workflow

## Purpose
This document explains the current metric extraction workflow across all decks, what each artifact means, which strategies are implemented, and what should be improved next.

## High-Level Flow
The current workflow has two complementary metric outputs:

1. `11_business_metrics.json`
- Main curated output.
- Linked to `metric_catalog` (`metric_catalog_id`, `metric_catalog_slug`).
- Uses the main pipeline (`MetricExtractionPipeline` + context strategies).

2. `11c_business_metrics_unfiltered.json`
- Deterministic extraction output from `02b_raw_content_by_box.txt`.
- Not catalog-filtered by design.
- Includes richer extraction metadata (patterns, scoring components, context selection trace, `table_id`, overall metric flags).

Both are useful:
- `11` is for canonical, queryable KPI results.
- `11c` is for discovery, debugging, quality analysis, and future LLM review routing.

## Artifact Pipeline
Main extraction artifacts per deck are written under:
- `qbr_extraction/qbr_pipeline/output/<Deck Name>/`

Important files:
- `02_raw_content.txt`: page-level raw text.
- `02b_raw_content_by_box.txt`: box-delimited content with `[TABLE]` blocks.
- `04_metrics_extracted.json`: regex/context metric extraction baseline.
- `09_tables_extracted.json`: extracted tables (now includes fallback and `table_id`).
- `11_business_metrics.json`: curated catalog-linked metrics.
- `11c_business_metrics_unfiltered.json`: deterministic unfiltered metrics.

## Step 1: Document Extraction
Implemented in `src/qbr_intelligence/pipeline/processor.py`.

- Kreuzberg extracts pages, text, images, chunks.
- For PPTX table coverage, if `result.tables` is empty, fallback parsing uses `parse_pptx_deck(...)`.
- This fallback populates `09_tables_extracted.json`.

### Table Fallback and IDs
`09_tables_extracted.json` now includes:
- `table_id` (deterministic UUID)
- `slide_number`
- `headers`
- `rows`
- `source` (`kreuzberg` or `pptx_fallback`)

Shared ID generation is in:
- `src/qbr_intelligence/pipeline/table_ids.py`

`table_id` is computed from:
- slide number
- normalized table rows

## Step 2: Deterministic Unfiltered Extraction (`11c`)
Implemented in:
- `src/qbr_intelligence/pipeline/unfiltered_metrics_extractor.py`
- `qbr_extraction/qbr_pipeline/pipelines/extraction/scripts/export_unfiltered_business_metrics.py`

### Parsing Strategy
Input is `02b_raw_content_by_box.txt` with box boundaries:
- `- - - -`
- box content
- `- - - -`

Pair extraction patterns:
- `inline`
- `inline_partial`
- `stacked`
- `value_first`
- `cross_box`
- `table_pipe`
- `table_stacked`
- `table_matrix_row_metric`
- `table_matrix_column_metric`

### Table Orientation Strategy
Two table orientations are supported:

1. Column-metric orientation
- Header columns are metric names.
- Row labels become context labels.
- Example: `Install Rate`, `CPE`, `CPA` as metric names.

2. Row-metric orientation
- First column is metric name.
- Column headers become context labels.
- Example: `CTR` with contexts `Rotational` and `Roadblocks`.

For table-derived metrics:
- `table_id` is attached to each record and metadata.
- `table_context_label` and `table_axis` are stored in metadata.

## Step 3: Name Sanitization
Metric names are normalized before scoring:
- trailing `*`, `†`, `‡`, `[1]`, `(1)` removed
- trailing separators removed

Examples:
- `CPE*` -> `CPE`
- `CPA†` -> `CPA`
- `Install Rate*` -> `Install Rate`

## Step 4: Context Strategy
Raw context is assembled by scoring multiple candidates:
- `box`
- `descriptor_plus_box`
- `richer_descriptor_plus_box`
- `nearby_plus_box`

Context improvements implemented:
- metric name is always included
- thin boxes are enriched with nearby descriptor/title context
- dimension labels can be appended as supporting context
- table metrics prefer title/descriptor + table block

## Step 5: Confidence Strategy
Confidence is composed from:

1. Pair score
- based on extraction pattern strength.

2. Name score
- based on length/tokens/metric hints/occurrence/support.

3. Blend score
- weighted combination (`pair 0.75`, `name 0.25`).
- additional blend-level penalties for long/sentence-like names.

### Penalty Explainability
Blend metadata now explicitly shows penalties:
- `long_name_penalty` or `very_long_name_penalty`
- `long_token_penalty` or `very_long_token_penalty`
- `sentence_like_penalty`
- `penalty_total`

This makes it clear why a metric confidence was lowered.

## Step 6: KPI Hinting from DB Catalog
`11c` metric-hint logic now uses:
- regex hints
- plus DB catalog hints (`metric_catalog.name` + `metric_aliases.alias`)

This avoids penalizing catalog-known KPI names that do not match the hardcoded regex list.

## Step 7: Overall/At-a-Glance Detection
Metrics on summary slides are tagged in `11c` with:
- `metadata.is_overall_metric`
- `metadata.overall_slide_title`
- `metadata.overall_slide_number`

Detection is title-based (`at a glance`, `overall summary`, `executive summary`, etc.).

## Catalog Strategy and Recent Changes
Catalog and dictionary were updated to improve KPI mapping:

- Added explicit `Investment` metric (`slug=investment`).
- Kept `Spend` separate (`slug=spend`).
- Added aliases/patterns for:
  - `Total Installs`
  - `Total Launches`
  - `Duration per Session`
  - `Avg Monthly Freq` variants.

Seeding behavior in `QBRProcessor._seed_metric_catalog` now:
- upserts metrics/aliases for existing DBs
- removes stale seeded aliases that are no longer in definitions.

## Cross-Deck Name Canonicalization
Implemented analysis script:
- `qbr_extraction/qbr_pipeline/pipelines/extraction/scripts/cluster_metric_names.py`

Outputs:
- `artifacts/metric_name_inventory.csv`
- `artifacts/metric_name_clusters.csv`
- `artifacts/metric_name_alias_suggestions.csv`

Purpose:
- group similar metric names across decks
- propose conservative `auto_fix` vs `review` alias suggestions.

Related plan:
- `qbr_extraction/qbr_pipeline/pipelines/extraction/docs/metric_name_canonicalization_plan.md`

## Operational Commands
Refresh extraction artifacts for a single deck:
```bash
.venv/bin/python qbr_extraction/qbr_pipeline/pipelines/extraction/run_extraction_pipeline.py "<deck>.pptx" --no-llm --export-extraction
```

Refresh `11c` for all extracted decks:
```bash
.venv/bin/python qbr_extraction/qbr_pipeline/pipelines/extraction/scripts/export_unfiltered_business_metrics.py \
  --root qbr_extraction/qbr_pipeline/output \
  --database-url "sqlite+aiosqlite:///qbr_intelligence.db" \
  --verbose
```

Run cross-deck name clustering:
```bash
.venv/bin/python qbr_extraction/qbr_pipeline/pipelines/extraction/scripts/cluster_metric_names.py \
  --root qbr_extraction/qbr_pipeline/output \
  --out-dir artifacts \
  --verbose
```

## Known Issue
`run_extraction_pipeline.py` may fail at DB storage with:
- `sqlalchemy.exc.MultipleResultsFound` in `_ensure_document_period(...)`

This is due to duplicate rows in `periods` for the same `period_label`.
Extraction artifacts are still written before this failure.

## Next Steps
1. Fix period uniqueness handling.
- Add safe lookup/upsert behavior in `_ensure_document_period`.
- Add a cleanup migration for duplicate period labels.

2. Merge selected `11c` metadata into `11`.
- Keep catalog linkage from `11`.
- Add useful deterministic metadata from `11c` (`table_id`, `is_overall_metric`, `pattern`, selected confidence components).

3. Add quality scorecards per deck.
- coverage by slide/table
- low-confidence distribution
- long-name/weak-context counts
- missing table-to-metric linkage checks.

4. Add approved alias application workflow.
- consume reviewed alias CSV
- apply canonical names
- preserve `original_name` in metadata
- re-measure quality deltas.

5. Add CI guardrails.
- no numeric-only names
- no unbounded sentence-like names above confidence threshold
- table metrics must include `table_id`.
