# PPTX Extraction Baseline (Phase 0)

This memo summarizes the current ingestion/extraction flow, output schema, and
constraints for extending PPTX extraction without breaking existing behavior.

## Current Pipeline (As Implemented)
- Entrypoints
  - `qbr_extraction/qbr_pipeline/pipelines/extraction/run_extraction_pipeline.py` runs the end-to-end extraction and optional LLM enhancement.
  - `qbr_extraction/qbr_pipeline/pipelines/extraction/scripts/extract_kreuzberg_standalone.py` is a standalone extraction script that writes files to `qbr_extraction/qbr_pipeline/output/`.
- Extraction
  - `src/qbr_intelligence/pipeline/processor.py` calls Kreuzberg
    `extract_file_sync()` with `ExtractionConfig` (OCR, images, page markers,
    keyword extraction, language detection, chunking + embeddings).
  - Slide parsing is currently done by splitting Kreuzberg content on
    `<!-- PAGE n -->` markers and extracting notes via `### Notes:`.
  - Metrics extraction is regex-based over raw content (percent, currency, rate).
  - Chart detection is heuristic (multiple percentages or colon-delimited lines).
  - Optional post-extraction embeddings fill missing vectors via
    SentenceTransformers (`post_embeddings.py`).
- Storage
  - SQLite DB `qbr_intelligence.db` via SQLAlchemy models in
    `src/qbr_intelligence/db/models.py`.
  - Core tables: `documents`, `slides`, `chunks`, `metrics`, `charts`, `images`,
    `entities`, `keywords`, `facets`.
- Retrieval
  - `src/qbr_intelligence/query/interface.py` exposes agent-friendly queries.
  - `search_content()` uses text search (`ILIKE`) over `chunks.content`.
  - Embeddings are stored in `chunks.embedding` but not used in the query layer.

## Existing Extraction Outputs (Files + Fields)
Emitted by `QBRProcessor._export_extraction_outputs()` and `extract_kreuzberg_standalone.py`:
- `01_extraction_metadata.json`
  - `source_file`, `extraction_timestamp`, `kreuzberg_version`, `mime_type`,
    `page_count`, `detected_languages`, `table_count`, `image_count`,
    `chunk_count`, `metadata` (Kreuzberg metadata blob).
- `02_raw_content.txt`
  - Kreuzberg text with `<!-- PAGE n -->` markers and inline HTML tables.
- `03_slides_parsed.json`
  - `slide_number`, `raw_text`, `speaker_notes`, `has_images`, `image_count`.
  - Optional `source_slide_number` when remapping page order is applied.
- `04_metrics_extracted.json`
  - `value`, `metric_type` (percentage/currency/rate), `context`, `slide_number`.
- `05_charts_detected.json`
  - `slide_number`, `chart_type`, `raw_elements`, `confidence`.
- `06_keywords_topics.json`
  - `business_terms_detected`, plus `raw_keywords` when available.
- `07_images_metadata.json`
  - `index`, `content_type`, `size_bytes`, `source_location`, `saved_path`.
- `08_chunks_rag.json`
  - `chunk_index`, `content`, `char_count`, `metadata` (`byte_start`, `byte_end`,
    `chunk_index`, `total_chunks`), `has_embedding`, `embedding_model`,
    `embedding_dimensions`.
- `09_tables_extracted.json`
  - `index`, `headers`, `rows`, `raw`.
- `00_slide_order_map.json` (only when remapping happens)
  - `slide_number`, `source_page_number`, `token_overlap`.

## Observed Quality Gaps (From `qbr_extraction/qbr_pipeline/output/agentiv_disney`)
- Slide structure is largely flattened; no typed elements, bounding boxes,
  or reading order beyond page markers.
- Titles are not explicitly detected; they appear inline as raw text.
- Tables appear inline as HTML fragments, but `table_count` is `0`, so
  structured table extraction is inconsistent.
- Image-heavy slides mostly show image references and loose bullet text.
- Metrics extraction is regex-only and can misfire:
  - Currency regex picks partial values (e.g., `$2` from `$2,556,629.87`).
  - Values are not linked to labels/KPIs.
- Chart detection is heuristic and can flag non-chart content as tables.
- Slide numbering can be remapped via token overlap; this is probabilistic and
  may misalign slide references if overlap is weak.

## Schema Assumptions / Potential Breakpoints
Areas that implicitly assume current shapes:
- `src/qbr_intelligence/pipeline/processor.py`:
  - `parse_slides()` expects `<!-- PAGE n -->` markers and `### Notes:` blocks.
  - `extract_metrics()` and `detect_charts()` read plain text only.
  - Output JSON files are written with fixed keys and file names.
- `src/qbr_intelligence/db/models.py` and
  `src/qbr_intelligence/schemas/documents.py`:
  - Pydantic schemas define strict fields; new fields must be additive or
    introduced via namespaced JSON in `extraction_metadata` or `metadata`.
- `src/qbr_intelligence/query/interface.py`:
  - Retrieval assumes chunks exist and are text-only; no slide/element structure.

## Candidate Extension Points
Safe, incremental hooks for new functionality:
- Replace/extend `parse_slides()` with a `python-pptx` parser for slide elements,
  titles, notes, tables, and bounding boxes.
- Add a structured `SlideModel`/`SlideElement` model alongside current
  `slides` table while preserving existing `raw_text`.
- Introduce new extraction stages (entity extraction, structured tables)
  as optional pipeline strategies and serialize them into additive output
  files or namespaced metadata.
- Use `extraction_metadata` and `metadata` for new namespaced fields to avoid
  breaking existing readers.

## Risks / Constraints
- Must preserve legacy output file formats and DB schema expectations.
- Avoid breaking `QBRProcessor.process_document()` and `extract_kreuzberg_standalone.py` output.
- Ensure that any new fields are additive and namespaced.
- Maintain deterministic slide-to-chunk mapping; remap logic should remain
  stable unless intentionally changed and documented.
