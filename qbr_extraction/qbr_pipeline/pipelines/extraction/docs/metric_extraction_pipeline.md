# Metric Extraction Pipeline (v2)

## Goals
- High-precision/recall extraction of QBR metrics from PPTX without OCR.
- Deterministic, tunable scoring with minimal LLM usage.
- Stable JSON schema for downstream use and evaluation.

## Architecture
- **Catalog**: Canonical metrics with aliases, expected units, and disambiguation tokens. Uses the DB catalog when available, with a code fallback for tests/CI.
- **Parser**: `python-pptx` adapter produces canonical slide model (text blocks, tables, notes). Falls back to `qbr_extraction/qbr_pipeline/output/*/03_slides_parsed.json` if PPTX has corrupted media.
- **Candidate Generation**: Label candidates from aliases; value candidates from numeric patterns with unit/scale detection and noise filters.
- **Linker**: Table-aware links (row/column) and text-block proximity links.
- **Scorer**: Deterministic weighted scoring; unit compatibility and layout relation dominate confidence.
- **Resolver**: Disambiguation (Reach vs Unique Reach), dedupe by preference, and source priority.
- **LLM Adjudicator (optional)**: Gated by low confidence or ties; tiny prompts; cached by deck hash + slide index + metric + candidate hash.

## Confidence Gating
- Default threshold: `0.62`.
- Adjudication if:
  - best score below threshold, OR
  - best/second are within `0.08`, OR
  - metric is in ambiguous class (Reach vs Unique Reach).

## Determinism
- When LLM is disabled, pipeline is fully deterministic.
- Outputs are sorted by slide, metric id, raw value, label for stable golden comparisons.

## Extending the Catalog
- Update `src/qbr_intelligence/metrics/catalog.py`.
- Add aliases and disambiguation tokens.
- If a new metric is added to the legacy DB catalog, ensure names match for catalog mapping.

## Updating Golden Tests
1. Run:
   - `uv run python scripts/metrics.py extract qbr_extraction/decks/Disney+\ US\ FY24\ H2.pptx --out /tmp/metrics.json`
2. Regenerate goldens via `uv run python` snippet in `WORKLOG.md` or by running the golden update helper if added.
3. Commit updated files under `qbr_extraction/tests/golden/`.

## LLM Adjudication Notes
- The adjudicator expects strict JSON: `{ chosen_index, normalized_value, unit, reasoning_short }`.
- Cache directory defaults to `qbr_extraction/qbr_pipeline/output/.metric_adjudicator_cache`.
- Use `LEGACY_LLM_METRICS=true` only if you want to re-enable the older full LLM refinement steps.
