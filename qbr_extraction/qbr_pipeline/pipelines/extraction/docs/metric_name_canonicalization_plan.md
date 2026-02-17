# Metric Name Canonicalization Plan

## Goal
Create a repeatable cross-deck process to normalize metric names, improve context consistency, and feed approved aliases into `metric_catalog`.

## Scope
- Input source: `qbr_extraction/qbr_pipeline/output/**/11c_business_metrics_unfiltered.json`
- Outputs: inventory, clusters, alias suggestions, reviewed alias map
- Applies to all decks and future runs

## Phase 1: Analyze (Implemented)
Run:

```bash
.venv/bin/python qbr_extraction/qbr_pipeline/pipelines/extraction/scripts/cluster_metric_names.py \
  --root qbr_extraction/qbr_pipeline/output \
  --out-dir artifacts \
  --verbose
```

Generated files:
- `artifacts/metric_name_inventory.csv`
- `artifacts/metric_name_clusters.csv`
- `artifacts/metric_name_alias_suggestions.csv`

Current logic:
- Sanitizes names (footnotes/trailing punctuation)
- Clusters by lexical + token similarity (+ unit/context signals)
- Produces conservative `auto_fix` vs `review` suggestions

## Phase 2: Human Review
Review `artifacts/metric_name_alias_suggestions.csv`:
- Accept/reject each suggestion
- Keep `review` rows for manual decisions
- Export approved mapping to `artifacts/approved_metric_aliases.csv`

Recommended approved CSV columns:
- `alias_name`
- `canonical_name`
- `status` (`approved`/`rejected`)
- `notes`

## Phase 3: Apply Canonical Mapping
Build a small application script to:
- Load approved aliases
- Rewrite metric names in `11c_business_metrics_unfiltered.json` (or downstream table)
- Preserve original name in metadata (e.g., `metadata.original_name`)

## Phase 4: Catalog Integration
Upsert approved aliases into DB:
- `metric_catalog`
- `metric_aliases`

Then rerun extraction and compare:
- unique metric names (before/after)
- review queue size (before/after)
- context quality stats

## Phase 5: Quality Gates
Add CI checks for new decks:
- no numeric-only metric names
- no long sentence-like metric names above threshold
- table-derived metrics must include `table_id`
- coverage checks by slide/table where expected
