# Extraction Pipeline Folder

This folder groups the current extraction workflow pieces into one place.

## Included scripts

- `run_extraction_stack.py`
  - Orchestrates:
    1. `qbr_extraction/qbr_pipeline/pipelines/extraction/run_extraction_pipeline.py`
    2. `qbr_extraction/qbr_pipeline/pipelines/extraction/scripts/export_unfiltered_business_metrics.py` (builds `11c`)
    3. `qbr_extraction/qbr_pipeline/pipelines/extraction/scripts/build_metric_normalization_stage4.py` (builds Stage-4 CSV)

- `sync_metric_aliases_from_stage4.py`
  - Reads `artifacts/metric_normalization_stage4.csv`.
  - Finds `catalog_mapped` names that still are not present in DB aliases.
  - Dry-run by default.
  - Use `--apply` to insert aliases into `metric_aliases`.

## Typical usage

Run full extraction stack on all decks:

```bash
uv run python qbr_extraction/qbr_pipeline/pipelines/extraction/run_extraction_stack.py \
  --db sqlite+aiosqlite:///qbr_intelligence.db \
  --override
```

Preview alias additions (no DB writes):

```bash
uv run python qbr_extraction/qbr_pipeline/pipelines/extraction/sync_metric_aliases_from_stage4.py \
  --database-url sqlite+aiosqlite:///qbr_intelligence.db
```

Apply alias additions to DB:

```bash
uv run python qbr_extraction/qbr_pipeline/pipelines/extraction/sync_metric_aliases_from_stage4.py \
  --database-url sqlite+aiosqlite:///qbr_intelligence.db \
  --apply
```

## Notes

- `sync_metric_aliases_from_stage4.py` only inserts aliases for rows where:
  - `normalization_status == catalog_mapped`
  - `canonical_name` exists in `metric_catalog`
  - alias is not already present in `metric_catalog.name` or `metric_aliases.alias`
- It also creates a lightweight sanitized variant for noisy aliases (for example trailing arrows).

## Compatibility

Legacy script entrypoints under `scripts/` remain as thin wrappers so existing commands continue to work.
