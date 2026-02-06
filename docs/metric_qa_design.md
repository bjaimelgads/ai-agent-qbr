# Metric QA (Structured Retrieval) Design

## Goals
- Answer metric questions using structured `metric_fact` data, not embeddings over slide text.
- Deterministic entity resolution, filters, and aggregation.
- Optional LLM use limited to intent parsing and phrasing.
- Provide citations (document_id, slide_id) in all answers.

## Data Model
**View:** `metric_fact`
- fact_id
- metric_id, metric_name
- value, unit, scale
- client_id, client_name
- region
- period_start, period_end, period_granularity, period_label
- document_id, slide_id
- confidence
- provenance: label_text, raw_value_text, snippet

**Source tables:** `metrics`, `metric_catalog`, `metric_aliases`, `documents`, `clients`, `regions`, `periods`.

The view is created on-demand by the MetricFactStore (safe for sqlite/postgres).

## Intent Schema
See `qbr_extraction/qbr_intelligence/schemas/metric_qa.py`:
- metric_ids
- client
- region
- period (type + range)
- aggregation
- grouping
- limit
- clarifications_needed

## Resolvers
- `MetricResolver`: alias + pattern matching with disambiguation tokens.
- `ClientResolver`: longest exact match against client catalog.
- `RegionResolver`: known region tokens + heuristic inference.
- `PeriodResolver`: Q/H/FY parsing + relative ranges using latest period_end.

## Query Planner
- Filters: metric_id, client, region, period overlap.
- Order: `period_end DESC` (latest) or `ASC` (trend).
- Default limit: 200.

## Answer Composer
- Deterministic math: latest/trend/compare/avg/sum/min/max.
- Produces `MetricAnswer` with citations.
- Optional LLM phrasing (disabled by default).

## Tool Integration
- Tool: `query_metrics(question: str, debug: bool=false) -> MetricAnswer`.
- Heuristic router detects metric questions and routes directly.
- Tool remains available to React planner for LLM-driven flows.

## Caching & Guardrails
- LRU cache for intent and phrasing results.
- LLM usage gated by `LLM_INTENT_ENABLED`, `LLM_ANSWER_ENABLED`, `LLM_MAX_CALLS_PER_QUERY`.

## Debug Mode
`MetricAnswer.debug` includes:
- parsed intent + resolution confidence
- SQL + parameters
- fallback reasons
- top rows

## Extensibility
- Add metrics: update `metric_catalog` / `metric_aliases`.
- Add regions: update `regions` table.
- Adjust fiscal year: set `FISCAL_YEAR_START_MONTH`.
