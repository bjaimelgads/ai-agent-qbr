"""Compose deterministic answers for metric QA."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable

from qbr_intelligence.schemas.metric_qa import AnswerCitation, MetricAnswer, QueryIntent

from .dao import MetricFactRow


@dataclass
class AggregationResult:
    summary: str
    rows: list[MetricFactRow]
    table: list[dict] | None
    assumptions: list[str]


class AnswerComposer:
    def compose(
        self,
        *,
        intent: QueryIntent,
        rows: list[MetricFactRow],
        assumptions: list[str],
    ) -> MetricAnswer:
        if not rows:
            return MetricAnswer(
                summary="I couldn't find any metrics that match those filters.",
                summary_text="I couldn't find any metrics that match those filters.",
                citations=[],
                confidence=0.0,
                assumptions=assumptions,
                followups=["Try specifying a different metric, client, or time period."],
            )

        aggregation = intent.aggregation or "all"
        result = _apply_aggregation(aggregation, rows)
        if aggregation == "latest" and len(rows) > 1:
            latest = max(rows, key=lambda r: (r.period_end or "", r.period_label or ""))
            latest_key = latest.period_label or latest.period_end
            latest_rows = [
                row
                for row in rows
                if (row.period_label or row.period_end) == latest_key
            ]
            if len(latest_rows) > 1 and any(r.llm_context_label for r in latest_rows):
                summary = (
                    f"Found {len(latest_rows)} values for "
                    f"{latest.metric_name or 'metric'} ({latest_key}) across contexts."
                )
                result = AggregationResult(
                    summary=summary,
                    rows=latest_rows,
                    table=[_row_to_table(r) for r in latest_rows],
                    assumptions=[],
                )
        assumptions.extend(result.assumptions)

        citations = _collect_citations(result.rows)
        confidence = _confidence(result.rows)

        return MetricAnswer(
            summary=result.summary,
            summary_text=result.summary,
            data=[_row_to_metric_row(r) for r in result.rows] if result.rows else None,
            table_data=result.table,
            citations=citations,
            confidence=confidence,
            assumptions=assumptions,
            followups=_followups(intent, rows),
        )


def _apply_aggregation(aggregation: str, rows: list[MetricFactRow]) -> AggregationResult:
    if aggregation == "all":
        ordered = sorted(rows, key=lambda r: (r.period_end or "", r.period_label or ""))
        summary = f"Found {len(ordered)} values for {ordered[0].metric_name or 'metric'} across contexts."
        table = [_row_to_table(r) for r in ordered]
        return AggregationResult(summary=summary, rows=ordered, table=table, assumptions=[])
    if aggregation == "trend":
        ordered = sorted(rows, key=lambda r: (r.period_end or ""))
        table = [_row_to_table(r) for r in ordered]
        summary = _trend_summary(ordered)
        return AggregationResult(summary=summary, rows=ordered, table=table, assumptions=[])
    if aggregation == "compare":
        ordered = sorted(rows, key=lambda r: (r.period_end or ""))
        if len(ordered) < 2:
            summary = "I only found one period to compare."
            return AggregationResult(summary=summary, rows=ordered, table=None, assumptions=["Only one period found."])
        first, second = ordered[-2], ordered[-1]
        delta = None
        pct = None
        if first.value is not None and second.value is not None:
            delta = second.value - first.value
            if first.value != 0:
                pct = delta / first.value * 100
        summary = _compare_summary(first, second, delta, pct)
        table = [_row_to_table(first), _row_to_table(second)]
        return AggregationResult(summary=summary, rows=[first, second], table=table, assumptions=[])
    if aggregation in {"average", "sum", "min", "max"}:
        values = [row.value for row in rows if row.value is not None]
        if not values:
            summary = "No numeric values were available to aggregate."
            return AggregationResult(summary=summary, rows=rows[:1], table=None, assumptions=["Missing numeric values."])
        if aggregation == "average":
            agg_value = mean(values)
            label = "average"
        elif aggregation == "sum":
            agg_value = sum(values)
            label = "total"
        elif aggregation == "min":
            agg_value = min(values)
            label = "minimum"
        else:
            agg_value = max(values)
            label = "maximum"
        summary = f"The {label} value is {_format_value(agg_value, rows[0].unit)} across {len(values)} records."
        return AggregationResult(summary=summary, rows=rows[: min(len(rows), 5)], table=None, assumptions=[])

    # Default to latest
    latest = max(rows, key=lambda r: (r.period_end or ""))
    summary = _latest_summary(latest)
    return AggregationResult(summary=summary, rows=[latest], table=None, assumptions=[])


def _latest_summary(row: MetricFactRow) -> str:
    metric = row.metric_name or "metric"
    period = row.period_label or row.period_end or "the latest period"
    value = _format_value(row.value, row.unit)
    client = f" for {row.client_name}" if row.client_name else ""
    region = f" in {row.region}" if row.region else ""
    return f"Latest {metric}{client}{region} is {value} ({period})."


def _trend_summary(rows: list[MetricFactRow]) -> str:
    metric = rows[0].metric_name or "metric"
    return f"Trend for {metric} across {len(rows)} periods is ready."


def _compare_summary(first: MetricFactRow, second: MetricFactRow, delta: float | None, pct: float | None) -> str:
    metric = second.metric_name or "metric"
    first_label = first.period_label or first.period_end or "earlier period"
    second_label = second.period_label or second.period_end or "later period"
    first_value = _format_value(first.value, first.unit)
    second_value = _format_value(second.value, second.unit)
    summary = f"{metric} moved from {first_value} ({first_label}) to {second_value} ({second_label})."
    if delta is not None:
        delta_value = _format_value(delta, second.unit)
        if pct is not None:
            summary += f" Change: {delta_value} ({pct:.1f}%)."
        else:
            summary += f" Change: {delta_value}."
    return summary


def _row_to_table(row: MetricFactRow) -> dict:
    return {
        "metric": row.metric_name,
        "value": row.value,
        "unit": row.unit,
        "period": row.period_label or row.period_end,
        "client": row.client_name,
        "region": row.region,
        "llm_context_label": row.llm_context_label,
    }


def _row_to_metric_row(row: MetricFactRow):
    from qbr_intelligence.schemas.metric_qa import MetricRow

    return MetricRow(
        metric=row.metric_name,
        value=row.value,
        unit=row.unit,
        period=row.period_label or row.period_end,
        client=row.client_name,
        region=row.region,
        llm_context_label=row.llm_context_label,
    )


def _format_value(value: float | None, unit: str | None) -> str:
    if value is None:
        return "unknown"
    if unit == "percent":
        return f"{value:.2f}%"
    if unit == "currency":
        return f"${value:,.2f}"
    return f"{value:,.2f}"


def _collect_citations(rows: Iterable[MetricFactRow]) -> list[AnswerCitation]:
    seen: set[tuple[int, int | None]] = set()
    citations: list[AnswerCitation] = []
    for row in rows:
        key = (row.document_id, row.slide_id)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            AnswerCitation(
                document_id=row.document_id,
                document_name=row.document_name,
                document_url=row.document_url,
                slide_id=row.slide_id,
                slide_number=row.slide_number,
                snippet=row.snippet,
            )
        )
    return citations


def _confidence(rows: list[MetricFactRow]) -> float | None:
    values = [row.confidence for row in rows if row.confidence is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _followups(intent: QueryIntent, rows: list[MetricFactRow]) -> list[str]:
    suggestions: list[str] = []
    if intent.aggregation != "trend":
        suggestions.append("Want a trend over time?")
    if not intent.region:
        suggestions.append("Specify a region to narrow the results.")
    if not intent.client:
        suggestions.append("Specify a client to narrow the results.")
    return suggestions[:3]
