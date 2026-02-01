"""Domain models for metric extraction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


UNIT_PERCENT = "percent"
UNIT_CURRENCY = "currency"
UNIT_COUNT = "count"
UNIT_RATIO = "ratio"
UNIT_UNKNOWN = "unknown"

SCALE_ONES = "ones"
SCALE_K = "K"
SCALE_M = "M"
SCALE_B = "B"


@dataclass(frozen=True)
class BBox:
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class TextBlock:
    slide_index: int
    block_id: str
    text: str
    source_type: str
    bbox: BBox | None = None

    def lines(self) -> list[str]:
        return [line for line in (self.text or "").splitlines() if line.strip()]


@dataclass(frozen=True)
class NotesBlock:
    slide_index: int
    block_id: str
    text: str
    source_type: str = "speaker_notes"


@dataclass(frozen=True)
class TableCell:
    slide_index: int
    table_id: str
    row: int
    col: int
    text: str
    cell_id: str


@dataclass(frozen=True)
class Table:
    slide_index: int
    table_id: str
    cells: tuple[TableCell, ...]
    nrows: int
    ncols: int

    def cell_at(self, row: int, col: int) -> TableCell | None:
        for cell in self.cells:
            if cell.row == row and cell.col == col:
                return cell
        return None

    def row_cells(self, row: int) -> list[TableCell]:
        return [cell for cell in self.cells if cell.row == row]

    def col_cells(self, col: int) -> list[TableCell]:
        return [cell for cell in self.cells if cell.col == col]


@dataclass(frozen=True)
class SlideContent:
    slide_index: int
    text_blocks: tuple[TextBlock, ...]
    tables: tuple[Table, ...]
    notes_blocks: tuple[NotesBlock, ...]


@dataclass(frozen=True)
class DeckContent:
    deck_id: str
    slides: tuple[SlideContent, ...]

    def slide_by_index(self, slide_index: int) -> SlideContent | None:
        for slide in self.slides:
            if slide.slide_index == slide_index:
                return slide
        return None


@dataclass(frozen=True)
class MetricAliasSpec:
    alias: str
    pattern: str | None = None
    priority: int | None = None
    unit_override: str | None = None


@dataclass(frozen=True)
class MetricCatalogEntry:
    metric_id: str
    name: str
    aliases: tuple[MetricAliasSpec, ...]
    expected_unit: str
    description: str | None = None
    formula: str | None = None
    disambiguation: tuple[str, ...] = ()
    priority: int = 0


@dataclass(frozen=True)
class LabelCandidate:
    metric_id: str
    metric_name: str
    alias: str
    label_text: str
    slide_index: int
    source_type: str
    block_id: str
    span: tuple[int, int]
    context_text: str
    priority: int = 0


@dataclass(frozen=True)
class ValueCandidate:
    raw_value_text: str
    normalized_value: float | None
    unit: str
    scale: str
    slide_index: int
    source_type: str
    block_id: str
    span: tuple[int, int]
    context_text: str
    qualifiers: dict[str, str] = field(default_factory=dict)
    is_range: bool = False


@dataclass(frozen=True)
class LinkCandidate:
    label: LabelCandidate
    value: ValueCandidate
    relation: str
    features: dict[str, float]
    score: float


@dataclass(frozen=True)
class MetricExtraction:
    deck_id: str
    slide_index: int
    metric_id: str
    metric_name: str
    value: float | None
    unit: str
    scale: str
    raw_value_text: str
    label_text: str
    qualifiers: dict[str, str]
    confidence: float
    extraction_method: str
    provenance: dict[str, str]

    def to_dict(self) -> dict:
        return {
            "deck_id": self.deck_id,
            "slide_index": self.slide_index,
            "metric_id": self.metric_id,
            "metric_name": self.metric_name,
            "value": self.value,
            "unit": self.unit,
            "scale": self.scale,
            "raw_value_text": self.raw_value_text,
            "label_text": self.label_text,
            "qualifiers": self.qualifiers,
            "confidence": self.confidence,
            "extraction_method": self.extraction_method,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class CandidateBundle:
    labels: tuple[LabelCandidate, ...]
    values: tuple[ValueCandidate, ...]

    def labels_for_slide(self, slide_index: int) -> list[LabelCandidate]:
        return [label for label in self.labels if label.slide_index == slide_index]

    def values_for_slide(self, slide_index: int) -> list[ValueCandidate]:
        return [value for value in self.values if value.slide_index == slide_index]


@dataclass(frozen=True)
class ExtractionDebug:
    labels: tuple[LabelCandidate, ...] = ()
    values: tuple[ValueCandidate, ...] = ()
    links: tuple[LinkCandidate, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "labels": [label.__dict__ for label in self.labels],
            "values": [value.__dict__ for value in self.values],
            "links": [
                {
                    "label": link.label.__dict__,
                    "value": link.value.__dict__,
                    "relation": link.relation,
                    "features": link.features,
                    "score": link.score,
                }
                for link in self.links
            ],
            "notes": list(self.notes),
        }


def normalize_unit(unit: str | None) -> str:
    if not unit:
        return UNIT_UNKNOWN
    unit = unit.lower()
    if unit in {UNIT_PERCENT, UNIT_CURRENCY, UNIT_COUNT, UNIT_RATIO, UNIT_UNKNOWN}:
        return unit
    return UNIT_UNKNOWN


def merge_qualifiers(*items: dict[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for item in items:
        if not item:
            continue
        merged.update({k: v for k, v in item.items() if v})
    return merged


def clamp_score(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def metric_key(metric: MetricExtraction) -> tuple:
    return (
        metric.slide_index,
        metric.metric_id,
        metric.value,
        metric.unit,
        tuple(sorted(metric.qualifiers.items())),
    )


def dedupe_metrics(metrics: Iterable[MetricExtraction]) -> list[MetricExtraction]:
    seen: set[tuple] = set()
    output: list[MetricExtraction] = []
    for metric in metrics:
        key = metric_key(metric)
        if key in seen:
            continue
        seen.add(key)
        output.append(metric)
    return output
