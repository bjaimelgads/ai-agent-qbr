"""Link label and value candidates using layout-aware heuristics."""

from __future__ import annotations

from collections import defaultdict

from qbr_intelligence.metrics.models import LabelCandidate, LinkCandidate, Table, ValueCandidate


def _line_index(text: str, position: int) -> int:
    return text.count("\n", 0, max(0, position))


def _extract_between(text: str, span_a: tuple[int, int], span_b: tuple[int, int]) -> str:
    start = min(span_a[1], span_b[1])
    end = max(span_a[0], span_b[0])
    return text[start:end]


def link_text_candidates(labels: list[LabelCandidate], values: list[ValueCandidate]) -> list[LinkCandidate]:
    links: list[LinkCandidate] = []
    values_by_block = defaultdict(list)
    for value in values:
        values_by_block[(value.slide_index, value.block_id)].append(value)

    for label in labels:
        block_values = values_by_block.get((label.slide_index, label.block_id), [])
        if not block_values:
            continue
        for value in block_values:
            distance = _distance_between(label.span, value.span)
            same_line = _line_index(label.context_text, label.span[0]) == _line_index(
                label.context_text, value.span[0]
            )
            between = _extract_between(label.context_text, label.span, value.span)
            has_separator = ":" in between or "=" in between or "-" in between
            features = {
                "distance": float(distance),
                "same_line": 1.0 if same_line else 0.0,
                "has_separator": 1.0 if has_separator else 0.0,
            }
            links.append(
                LinkCandidate(
                    label=label,
                    value=value,
                    relation="same_block",
                    features=features,
                    score=0.0,
                )
            )
    return links


def link_table_candidates(tables: list[Table], labels: list[LabelCandidate], values: list[ValueCandidate]) -> list[LinkCandidate]:
    links: list[LinkCandidate] = []
    value_by_cell = {value.block_id: value for value in values}
    label_by_cell = {label.block_id: label for label in labels}

    for table in tables:
        cells_by_pos = {(cell.row, cell.col): cell for cell in table.cells}
        for cell in table.cells:
            label = label_by_cell.get(cell.cell_id)
            if not label:
                continue
            for col in range(cell.col + 1, table.ncols):
                target = cells_by_pos.get((cell.row, col))
                if not target:
                    continue
                value = value_by_cell.get(target.cell_id)
                if not value:
                    continue
                col_distance = col - cell.col
                features = {
                    "row_distance": 0.0,
                    "col_distance": float(col_distance),
                    "direction": 1.0,
                    "header_relation": 1.0 if cell.row == 0 else 0.0,
                }
                links.append(
                    LinkCandidate(
                        label=label,
                        value=value,
                        relation="table_row_right",
                        features=features,
                        score=0.0,
                    )
                )
            for row in range(cell.row + 1, table.nrows):
                target = cells_by_pos.get((row, cell.col))
                if not target:
                    continue
                value = value_by_cell.get(target.cell_id)
                if not value:
                    continue
                row_distance = row - cell.row
                features = {
                    "row_distance": float(row_distance),
                    "col_distance": 0.0,
                    "direction": 1.0,
                    "header_relation": 1.0 if cell.col == 0 else 0.0,
                }
                links.append(
                    LinkCandidate(
                        label=label,
                        value=value,
                        relation="table_col_down",
                        features=features,
                        score=0.0,
                    )
                )
    return links


def _distance_between(span_a: tuple[int, int], span_b: tuple[int, int]) -> int:
    if span_a[1] <= span_b[0]:
        return span_b[0] - span_a[1]
    if span_b[1] <= span_a[0]:
        return span_a[0] - span_b[1]
    return 0
