from __future__ import annotations

from qbr_intelligence.metrics.candidates import build_candidates
from qbr_intelligence.metrics.linking import link_table_candidates
from qbr_intelligence.metrics.models import Table, TableCell


def test_table_row_linking_prefers_right_cell() -> None:
    cells = (
        TableCell(slide_index=1, table_id="t1", row=0, col=0, text="CTR", cell_id="t1:r0c0"),
        TableCell(slide_index=1, table_id="t1", row=0, col=1, text="2.1%", cell_id="t1:r0c1"),
        TableCell(slide_index=1, table_id="t1", row=1, col=0, text="Spend", cell_id="t1:r1c0"),
        TableCell(slide_index=1, table_id="t1", row=1, col=1, text="$10K", cell_id="t1:r1c1"),
    )
    table = Table(slide_index=1, table_id="t1", cells=cells, nrows=2, ncols=2)

    labels = []
    values = []
    for cell in cells:
        bundle = build_candidates(
            slide_index=1,
            source_type="table_cell",
            block_id=cell.cell_id,
            text=cell.text,
        )
        labels.extend(bundle.labels)
        values.extend(bundle.values)

    links = link_table_candidates([table], labels, values)
    assert any(link.relation == "table_row_right" for link in links)
    assert any(
        link.label.metric_id == "click_through_rate" and link.value.raw_value_text == "2.1%"
        for link in links
    )
