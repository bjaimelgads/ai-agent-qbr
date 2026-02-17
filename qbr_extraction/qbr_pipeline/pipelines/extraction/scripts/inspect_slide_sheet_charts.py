#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from dotenv import load_dotenv

from qbr_intelligence.infrastructure.google_slides import GoogleSlidesClient
from qbr_intelligence.infrastructure.google_sheets import GoogleSheetsClient


def _extract_presentation_id(value: str) -> str:
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", value)
    if not m:
        raise ValueError("Unable to parse presentation id from URL")
    return m.group(1)


def _extract_slide_id(value: str) -> str | None:
    m = re.search(r"slide=id\.([a-zA-Z0-9_]+)", value)
    if m:
        return m.group(1)
    m = re.search(r"#slide=id\.([a-zA-Z0-9_]+)", value)
    if m:
        return m.group(1)
    return None


def _col_label(idx: int) -> str:
    # 0 -> A, 25 -> Z, 26 -> AA
    idx += 1
    out = []
    while idx:
        idx, rem = divmod(idx - 1, 26)
        out.append(chr(ord("A") + rem))
    return "".join(reversed(out))


def _grid_to_a1(grid: dict, sheet_title: str) -> str:
    start_row = int(grid.get("startRowIndex", 0))
    end_row = int(grid.get("endRowIndex", start_row + 1))
    start_col = int(grid.get("startColumnIndex", 0))
    end_col = int(grid.get("endColumnIndex", start_col + 1))

    start = f"{_col_label(start_col)}{start_row + 1}"
    end = f"{_col_label(max(end_col - 1, start_col))}{max(end_row, start_row + 1)}"
    return f"'{sheet_title}'!{start}:{end}"


def _collect_grid_ranges(spec: dict) -> list[dict]:
    ranges: list[dict] = []

    def walk(x):
        if isinstance(x, dict):
            if "sheetId" in x and (
                "startRowIndex" in x or "endRowIndex" in x or "startColumnIndex" in x or "endColumnIndex" in x
            ):
                ranges.append(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for item in x:
                walk(item)

    walk(spec)
    return ranges


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect linked Google Sheets charts from a Google Slides slide."
    )
    parser.add_argument("--slide-url", required=True)
    parser.add_argument("--out", help="Optional JSON output path")
    args = parser.parse_args()

    load_dotenv(Path(".env"))

    pres_id = _extract_presentation_id(args.slide_url)
    slide_id = _extract_slide_id(args.slide_url)
    if not slide_id:
        raise SystemExit("Could not parse slide id from URL")

    slides_client = GoogleSlidesClient.from_env(base_dir=Path("."))
    if not slides_client:
        raise SystemExit("Google Slides auth not configured")
    sheets_client = GoogleSheetsClient.from_env(base_dir=Path("."))
    if not sheets_client:
        raise SystemExit("Google Sheets auth not configured")

    presentation = (
        slides_client._service.presentations()  # noqa: SLF001
        .get(presentationId=pres_id)
        .execute()
    )
    target_slide = None
    for slide in presentation.get("slides", []):
        if slide.get("objectId") == slide_id:
            target_slide = slide
            break
    if not target_slide:
        raise SystemExit(f"Slide id not found: {slide_id}")

    page_elements = target_slide.get("pageElements", [])
    chart_refs = []
    for element in page_elements:
        sc = element.get("sheetsChart")
        if not sc:
            continue
        spreadsheet_id = sc.get("spreadsheetId")
        chart_id = sc.get("chartId")
        if spreadsheet_id and chart_id is not None:
            chart_refs.append(
                {
                    "element_id": element.get("objectId"),
                    "spreadsheet_id": spreadsheet_id,
                    "chart_id": int(chart_id),
                }
            )

    output = {
        "presentation_id": pres_id,
        "slide_id": slide_id,
        "chart_refs": [],
    }

    for ref in chart_refs:
        spreadsheet = sheets_client.get_spreadsheet(ref["spreadsheet_id"])
        sheets_by_id = {
            int(s.get("properties", {}).get("sheetId")): s.get("properties", {}).get("title", "Sheet1")
            for s in spreadsheet.get("sheets", [])
            if s.get("properties", {}).get("sheetId") is not None
        }
        chart = sheets_client.get_chart(ref["spreadsheet_id"], ref["chart_id"])
        if not chart:
            output["chart_refs"].append({**ref, "error": "chart_not_found_in_spreadsheet"})
            continue

        spec = chart.get("spec", {})
        grid_ranges = _collect_grid_ranges(spec)
        a1_ranges = []
        for grid in grid_ranges:
            sheet_id = int(grid.get("sheetId", -1))
            sheet_title = sheets_by_id.get(sheet_id, f"sheet_{sheet_id}")
            try:
                a1_ranges.append(_grid_to_a1(grid, sheet_title))
            except Exception:
                continue

        values_by_range = {}
        for a1 in sorted(set(a1_ranges)):
            try:
                values_by_range[a1] = sheets_client.get_values(ref["spreadsheet_id"], a1)
            except Exception as exc:
                values_by_range[a1] = {"error": str(exc)}

        output["chart_refs"].append(
            {
                **ref,
                "chart_title": spec.get("title"),
                "chart_subtitle": spec.get("subtitle"),
                "a1_ranges": sorted(set(a1_ranges)),
                "values_by_range": values_by_range,
            }
        )

    payload = json.dumps(output, indent=2, ensure_ascii=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        print(f"Wrote: {out_path}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
