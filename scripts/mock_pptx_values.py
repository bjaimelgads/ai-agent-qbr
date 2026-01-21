#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


NUMBER_RE = re.compile(r"(?<!\d)(\$?)(\d[\d,]*)(?:\.(\d+))?(%?)(?!\d)")


def _format_number(
    value: float,
    decimals: int,
    use_commas: bool,
    pad_width: int | None,
) -> str:
    if decimals > 0:
        fmt = f"{{:{',' if use_commas else ''}.{decimals}f}}"
        formatted = fmt.format(value)
    else:
        rounded = int(round(value))
        formatted = f"{rounded:,}" if use_commas else str(rounded)
    if pad_width:
        formatted = formatted.zfill(pad_width)
    return formatted


def _adjust_number(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        prefix, digits, decimals, suffix = match.groups()
        use_commas = "," in digits
        raw_digits = digits.replace(",", "")
        pad_width = None
        if raw_digits.startswith("0") and len(raw_digits) > 1 and decimals is None:
            pad_width = len(raw_digits)

        value = float(f"{raw_digits}.{decimals}") if decimals else float(raw_digits)

        # Adjustments are deterministic to make diffs easy to inspect.
        if suffix == "%" and value <= 1000:
            value += 2
        elif decimals is None and 2000 <= value <= 2035:
            value += 1
        elif value >= 1000:
            value *= 1.03
        elif value >= 100:
            value += 7
        elif value >= 10:
            value += 3
        else:
            value += 1

        decimal_len = len(decimals) if decimals else 0
        formatted = _format_number(value, decimal_len, use_commas, pad_width)
        return f"{prefix}{formatted}{suffix}"

    return NUMBER_RE.sub(repl, text)


NAMESPACES = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def _register_namespaces() -> None:
    for prefix, uri in NAMESPACES.items():
        ET.register_namespace(prefix, uri)


def _update_xml_file(xml_path: Path) -> bool:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    changed = False

    for tag in ("a:t", "c:v"):
        for elem in root.findall(f".//{tag}", NAMESPACES):
            if not elem.text:
                continue
            updated = _adjust_number(elem.text)
            if updated != elem.text:
                elem.text = updated
                changed = True

    if changed:
        tree.write(xml_path, encoding="UTF-8", xml_declaration=True)
    return changed


def _extract_pptx(input_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    # unzip returns non-zero when a CRC fails, but still extracts files.
    subprocess.run(
        ["unzip", "-o", str(input_path), "-d", str(extract_dir)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _zip_dir(source_dir: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in source_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(source_dir))


def mock_pptx_values(input_path: Path, output_path: Path) -> int:
    _register_namespaces()
    updated_count = 0

    with tempfile.TemporaryDirectory() as tempdir:
        extract_dir = Path(tempdir)
        _extract_pptx(input_path, extract_dir)

        xml_paths = (
            list(extract_dir.glob("ppt/slides/*.xml"))
            + list(extract_dir.glob("ppt/notesSlides/*.xml"))
            + list(extract_dir.glob("ppt/charts/*.xml"))
        )
        for xml_path in xml_paths:
            if _update_xml_file(xml_path):
                updated_count += 1

        _zip_dir(extract_dir, output_path)

    return updated_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a mocked PPTX with adjusted numeric values."
    )
    parser.add_argument("input", type=Path, help="Source PPTX file")
    parser.add_argument("output", type=Path, help="Destination PPTX file")
    args = parser.parse_args()

    updated = mock_pptx_values(args.input, args.output)
    print(f"Updated {updated} shapes/tables with numeric changes.")


if __name__ == "__main__":
    main()
