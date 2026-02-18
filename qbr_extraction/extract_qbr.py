#!/usr/bin/env python3
"""Compatibility wrapper for legacy extract_qbr entrypoint.

Use the canonical script instead:
qbr_extraction/qbr_pipeline/pipelines/extraction/scripts/extract_kreuzberg_standalone.py
"""

from __future__ import annotations

import runpy
from pathlib import Path

TARGET = (
    Path(__file__).resolve().parent
    / "qbr_pipeline"
    / "pipelines"
    / "extraction"
    / "scripts"
    / "extract_kreuzberg_standalone.py"
)
runpy.run_path(str(TARGET), run_name="__main__")
