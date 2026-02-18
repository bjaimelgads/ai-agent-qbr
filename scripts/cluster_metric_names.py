#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / 'qbr_extraction' / 'qbr_pipeline' / 'pipelines' / 'extraction' / 'scripts' / 'cluster_metric_names.py'
runpy.run_path(str(TARGET), run_name='__main__')
