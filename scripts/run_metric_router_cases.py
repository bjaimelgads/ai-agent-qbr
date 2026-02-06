"""Run metric router cases and emit results."""

from __future__ import annotations

import json
import os
from pathlib import Path

import importlib.util
import sys


def _load_metric_router(repo_root: Path):
    os.environ.setdefault("QBR_INTELLIGENCE_LIGHT_IMPORT", "1")
    module_path = repo_root / "src" / "ai_agent_qbr" / "infrastructure" / "metric_router.py"
    spec = importlib.util.spec_from_file_location("metric_router", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load metric router from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["metric_router"] = module
    spec.loader.exec_module(module)
    return module.MetricQueryRouter


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    cases_path = repo_root / "tests" / "metric_router_cases.json"
    if not cases_path.exists():
        raise SystemExit(f"Missing cases file: {cases_path}")

    cases = json.loads(cases_path.read_text())
    MetricQueryRouter = _load_metric_router(repo_root)
    router = MetricQueryRouter()
    results = []
    failures = 0

    for case in cases:
        query = case.get("query", "")
        expected = bool(case.get("expected_route", False))
        actual = bool(router.is_metric_query(query))
        passed = expected == actual
        if not passed:
            failures += 1
        results.append(
            {
                "id": case.get("id"),
                "query": query,
                "expected_route": expected,
                "actual_route": actual,
                "passed": passed,
                "source_deck": case.get("source_deck"),
                "note": case.get("note"),
            }
        )

    report = {
        "total": len(results),
        "passed": len(results) - failures,
        "failed": failures,
        "results": results,
    }

    out_dir = repo_root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metric_router_cases_results.json"
    out_path.write_text(json.dumps(report, indent=2))

    print(f"Cases: {report['total']}  Passed: {report['passed']}  Failed: {report['failed']}")
    print(f"Results saved to: {out_path}")
    if failures:
        print("Failed cases:")
        for item in results:
            if not item["passed"]:
                print(f"- {item['id']}: expected={item['expected_route']} actual={item['actual_route']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
