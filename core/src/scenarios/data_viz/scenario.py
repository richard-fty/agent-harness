"""Data visualization story benchmark scenario."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.runtime.trace import Trace
from scenarios.base import Scenario
from scenarios.data_viz.evaluator import evaluate


class DataVizScenario(Scenario):
    @property
    def name(self) -> str:
        return "data_viz"

    def get_skill_names(self) -> list[str]:
        return ["data_viz"]

    def is_app_artifact(self) -> bool:
        return True

    def get_test_cases(self) -> list[dict[str, Any]]:
        cases_dir = Path(__file__).parent / "cases"
        cases = []
        for path in sorted(cases_dir.glob("case_*.json")):
            cases.append(json.loads(path.read_text(encoding="utf-8")))
        return cases

    def evaluate(self, trace: Trace, test_case: dict[str, Any]) -> dict[str, Any]:
        return evaluate(trace, test_case)
