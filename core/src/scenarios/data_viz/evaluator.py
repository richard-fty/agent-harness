"""Evaluator for data visualization story app cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.runtime.trace import Trace

LAYER_WEIGHTS = {
    "skill_planner_score": 0.10,
    "artifact_runtime_score": 0.15,
    "data_correctness_score": 0.40,
    "viz_interaction_score": 0.20,
    "story_layout_accessibility_score": 0.15,
}


def evaluate(trace: Trace, test_case: dict[str, Any]) -> dict[str, Any]:
    gates = trace.gate_results
    install_passed = bool(gates.get("install"))
    build_passed = bool(gates.get("build"))
    test_passed = bool(gates.get("test"))
    replay_mode = not trace.tool_calls and not trace.skill_loads and not trace.plan_updates

    test_results = _load_playwright_results(trace)
    layer_checks = _layer_checks(test_results)

    skill_planner_score: float | None = None if replay_mode else _score_skill_planner(trace, test_case)
    artifact_runtime_score = _score_artifact_runtime(
        trace,
        install_passed=install_passed,
        build_passed=build_passed,
        test_results=test_results,
        replay_mode=replay_mode,
    )
    data_correctness_score = _score_checks(layer_checks.get("data", []))
    viz_interaction_score = _score_checks(layer_checks.get("viz", []))
    story_layout_accessibility_score = _score_checks(layer_checks.get("story", []))

    if replay_mode:
        weighted = {
            "artifact_runtime_score": artifact_runtime_score,
            "data_correctness_score": data_correctness_score,
            "viz_interaction_score": viz_interaction_score,
            "story_layout_accessibility_score": story_layout_accessibility_score,
        }
        weights = {
            key: weight for key, weight in LAYER_WEIGHTS.items()
            if key in weighted
        }
        total_score = _weighted(weighted, weights)
        replay_total_score = total_score
    else:
        total_score = _weighted(
            {
                "skill_planner_score": skill_planner_score or 0.0,
                "artifact_runtime_score": artifact_runtime_score,
                "data_correctness_score": data_correctness_score,
                "viz_interaction_score": viz_interaction_score,
                "story_layout_accessibility_score": story_layout_accessibility_score,
            },
            LAYER_WEIGHTS,
        )
        replay_total_score = None

    return {
        "test_case_id": test_case["id"],
        "total_score": round(total_score, 3),
        "replay_total_score": None if replay_total_score is None else round(replay_total_score, 3),
        "skill_planner_score": None if skill_planner_score is None else round(skill_planner_score, 3),
        "artifact_runtime_score": round(artifact_runtime_score, 3),
        "data_correctness_score": round(data_correctness_score, 3),
        "viz_interaction_score": round(viz_interaction_score, 3),
        "story_layout_accessibility_score": round(story_layout_accessibility_score, 3),
        "install_passed": install_passed,
        "build_passed": build_passed,
        "test_passed": test_passed,
        "app_preview_created": _has_app_preview(trace),
        "steps": trace.step_count,
        "tokens": trace.total_usage.total_tokens,
        "cost_usd": trace.total_usage.cost_usd,
        "duration_seconds": trace.duration_seconds,
        "details": {
            "missing_plan_concepts": _missing_plan_concepts(trace, test_case),
            "metric_failures": _failed_titles(layer_checks.get("data", [])),
            "viz_failures": _failed_titles(layer_checks.get("viz", [])),
            "story_failures": _failed_titles(layer_checks.get("story", [])),
            "runtime_failures": _failed_titles(layer_checks.get("runtime", [])),
            "playwright_report_found": bool(test_results),
            "replay_mode": replay_mode,
        },
    }


def _weighted(scores: dict[str, float], weights: dict[str, float]) -> float:
    denom = sum(weights.values())
    if denom <= 0:
        return 0.0
    return sum(scores.get(key, 0.0) * weight for key, weight in weights.items()) / denom


def _score_skill_planner(trace: Trace, test_case: dict[str, Any]) -> float:
    score = 0.0
    if any(item.get("skill_name") == "data_viz" for item in trace.skill_loads):
        score += 0.30
    if trace.plan_updates:
        score += 0.25
    missing = _missing_plan_concepts(trace, test_case)
    required = test_case.get("required_plan_steps", [])
    if required:
        score += 0.25 * ((len(required) - len(missing)) / len(required))
    else:
        score += 0.25
    if _plan_status_progressed(trace):
        score += 0.20
    return min(1.0, score)


def _score_artifact_runtime(
    trace: Trace,
    *,
    install_passed: bool,
    build_passed: bool,
    test_results: list[dict[str, Any]],
    replay_mode: bool,
) -> float:
    score = 0.0
    score += 0.25 * int(install_passed)
    score += 0.25 * int(build_passed)
    score += 0.20 * int(_score_checks(_layer_checks(test_results).get("runtime", [])) == 1.0)
    if replay_mode:
        score += 0.15
    else:
        score += 0.15 * int(_has_app_preview(trace))
    score += 0.15 * int(not _failed_titles(_layer_checks(test_results).get("runtime", [])))
    return min(1.0, score)


def _load_playwright_results(trace: Trace) -> list[dict[str, Any]]:
    for artifact in trace.artifacts:
        if artifact.get("kind") == "test_report" and artifact.get("format") == "playwright_json":
            path = artifact.get("path")
            if path:
                try:
                    payload = json.loads(Path(path).read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return []
                return list(_iter_specs(payload))
    return []


def _iter_specs(node: Any) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if "title" in node and "ok" in node:
            specs.append(node)
        for key in ("suites", "specs", "tests"):
            value = node.get(key)
            if isinstance(value, list):
                for item in value:
                    specs.extend(_iter_specs(item))
    elif isinstance(node, list):
        for item in node:
            specs.extend(_iter_specs(item))
    return specs


def _layer_checks(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    layers = {"runtime": [], "data": [], "viz": [], "story": []}
    for item in results:
        title = str(item.get("title", ""))
        lowered = title.lower()
        for layer in layers:
            if lowered.startswith(f"[{layer}]"):
                layers[layer].append(item)
                break
    return layers


def _score_checks(checks: list[dict[str, Any]]) -> float:
    if not checks:
        return 0.0
    passed = sum(1 for item in checks if bool(item.get("ok")))
    return passed / len(checks)


def _failed_titles(checks: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("title", "")) for item in checks if not bool(item.get("ok"))]


def _missing_plan_concepts(trace: Trace, test_case: dict[str, Any]) -> list[str]:
    required = [str(item).lower() for item in test_case.get("required_plan_steps", [])]
    if not required:
        return []
    plan_text = " ".join(
        json.dumps(update.get("payload", {}), sort_keys=True).lower()
        for update in trace.plan_updates
    )
    return [item for item in required if item not in plan_text]


def _plan_status_progressed(trace: Trace) -> bool:
    statuses = set()
    for update in trace.plan_updates:
        payload = update.get("payload", {})
        for item in payload.get("tasks", []):
            if isinstance(item, dict):
                statuses.add(item.get("status"))
    return any(status in statuses for status in {"in_progress", "completed", "failed"})


def _has_app_preview(trace: Trace) -> bool:
    if any(item.get("kind") == "app_preview" for item in trace.artifacts):
        return True
    return any(
        event.get("event_type") == "artifact_created"
        and event.get("data", {}).get("kind") == "app_preview"
        for event in trace.artifact_events
    )
