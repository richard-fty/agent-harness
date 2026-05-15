# Data Viz Eval Plan

Eval harness plan for the `data_viz` skill pack and scenario.

Status: draft
Owner: Apex Agent
Last updated: 2026-05-09

## 1. Purpose

The data-viz harness measures whether the `data_viz` agent does the right work end-to-end: routes correctly, derives metrics from data (not memory), produces a polished interactive report app, and grounds every numeric claim in the source dataset.

Three layers of evaluation, all measured from one agent run:

1. **Process** — skill routing, plan structure, tool argument grounding (e.g., reads the right CSV).
2. **Build & runtime** — install / build / test gates pass; an `app_preview` artifact is created and finalized.
3. **Outcome** — the produced app passes deterministic Playwright assertions against expected metrics, has the required visual structure, and survives a region/segment filter recompute.

Most agents that "sound right" on a data-story prompt fail one of layers 2 or 3. This harness is the line that catches that.

## 2. Current State

What's already built (Phase 1 of `doc/data-viz-story-agent-implementation-guide.md` is largely complete):

- Skill pack: `core/src/skill_packs/data_viz/` (SKILL.md, REFERENCE.md, skill.py, tools.py)
- Scenario: `core/src/scenarios/data_viz/scenario.py` with `is_app_artifact() → True`
- Evaluator: `core/src/scenarios/data_viz/evaluator.py` — five-layer scorecard with replay-mode handling
- Template: `core/src/scenarios/data_viz/templates/vite-data-story/`
- One case: `core/src/scenarios/data_viz/cases/case_001.json` (`sales_story_001` — DV1)
- Case fixtures: `cases/case_001/{public,hidden,golden}/` populated
- Runner integration: `run_app_artifact_benchmark` in `core/src/eval/runner.py:515` already handles install/build/test gates, hidden-asset injection, artifact capture, and one-shot evaluation

What the existing evaluator (`scenarios/data_viz/evaluator.py:evaluate`) measures:

- replay_mode auto-detected from empty `tool_calls` / `skill_loads` / `plan_updates`
- five layer scores via `LAYER_WEIGHTS = {skill_planner: 0.10, artifact_runtime: 0.15, data_correctness: 0.40, viz_interaction: 0.20, story_layout_accessibility: 0.15}`
- Playwright JSON parsed for per-assertion pass/fail bucketed by layer
- final result includes `replay_total_score` separate from agent-mode `total_score`

What's missing:

- Only one case (DV1). The implementation guide names DV2 (Support Operations) and DV3 (Marketing ROI) — not yet authored.
- Comparator extension to diff per-layer scores in `core/src/eval/comparator.py` (not yet present).
- No baseline at `core/src/eval/baselines/data_viz.json`.
- No actual-agent run tracked yet (only replay against golden patch).

## 3. Evaluation Model

The scorecard already exists in code; this section documents intent and weights so the doc and code stay aligned.

| Layer | Weight | What It Proves |
|---|---:|---|
| Skill / planner compliance | 10% | The agent loaded `data_viz` and produced a plan covering inspect → derive → build → filter → verify → preview |
| Artifact / runtime health | 15% | Install + build gates pass, app loads in Playwright, `app_preview` artifact is created and finalized |
| Data correctness | 40% | Headline KPIs, chart values, filter recomputation, ranks, missing-data handling all match `expected_metrics.json` |
| Visualization / interaction | 20% | Chart types match the question, axes/legends present, filter changes visible values, no horizontal overflow |
| Story / layout / accessibility | 15% | Headline traceable to a metric, `<main>` landmark, mobile screenshot has no overlap, accessible names on controls |

Scorecard contract (already returned by `evaluator.evaluate`):

```python
# core/src/scenarios/data_viz/evaluator.py:evaluate (existing)
{
  "test_case_id": "sales_story_001",
  "total_score": 0.86,
  "replay_total_score": None,            # set when replay_mode == True
  "skill_planner_score": 0.9,            # None in replay mode
  "artifact_runtime_score": 1.0,
  "data_correctness_score": 0.82,
  "viz_interaction_score": 0.8,
  "story_layout_accessibility_score": 0.78,
  "install_passed": True,
  "build_passed": True,
  "test_passed": True,
  "app_preview_created": True,
  "details": {
    "missing_plan_concepts": [],
    "metric_failures": ["region_share_apac"],
    "viz_failures": [],
    "story_failures": [],
    "runtime_failures": [],
    "playwright_report_found": True,
    "replay_mode": False,
  },
}
```

## 3.5 Mapping To Existing Harness

The eval reads from harness surfaces that already exist. None of the references below are aspirational.

### Scenario contract

```python
# core/src/scenarios/data_viz/scenario.py:DataVizScenario (existing)
class DataVizScenario(Scenario):
    @property
    def name(self) -> str: return "data_viz"
    def get_skill_names(self) -> list[str]: return ["data_viz"]
    def is_app_artifact(self) -> bool: return True   # routes through run_app_artifact_benchmark
    def get_test_cases(self) -> list[dict[str, Any]]: ...
    def evaluate(self, trace: Trace, test_case: dict) -> dict:
        return evaluate(trace, test_case)            # delegates to evaluator.py
```

`is_app_artifact() == True` is what dispatches the case through `core/src/eval/runner.py:run_app_artifact_benchmark` — the gated app runner that installs, builds, copies hidden Playwright fixtures only at test time, and captures patch / logs / Playwright JSON / screenshots / preview URL.

### Trace surfaces by scoring layer

| Layer | Reads from | Source of truth |
|---|---|---|
| Skill / planner | `trace.skill_loads`, `trace.plan_updates` | Recorded by `Trace.record_skill_load` (`trace.py:167`) and `Trace.record_plan_update` (`trace.py:159`); populated by `_map_event_to_trace` in `managed_runtime.py` |
| Artifact / runtime | `trace.gate_results`, `trace.artifact_events` | `gate_results` populated by `_run_app_artifact_gates` (`runner.py:_run_app_artifact_gates`); `artifact_events` populated by the `ToolContext.trace` bridge in `tool_context.py:115, 141, 166, 191` |
| Data / viz / story | Playwright JSON parsed from `trace.artifacts` | Captured by `_capture_app_artifact_outputs` (`runner.py:_capture_app_artifact_outputs`); JSON extracted from `.gate_logs/test.stdout` via `_extract_playwright_json` (`runner.py:_extract_playwright_json`); per-assertion pass/fail bucketed by layer in `evaluator.py:_layer_checks` |
| Tool argument grounding (e.g., dataset path) | `trace.tool_calls` | Already populated by `Trace.record_tool_call` (`trace.py:134`) |

The Playwright spec assigns each test to a layer via title prefix or tag. Existing convention in `cases/case_001/hidden/dashboard.spec.mjs` uses `data:`, `viz:`, `story:` prefixes; the evaluator's `_layer_checks` reads those.

### Concrete lookup snippets

Skill / planner score — already implemented; this is the existing function for reference, not new code:

```python
# core/src/scenarios/data_viz/evaluator.py:_score_skill_planner (existing)
def _score_skill_planner(trace: Trace, test_case: dict) -> float:
    skill_loaded = any(s["skill_name"] == "data_viz" for s in trace.skill_loads)
    plan_exists = bool(trace.plan_updates)
    required = set(test_case.get("required_plan_steps", []))
    found = _plan_concepts(trace)
    plan_coverage = (len(required & found) / len(required)) if required else 1.0
    return (
        0.30 * float(skill_loaded)
      + 0.25 * float(plan_exists)
      + 0.25 * plan_coverage
      + 0.20 * float(_plan_status_progressed(trace))
    )
```

Artifact / runtime score — combines gate results with artifact-event observation:

```python
# core/src/scenarios/data_viz/evaluator.py:_score_artifact_runtime (existing)
def _score_artifact_runtime(trace, *, install_passed, build_passed, test_results, replay_mode) -> float:
    score = 0.0
    score += 0.25 * float(install_passed)
    score += 0.25 * float(build_passed)
    score += 0.20 * float(bool(test_results))           # Playwright at least executed
    if not replay_mode:
        score += 0.15 * float(_has_app_preview(trace))  # waived in replay (no agent ran)
    score += 0.15 * float(not _layer_has_failure(trace, "runtime"))
    return score


def _has_app_preview(trace: Trace) -> bool:
    """Walk trace.artifact_events for kind=app_preview that finalized."""
    created_ids = {
        ev["data"].get("artifact_id")
        for ev in trace.artifact_events
        if ev["event_type"] == "artifact_created"
        and ev["data"].get("kind") == "app_preview"
    }
    finalized_ids = {
        ev["data"].get("artifact_id")
        for ev in trace.artifact_events
        if ev["event_type"] == "artifact_finalized"
    }
    return bool(created_ids & finalized_ids)
```

### Replay-mode contract

Replay mode (`--replay sales_story_001` on the runner CLI) applies a golden patch and runs gates without invoking the agent. The evaluator detects this from empty `trace.tool_calls`/`trace.skill_loads`/`trace.plan_updates` and:

- waives `skill_planner_score` (returns `None`)
- waives the `app_preview` sub-signal of `artifact_runtime_score`
- renormalizes weights into `replay_total_score`
- leaves `total_score` set to the same renormalized number for backwards compatibility with the comparator

Phase 1 exit criterion: `--replay sales_story_001` returns `replay_total_score == 1.0`. Currently true.

### Comparator + baseline path

Same hooks as every other scenario in the harness:

- `core/src/eval/comparator.py:compare_against_baseline(results, baseline)` — diff totals
- `core/src/eval/comparator.py:format_regression_gate(gate)` — render the regression message
- `core/src/eval/baselines/data_viz.json` — TO ADD once Phase 2 is green; consumed by the existing `--baseline` and `--update-baseline` flags on the runner

For per-layer regression diffs (e.g., "data_correctness dropped from 0.92 to 0.71"), `comparator.py` needs the same extension described in §9.1 of the data-viz implementation guide. Once that lands, both `data_viz` and `stock_strategy` scorecards benefit.

### Mock mode and CI

App-artifact scenarios bypass `--mock` because the install/build/test gates execute real shell commands inside a sandbox. Replay (`--replay <case_id>`) is the deterministic path for CI. Don't add `--mock` support to gates.

## 4. Step-Level Checks

The harness reads trace events, not only the final app.

### 4.1 Routing

Pass:

```text
"Build a sales story from sales.csv"  → data_viz loaded
"Show me how revenue split by region" → data_viz loaded (chart-shaped intent)
"What's the weather"                  → data_viz NOT loaded
"List my open files"                  → data_viz NOT loaded (operational, not analytical)
```

Forbidden co-loads (negative routing):

```text
data_viz must NOT load alongside stock_strategy unless the prompt explicitly mixes both
```

### 4.2 Required plan structure

The case manifest declares `required_plan_steps`:

```json
"required_plan_steps": [
  "inspect dataset",
  "derive metrics",
  "build story layout",
  "add filters",
  "verify calculations",
  "start preview"
]
```

Evaluator's `_plan_concepts(trace)` extracts concept tokens from `trace.plan_updates` payloads and intersects with the required set. Missing concepts surface as `details.missing_plan_concepts`.

### 4.3 Tool argument grounding

For data-viz, the most important argument check is **the agent reads the correct dataset path**. From `cases/case_001/public/sales.csv`:

```python
# evaluator addition for tool argument grounding (TO ADD)
def _score_dataset_grounded(trace: Trace, test_case: dict) -> bool:
    expected = set(test_case.get("public_data", []))
    if not expected:
        return True
    referenced = {
        c["arguments"].get("path")
        for c in trace.tool_calls
        if c["name"] in {"read_file", "profile_dataset"}
    }
    return bool(expected & referenced)
```

This is currently an implicit pass — Phase 2 promotes it to an explicit sub-check inside `_score_skill_planner` or a new `process_grounding` micro-layer.

### 4.4 Ordering constraints (partial-order)

```text
inspect dataset    must precede   derive metrics
derive metrics     must precede   build story layout
build story layout must precede   start preview
```

Evaluator can enforce these by scanning `trace.plan_updates` for status changes in order, or by inspecting `trace.tool_calls` for the corresponding tool sequence. Loose enforcement only — strict sequence overfits.

### 4.5 Recovery

For prompts with deliberately ambiguous data ("revenue but no margin column"):

- Run should not crash.
- Final output should explain the limitation.
- Agent should not fabricate the missing column.

Test by adding a DV-edge case with a column-truncated CSV; expect `data_correctness_score` low, `safety/viz_interaction` unaffected, `final_output` mentions the missing column.

## 5. Outcome Checks

### 5.1 Headline KPIs

Each case's `expected_metrics.json` contains the deterministic ground truth. Playwright reads `data-value` attributes from rendered DOM and compares.

```html
<!-- vite-data-story template convention -->
<section data-testid="kpi-total-revenue" data-value="4311320.46">
  $4.31M revenue
</section>
```

Failures land in `details.metric_failures` keyed by `data-testid`.

### 5.2 Chart structure

```text
data-testid="monthly-revenue-chart"      data-points="12"
data-testid="region-revenue-chart"       data-bars="<n_regions>"
data-testid="category-revenue-chart"     data-bars="<n_categories>"
```

Failures land in `details.viz_failures`.

### 5.3 Filter recomputation

Playwright drives the region filter and re-asserts KPIs:

```text
select "North America" from #region-filter
expect [data-testid="kpi-total-revenue"][data-value] = expected_metrics.region_filtered_total["North America"]
```

The agent that hardcodes KPIs will pass §5.1 but fail §5.3. This is the bench's most important check.

### 5.4 Story / layout / accessibility

Deterministic-only for MVP:

```text
- <main> landmark exists
- headline <h1> contains the top-region name
- mobile (375×812) viewport: no horizontal overflow
- every <button> and <select> has an accessible name
- chart containers have data-summary attribute
```

Subjective checks (does the headline match the strongest finding?) are deferred to an optional LLM-judge pass.

## 6. Test Case Schema

Existing case shape (`cases/case_001.json`) plus the fields the evaluator reads:

```json
{
  "id": "support_ops_002",
  "title": "Where are SLA breaches coming from",
  "tier": "DV2",
  "template": "vite-data-story",
  "input": "...",
  "public_data": ["cases/case_002/public/tickets.csv"],
  "hidden_playwright_spec": "cases/case_002/hidden/dashboard.spec.mjs",
  "expected_metrics": "cases/case_002/hidden/expected_metrics.json",
  "golden_patch": "cases/case_002/golden/patch.diff",
  "allowed_paths": ["src/**", "package.json", "index.html"],
  "required_plan_steps": [
    "inspect dataset", "derive metrics", "build story layout",
    "add filters", "verify calculations", "start preview"
  ],
  "manifest": {
    "install": ["pnpm", "install", "--frozen-lockfile", "--prefer-offline"],
    "build": ["pnpm", "build"],
    "test": ["pnpm", "exec", "playwright", "test",
             "cases/case_002/hidden/dashboard.spec.mjs", "--reporter=json"],
    "budget": { "max_steps": 40, "max_cost_usd": 1.0, "max_wall_sec": 420 },
    "limits": { "memory_mb": 1024, "cpus": 1.0, "max_stdout_kb": 2048 }
  }
}
```

No new fields versus DV1 — the schema is stable.

## 7. Scenario Set

Status of each case:

| Case | Tier | Status | Purpose |
|---|---|---|---|
| `sales_story_001` | DV1 | ✅ exists | Core happy path: monthly trend, region split, top categories, filter |
| `support_ops_002` | DV2 | ❌ to add | SLA breach analysis: rate computation, team filter |
| `marketing_roi_003` | DV3 | ❌ to add | CAC / ROAS formulas: best-vs-worst channel claim, no division-by-zero |
| `bad_data_004` | DV-edge | ❌ to add | Missing column / partial data — recovery, no fabrication |
| `no_skill_needed_005` | DV-neg | ❌ to add | Generic file-listing prompt — `data_viz` must NOT load |

DV2 and DV3 are spec'd in `doc/data-viz-story-agent-implementation-guide.md` §10. Edge and negative cases are new — author them when broadening coverage past three.

## 8. Implementation Plan

### Phase 1 — Replay green ✅

Largely done:

- DV1 case fixtures populated.
- Evaluator returns `replay_total_score = 1.0` on `--replay sales_story_001`.
- Hidden assets absent from workspace before/after the test gate.

Remaining nit: confirm by running `uv run python -m eval --scenario data_viz --replay sales_story_001` and asserting `replay_total_score == 1.0`. If not, fix golden patch / expected_metrics until it is.

### Phase 2 — Agent runs

- Run the actual agent on DV1 (no `--replay`); collect failures.
- Tune `core/src/skill_packs/data_viz/SKILL.md` only when failure traces show a workflow gap (do NOT tune to make any specific test pass).
- Author DV2 + DV3 cases following DV1's `public/` / `hidden/` / `golden/` layout.
- Confirm `_capture_app_artifact_outputs` (`runner.py:_capture_app_artifact_outputs`) stores patch diff, install/build/test logs, Playwright JSON, screenshots, and preview URL for each.

Exit criteria:

- DV1, DV2, DV3 each return a non-zero `total_score` from a real agent run.
- Every failure attributable to exactly one layer score.
- Workspace audit confirms hidden specs are not visible to the agent before completion.

### Phase 3 — Comparator + baseline

- Extend `core/src/eval/comparator.py` to diff per-layer scores in addition to `total_score`. Same change benefits stock-strategy.
- Freeze `core/src/eval/baselines/data_viz.json` from a stable Phase 2 run.
- Add the regression gate to CI: fail PRs that drop `total_score` by >5% or any individual layer below 0.7 on DV1.

### Phase 4 — Coverage expansion

- Add DV-edge (bad data) and DV-neg (no_skill_needed) cases.
- Optionally: LLM-judge pass on the headline-matches-finding check, weighted in at 5% under story_layout_accessibility.

## 9. Pass / Fail Semantics

Hard fail (case scores 0 regardless of layer breakdown):

- `data_viz` skill not loaded for an explicit data-story prompt
- `data_viz` loaded for `no_skill_needed` prompt (when DV-neg case ships)
- Install or build gate fails
- App preview artifact never created (agent mode only — waived in replay)
- Hidden specs visible in workspace before test gate (process integrity violation)

Soft penalties (layer score reduction, not full case fail):

- `data_correctness` < 0.6 — KPIs or filter recompute wrong
- `viz_interaction` < 0.6 — chart container or label assertions fail
- `story_layout_accessibility` < 0.6 — overflow or missing landmark
- Step count above `manifest.budget.max_steps`
- Cost above `manifest.budget.max_cost_usd`

The 40% weight on data correctness means a clean build with wrong numbers caps the case at ~0.6 — which is the right signal: the app shipped, but it's lying.

## 10. Interview Summary

```text
The data-viz harness evaluates the agent end-to-end on a five-layer scorecard:
process (10%), runtime (15%), data correctness (40%), visualization (20%),
and story/layout (15%). Process reads trace.skill_loads / plan_updates / tool_calls
from the runtime trace. Runtime reads gate_results and artifact_events. Outcome
runs Playwright inside a sandbox against hidden expected_metrics.json that the
agent never sees, and parses the JSON report into per-layer pass/fail buckets.
Data correctness — the heaviest layer — checks that headline KPIs and filter
recomputations match the dataset, which is the test most agents fail by
hardcoding values that look plausible. Replay mode applies a golden patch and
verifies the bench itself works without invoking the agent. Comparator + baseline
gate regressions per-layer so a drop is attributable to process, runtime,
data, viz, or story rather than a single mystery total.
```
