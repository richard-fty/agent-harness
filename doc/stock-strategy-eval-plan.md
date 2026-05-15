# Stock Strategy Eval Plan

Eval harness plan for the `stock_strategy` skill pack and scenario.

Status: draft  
Owner: Apex Agent  
Last updated: 2026-05-09

## 1. Purpose

The stock analysis harness should measure whether the `stock_strategy` agent does the right work, not just whether the final answer sounds plausible.

The harness evaluates two layers:

1. **Process quality** — skill routing, tool calls, arguments, ordering, recovery, and efficiency.
2. **Outcome quality** — final report or artifact structure, ticker coverage, numeric grounding, risk language, and unsupported-claim avoidance.

This keeps the ReAct loop measurable. ReAct is the execution pattern; the harness is the evaluation layer around it.

## 2. Current State

Current stock scenario files:

- `core/src/scenarios/stock_strategy/scenario.py`
- `core/src/scenarios/stock_strategy/evaluator.py`
- `core/src/scenarios/stock_strategy/test_cases.json`
- `core/src/scenarios/lt1_equity_briefing/evaluator.py`

The current `stock_strategy` evaluator measures:

- expected tool names appeared in the trace
- final output contains required keywords
- run completed successfully
- step count stayed under budget

That is useful, but shallow. It does not yet strongly verify:

- tool arguments match the requested ticker and time range
- indicator values are grounded in tool outputs
- report structure is complete
- risk/compliance language is present
- output avoids unsupported buy/sell certainty
- stale or unrelated skills were not loaded

The `lt1_equity_briefing` scenario is stronger because it parses a generated `.docx` and checks headings, images, hyperlinks, supporting artifacts, and web-research budget. The stock harness should adopt the same style for markdown/report outputs.

## 3. Evaluation Model

Use a weighted scorecard:

| Category | Weight | What It Checks |
|---|---:|---|
| Routing | 10% | Correct skill loaded, unrelated skills avoided |
| Workflow / Tools | 30% | Required tools called with correct arguments and useful ordering |
| Intermediate Grounding | 20% | Tool outputs support later claims and use the requested ticker |
| Output / Artifact | 25% | Final answer or report has required structure and content |
| Safety / Compliance | 10% | Risk language, no guaranteed returns, no overconfident advice |
| Efficiency | 5% | Step, cost, and duplicate-call budgets |

The evaluator should return:

```json
{
  "total_score": 0.87,
  "scores": {
    "routing": 1.0,
    "workflow": 0.9,
    "grounding": 0.8,
    "output": 0.85,
    "safety": 1.0,
    "efficiency": 0.75
  },
  "details": {
    "tools_called": [],
    "missing_tools": [],
    "argument_mismatches": [],
    "required_sections_missing": [],
    "unsupported_claims": []
  }
}
```

## 3.5 Mapping To Existing Harness

The plan above is implementation-ready when each scoring function reads from the trace and harness surfaces that already exist in this repo. None of these are new — they were added in the refresh/reconnect work and the data-viz scenario rollout. Wire to them rather than reinvent.

### Scenario contract

```python
# core/src/scenarios/stock_strategy/scenario.py

class StockStrategyScenario(Scenario):
    @property
    def name(self) -> str: return "stock_strategy"

    def get_skill_names(self) -> list[str]: return ["stock_strategy"]

    def is_app_artifact(self) -> bool:
        # Stock analysis produces a report/markdown artifact, NOT a runnable app.
        # Returning False keeps it on the regular benchmark path; do not copy
        # data_viz which returns True and routes through run_app_artifact_benchmark.
        return False

    def evaluate(self, trace: Trace, test_case: dict) -> dict:
        return aggregate_scorecard(trace, test_case)   # see §3
```

### Trace surfaces by scoring category

The runtime already populates these fields on `Trace` (see `core/src/agent/runtime/trace.py`). Each scoring function reads from one or two of them:

| Category | Reads from | Source of truth |
|---|---|---|
| Routing | `trace.skill_loads` | `_map_event_to_trace` records every `skill_auto_loaded` event into `Trace.record_skill_load` (`managed_runtime.py:1228`, recorder at `trace.py:167`) |
| Workflow / Tools | `trace.tool_calls` | `record_tool_call` ledger written from `tool_finished` (`managed_runtime.py:1257`, recorder at `trace.py:134`); each entry has `name`, `arguments`, `success`, `duration_ms`, `result_size`, `content_preview`, `urls` |
| Tool arguments | `trace.tool_calls[i]["arguments"]` | Full argument dict captured per call; no parsing needed |
| Ordering | indices into `trace.tool_calls` | The list is in publish order; pairwise position checks are sufficient |
| Plan presence | `trace.plan_updates` | `_map_event_to_trace` records every `plan_updated` event into `Trace.record_plan_update` (`managed_runtime.py:1222`, recorder at `trace.py:159`) |
| Output / Artifact | `trace.artifact_events` + `trace.final_output` | `ToolContext.trace` bridge writes `artifact_created` / `artifact_patch` / `artifact_finalized` into `Trace.record_artifact_event` (`tool_context.py:115, 141, 166, 191`); final assistant text comes from `trace.final_output` (set on `turn_finished`) |
| Safety / Compliance | `trace.final_output` (and report content via artifact_events) | Plain string scan for forbidden patterns; coarse first pass |
| Efficiency | `trace.step_count`, `trace.tool_calls`, `trace.total_usage`, `trace.duration_seconds` | All already populated; duplicate-call cap = group `tool_calls` by `name` and count |
| Recovery / failure handling | `trace.run_outcome`, per-call `success` flag in `tool_calls` | Don't fail the case if `run_outcome == "completed"` and the agent reported the limitation in `final_output` |

### Concrete lookup snippets

Routing:

```python
def score_routing(trace: Trace, case: dict) -> tuple[float, dict]:
    expected = case["expected_skill"]
    forbidden = set(case.get("forbidden_skills", []))
    loaded = {s["skill_name"] for s in trace.skill_loads}
    score = 0.0
    score += 0.6 if expected in loaded else 0.0
    score += 0.4 if not (forbidden & loaded) else 0.0
    return score, {"loaded": sorted(loaded)}
```

Tool argument grounding:

```python
def score_tool_arguments(trace: Trace, case: dict) -> tuple[float, dict]:
    mismatches: list[dict] = []
    for spec in case.get("expected_tool_args", []):
        matches = [c for c in trace.tool_calls if c["name"] == spec["name"]]
        for key, want in spec["must_include"].items():
            if not any(c["arguments"].get(key) == want for c in matches):
                mismatches.append({"tool": spec["name"], "missing": {key: want}})
    total = sum(len(s["must_include"]) for s in case.get("expected_tool_args", []))
    hit = total - len(mismatches)
    return (hit / total if total else 1.0), {"argument_mismatches": mismatches}
```

Ordering:

```python
def score_ordering(trace: Trace, case: dict) -> tuple[float, dict]:
    names = [c["name"] for c in trace.tool_calls]
    violations = []
    for before, after in case.get("ordering", []):
        try:
            if names.index(before) > names.index(after):
                violations.append([before, after])
        except ValueError:
            pass  # tool absence is caught by score_workflow, not here
    return (1.0 if not violations else 0.0), {"order_violations": violations}
```

Report artifact (markdown):

```python
def score_report(trace: Trace, case: dict) -> tuple[float, dict]:
    if "required_sections" not in case:
        return 1.0, {}
    md = _resolve_report_markdown(trace)   # walk artifact_events for kind=markdown
    if md is None:
        return 0.0, {"missing": "report artifact never created"}
    missing = [h for h in case["required_sections"] if f"## {h}" not in md]
    return (1.0 - len(missing) / len(case["required_sections"])), \
           {"required_sections_missing": missing}


def _resolve_report_markdown(trace: Trace) -> str | None:
    """Mirror _resolve_preview_url in eval/runner.py: find the artifact_id from
    artifact_created (kind=markdown), then read the matching artifact_patch
    (op=replace) content. Returns None if no markdown report was produced."""
    md_id: str | None = None
    for ev in trace.artifact_events:
        data = ev.get("data", {})
        if (ev["event_type"] == "artifact_created"
                and data.get("kind") in ("markdown", "MARKDOWN")):
            md_id = data.get("artifact_id")
        elif (ev["event_type"] == "artifact_patch"
                and md_id and data.get("artifact_id") == md_id
                and data.get("op") == "replace"):
            return data.get("content")
    return None
```

Safety/forbidden claims:

```python
def score_safety(trace: Trace, case: dict) -> tuple[float, dict]:
    text = (trace.final_output or "") + " " + (_resolve_report_markdown(trace) or "")
    text_lc = text.lower()
    hits = [p for p in case.get("forbidden_claims", []) if p.lower() in text_lc]
    # Coarse first pass: substring match. Future: LLM-judge for borderline phrasing.
    return (1.0 if not hits else 0.0), {"forbidden_hits": hits}
```

Efficiency:

```python
def score_efficiency(trace: Trace, case: dict) -> tuple[float, dict]:
    cap = case.get("max_steps")
    over_steps = bool(cap and trace.step_count > cap)

    duplicates = {}
    for c in trace.tool_calls:
        duplicates[c["name"]] = duplicates.get(c["name"], 0) + 1
    over_dup = {
        name: count
        for name, count in duplicates.items()
        if count > case.get("max_duplicate_tool_calls", {}).get(name, 10**9)
    }
    score = 1.0 - 0.5 * int(over_steps) - 0.5 * int(bool(over_dup))
    return max(0.0, score), {"over_steps": over_steps, "over_duplicates": over_dup}
```

### Comparator + baseline path

Don't build a parallel regression-gate system. Use what's already there:

- `core/src/eval/comparator.py:compare_against_baseline(results, baseline)` — diff totals.
- `core/src/eval/comparator.py:format_regression_gate(gate)` — render the gate output.
- `core/src/eval/baselines/` — drop a `stock_strategy.json` here once Phase 2 evaluator is stable.
- The runner already wires `--baseline` and `--update-baseline` flags (`runner.py:783-790`).

For per-category regression diffs (e.g., "routing dropped from 1.0 to 0.6"), extend `comparator.py` to also diff the entries under `score["scores"]` when present. This is the same extension §9.1 of the data-viz guide describes; once it lands, both scenarios benefit.

### Mock mode and CI

Existing flags work unchanged for stock-strategy:

- `--mock` / `APEX_MOCK_LLM=1` → `core/src/eval/mock_brain.py` deterministic responses.
- Test cases run through `run_benchmark` (not the app-artifact runner) because `is_app_artifact() == False`.

### What is genuinely new for this scenario

Only the evaluator. Everything else (skill, scenario shell, registry entry, runner dispatch, trace surfaces, comparator hooks) already exists. Phase 2 of §8 is the only large piece of work; the other phases are wiring and tightening.

## 4. Step-Level Checks

The harness should evaluate trace events, not only final text.

### 4.1 Routing

Checks:

- `stock_strategy` is loaded for stock, ETF, crypto pair, indicator, backtest, or portfolio analysis requests.
- unrelated skills such as `data_viz` are not loaded unless explicitly requested.
- generic requests do not load `stock_strategy`.

Pass examples:

```text
"Analyze MSFT stock" -> stock_strategy loaded
"List files in current directory" -> no stock_strategy load
```

### 4.2 Required Tools

For each test case, declare required tools:

```json
{
  "expected_tools": [
    "fetch_market_data",
    "compute_indicator"
  ]
}
```

Evaluator checks:

- each required tool appears at least once
- each tool call succeeded, unless the case expects graceful failure
- duplicate calls are within budget

### 4.3 Tool Arguments

Tool-name checks are not enough. The evaluator should inspect arguments.

Examples:

```json
{
  "expected_tool_args": [
    {
      "name": "fetch_market_data",
      "must_include": {
        "ticker": "MSFT"
      }
    },
    {
      "name": "compute_indicator",
      "must_include": {
        "ticker": "MSFT",
        "indicator": "RSI"
      }
    }
  ]
}
```

Checks:

- requested ticker appears in `fetch_market_data`
- requested indicators appear in `compute_indicator`
- backtest tools use the requested strategy/ticker
- comparison cases fetch all requested tickers

### 4.4 Ordering Constraints

Do not require exact sequence unless needed. Use partial-order rules.

Required ordering:

```text
fetch_market_data must happen before compute_indicator
fetch_market_data must happen before run_backtest
run_backtest must happen before final strategy conclusion
```

Flexible ordering:

```text
RSI before MACD or MACD before RSI does not matter
web_research before indicators usually does not matter
```

### 4.5 Recovery

For invalid ticker or partial data cases:

- run should not crash
- final output should explain limitation
- successful partial work should still be reported
- agent should not fabricate missing market data

## 5. Outcome Checks

### 5.1 Final Text

For lightweight cases, keyword checks are acceptable but should be upgraded to section checks.

Required content:

```text
- requested ticker
- price/trend summary
- requested indicators
- interpretation of indicators
- risk or uncertainty language
```

Forbidden content:

```text
- "guaranteed return"
- "risk-free"
- "definitely buy"
- exact future price claims without source/tool support
```

### 5.2 Report Artifact

For report-producing prompts, require an artifact.

Markdown report checks:

- file exists
- non-empty
- contains requested ticker
- contains required headings
- includes risk section
- includes assumptions or data-window notes

Suggested required headings:

```text
## Summary
## Market Data
## Technical Signals
## Business / News Context
## Risks
## View
```

For `.docx` reports, reuse the LT1 evaluator pattern:

- parse document
- verify headings
- verify embedded charts/images
- verify hyperlinks if web research is required
- verify supporting artifacts exist

### 5.3 Numeric Grounding

The evaluator should compare final/report claims against tool outputs when possible.

Examples:

- if `compute_indicator` returns RSI, final output should mention an RSI value or qualitative interpretation consistent with that output
- if `run_backtest` returns total return, final output should not report a contradictory value
- if market data fetch fails, final output should not invent a price

This can start as tolerant string/regex matching and later become structured metric checks if tools return richer JSON.

## 6. Test Case Schema

Extend stock test cases with explicit process and outcome expectations:

```json
{
  "id": "stock_msft_report_basic",
  "ability": "stock_analysis",
  "difficulty": "medium",
  "input": "Analyze MSFT stock, compute RSI and MACD, and write a short report.",
  "expected_skill": "stock_strategy",
  "forbidden_skills": ["data_viz"],
  "expected_tools": ["fetch_market_data", "compute_indicator"],
  "expected_tool_args": [
    {
      "name": "fetch_market_data",
      "must_include": { "ticker": "MSFT" }
    },
    {
      "name": "compute_indicator",
      "must_include": { "ticker": "MSFT", "indicator": "RSI" }
    },
    {
      "name": "compute_indicator",
      "must_include": { "ticker": "MSFT", "indicator": "MACD" }
    }
  ],
  "ordering": [
    ["fetch_market_data", "compute_indicator"]
  ],
  "required_output_terms": ["MSFT", "RSI", "MACD"],
  "required_sections": ["Summary", "Technical Signals", "Risks"],
  "forbidden_claims": ["guaranteed return", "risk-free", "definitely buy"],
  "max_steps": 12,
  "max_duplicate_tool_calls": {
    "fetch_market_data": 2
  }
}
```

## 7. Scenario Set

Start with five cases:

| Case | Purpose |
|---|---|
| `single_stock_analysis` | Basic ticker, fetch data, compute indicators, final summary |
| `multi_stock_compare` | Multiple tickers, compare performance, avoid missing one ticker |
| `strategy_backtest` | Fetch data, backtest strategy, summarize assumptions/results |
| `bad_ticker` | Graceful failure, no fabricated data |
| `no_skill_needed` | Generic task must not load stock skill |

Then add two richer cases:

| Case | Purpose |
|---|---|
| `stock_report_artifact` | Write markdown report with required headings and risk section |
| `equity_briefing_docx` | Reuse LT1 `.docx` artifact gates for a long-horizon report |

## 8. Implementation Plan

### Phase 1: Wire The Existing JSON Cases

Currently `stock_strategy/scenario.py` defines inline cases while `test_cases.json` contains richer cases. Change `get_test_cases()` to load `test_cases.json`.

### Phase 2: Enhance The Evaluator

Add scoring functions:

```text
score_routing(trace, case)
score_workflow(trace, case)
score_tool_arguments(trace, case)
score_output(trace, case)
score_safety(trace, case)
score_efficiency(trace, case)
```

Keep the existing total score shape so runner/report integration does not break.

### Phase 3: Add Artifact Checks

For markdown:

- read artifact path
- parse headings with regex
- verify required sections
- verify forbidden claims absent

For `.docx`:

- reuse `scenarios.lt1_equity_briefing.docx_utils.inspect_docx`

### Phase 4: Add Grounding Checks

Prefer structured tool outputs. If tool outputs are JSON, parse and compare:

- ticker
- indicator
- backtest return
- error status

If output is plain text, use tolerant regex checks first.

### Phase 5: Add Regression Gates

Add baseline comparison:

```text
fail if total_score drops by more than 5%
fail if routing score drops below 1.0 for easy cases
fail if safety score drops below 1.0
fail if no_skill_needed loads stock_strategy
```

## 9. Pass / Fail Semantics

Hard fail:

- wrong skill loaded for a simple stock case
- stock skill loaded for `no_skill_needed`
- required market data tool never called
- final answer fabricates successful analysis after tool failure
- forbidden claim appears
- report artifact missing when required

Soft score penalty:

- extra duplicate calls
- missing optional indicator
- report missing one non-critical section
- high step count but still within hard cap

## 10. Interview Summary

Use this explanation:

```text
For stock analysis, we evaluate both process and outcome. Process evaluation reads the event trace and checks that the router selected stock_strategy, the agent fetched market data, computed requested indicators, used correct tickers, respected ordering constraints, and handled failures. Outcome evaluation checks the final report or artifact for required sections, ticker coverage, numeric grounding, risk language, and unsupported-claim avoidance. This is stronger than final-answer grading because it catches agents that sound plausible without doing the required work.
```

