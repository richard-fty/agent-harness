# Data Viz Story Agent Implementation Guide

## 1. Goal

Build a data visualization story flow on top of the existing Apex coding agent.

The target user is not a CLI user. They upload or provide data, describe the question they care about, and receive a good-looking interactive data story in the artifact side panel:

```text
User data + question
  -> data understanding
  -> metric derivation
  -> visual story structure
  -> interactive report app
  -> app_preview artifact
  -> deterministic eval score
```

The product should feel closer to a polished analyst-made story report than a generic dashboard. It should have a clear narrative, trustworthy calculations, and enough interaction for a non-technical user to explore the data.

## 2. Product Shape

The first screen of the generated artifact should be the report itself, not a landing page or explanation page.

Recommended story flow:

1. **Headline insight**: one sentence that answers the user's main question.
2. **KPI strip**: 3-5 key numbers with units, deltas, and short labels.
3. **Main chart**: the primary trend, comparison, or relationship.
4. **Breakdown section**: secondary charts that explain what drives the headline.
5. **Interactive controls**: filters for time, segment, category, or region.
6. **Data notes**: concise caveats, missing-data notes, and calculation definitions.
7. **Table/details drawer**: inspectable rows or grouped values for trust.

The output should be visually restrained and data-first:

- clean chart grid, readable labels, visible units
- strong hierarchy between headline, KPIs, charts, and details
- responsive layout with no horizontal overflow
- chart colors chosen for comparison, not decoration
- no decorative hero section, generic marketing copy, or fake product framing

## 3. Skill Strategy

Create a new `data_viz` skill rather than expanding the current `coding` skill.

The current `coding` skill should remain the generic execution layer:

- inspect files
- plan edits
- modify app code
- build and test
- start preview
- emit patch and preview artifacts

The new `data_viz` skill should add domain rules:

- inspect the dataset/schema before designing charts
- derive metrics from source data, not hardcoded final values
- choose chart types based on the question and data shape
- label axes, legends, units, and filters clearly
- make written insights traceable to computed values
- include verification steps before claiming completion
- produce an `app_preview` artifact

Recommended files:

```text
core/src/skill_packs/data_viz/
  SKILL.md
  REFERENCE.md
  __init__.py
  skill.py
  tools.py
```

Register the skill in `core/src/skill_packs/registry.py` by adding a block to
`discover_skills()` mirroring the existing `CodingSkill` import.

Initial `data_viz` tools can be empty or thin wrappers. The first version can rely on the existing core file/shell tools and coding skill tools. New tools should follow the `ToolDef` pattern already used in `core/src/skill_packs/coding/tools.py` so the runtime, planner, and artifact emission paths work without modification. Add specialized tools only when they improve eval reliability, for example:

- `profile_dataset(path)` for schema, column types, missing values, and sample rows
- `compute_metric(data_path, expression)` for deterministic metric checks
- `emit_data_story_spec(spec_json)` for a structured intermediate artifact

## 4. Scenario Strategy

Start by adding data-viz cases to the existing coding harness only if speed is the priority. For the durable version, create a first-class `data_viz` scenario:

```text
core/src/scenarios/data_viz/
  scenario.py
  evaluator.py
  cases/
    case_001.json
    case_001/
      public/
        dataset.csv
        expected_schema.json
      hidden/
        dashboard.spec.mjs
        expected_metrics.json
      golden/
        patch.diff
  templates/
    vite-data-story/
      index.html
      package.json
      playwright.config.mjs
      src/
        main.js
        styles.css
        data.csv
```

`DataVizScenario` subclasses `Scenario` in `core/src/scenarios/base.py` and is registered in `core/src/scenarios/registry.py` next to `CodingScenario`. It reuses the existing `Trace` shape (`steps`, `tool_calls`, `artifacts`, `gate_results`, `total_usage`) plus the new typed event surfaces added in §9.0.

The important design change from the current coding scenario is hidden tests:

- copy `cases/<id>/public/**` into the workspace before the agent runs
- keep `cases/<id>/hidden/**` outside the workspace during agent execution
- copy `hidden/**` into the workspace immediately before the `test` gate, and remove (or unmount) it after
- apply golden patches only in replay mode (`--replay <case_id>` on the runner CLI)

This prevents the agent from reading the exact evaluator while still allowing deterministic scoring. The hidden split requires the parameterized workspace setup added in §9.0, not the current `_prepare_coding_workspace`, which hardcodes `scenarios/coding`.

## 5. Case Format

Example case:

```json
{
  "id": "sales_story_001",
  "title": "Tell the sales growth story",
  "tier": "DV1",
  "template": "vite-data-story",
  "input": "Build an interactive data story showing what drove 2025 revenue growth. Include monthly revenue, revenue by region, top product categories, and a year/region filter.",
  "public_data": ["cases/case_001/public/sales.csv"],
  "hidden_playwright_spec": "cases/case_001/hidden/dashboard.spec.mjs",
  "expected_metrics": "cases/case_001/hidden/expected_metrics.json",
  "golden_patch": "cases/case_001/golden/patch.diff",
  "allowed_paths": ["src/**", "package.json", "index.html"],
  "required_plan_steps": [
    "inspect dataset",
    "derive metrics",
    "build story layout",
    "add filters",
    "verify calculations",
    "start preview"
  ],
  "manifest": {
    "install": ["pnpm", "install", "--frozen-lockfile", "--prefer-offline"],
    "build": ["pnpm", "build"],
    "test": ["pnpm", "exec", "playwright", "test", "cases/case_001/hidden/dashboard.spec.mjs", "--reporter=json"],
    "budget": { "max_steps": 40, "max_cost_usd": 1.0, "max_wall_sec": 420 }
  }
}
```

## 6. Evaluation Layers

Evaluate each run as a stack. The whole score should explain whether the agent failed because it ignored process, broke the app, miscomputed the data, or produced a weak story.

All layer signals come from data the runner produces, but several of these surfaces do not exist on `Trace` today and are added by §9.0. `DataVizScenario.evaluate(trace, case)` reads:

- `trace.gate_results` — already populated by the gate runner. Source of install/build/test pass/fail.
- `trace.plan_updates`, `trace.skill_loads`, `trace.artifact_events` — *new* fields added in §9.0 (`Trace` extension). `plan_updates` and `skill_loads` are populated by `_map_event_to_trace`; `artifact_events` are populated by `ToolContext.trace` inside the artifact helpers. Source of `plan_updated`, `skill_auto_loaded`, `artifact_created` (filter by `kind == "app_preview"`), `artifact_patch`, and `artifact_finalized`.
- `trace.artifacts` — *expanded* by §9.0 to include install/build/test logs, the Playwright JSON report, screenshots, and the preview URL alongside the existing patch diff. Per-assertion pass/fail comes from parsing the JSON report.
- `trace.tool_calls` — already populated. Useful as a fallback signal for plan-step coverage if a managed event was missed.
- `case` JSON for `expected_metrics` and `required_plan_steps`.

For this trace to be the *agent's* trace (not a fresh local one), `run_single` must return it; see §9.0.

`evaluate()` returns the layered shape in §9.3 so the comparator can diff per-layer drops.

### 6.0 Replay-mode scoring

Replay mode (`--replay <case_id>`) only applies a golden patch. It does **not** run the agent, load skills, update plans, or call `start_app_preview`, so `trace.plan_updates`, `trace.skill_loads`, and `trace.artifact_events` are empty by construction. Treating those as failures would make replay impossible to pass.

`DataVizScenario.evaluate` must therefore branch on `trace.run_outcome` (or a `replay_mode` flag passed via the case dict / trace metadata):

- **Replay mode**: skip §6.1 (skill/planner) and §6.2's artifact-event sub-signals. Score only on what the patch actually produces — install, build, test pass/fail, data correctness, viz, and story layers. Renormalize weights or assert that the omitted layers contribute 0 to a `replay_total_score` separate from the agent-mode total.
- **Agent mode**: full layered scoring as described in §6.1–§6.5.

Replay's purpose is to prove the *evaluator and gates* work, not to score agent behavior. The Phase 1 exit criterion is "replay reaches the maximum achievable score under replay-mode scoring" (everything except skill/planner and artifact-events at 1.0), not "all five layers at 1.0."

Recommended weights:

| Layer | Weight | What It Proves |
|---|---:|---|
| Skill and planner compliance | 10% | The agent used the right workflow |
| Artifact and runtime health | 15% | The app builds, runs, and appears in the side panel |
| Data correctness | 40% | Metrics and transformations are right |
| Visualization and interaction | 20% | Charts and filters answer the question |
| Story, layout, and accessibility | 15% | The output is understandable and polished |

### 6.1 Skill and Planner Compliance

Signals:

- `data_viz` skill was loaded or auto-loaded for the task
- `plan_updated` exists
- the plan includes dataset inspection, metric derivation, layout, interaction, verification, and preview
- statuses progress beyond all `pending`
- final answer does not claim completion before build/test/preview

Suggested scoring:

```text
0.30 skill loaded
0.25 plan exists
0.25 required plan concepts present
0.20 statuses progress coherently
```

### 6.2 Artifact and Runtime Health

Signals:

- install gate passes
- build gate passes
- Playwright can open the app
- no uncaught console errors
- `artifact_created.kind == "app_preview"` appears
- `artifact_finalized` appears for the preview artifact
- preview content is a localhost URL

Suggested scoring:

```text
0.25 install
0.25 build
0.20 app loads
0.15 preview artifact created/finalized
0.15 no console/runtime errors
```

### 6.3 Data Correctness

This should be the heaviest component.

Use fixture-derived expected metrics, not screenshots, as the source of truth:

- totals
- averages
- rates
- deltas
- ranks
- group-by values
- time buckets
- filter-dependent recomputation

Playwright should read visible text or exposed DOM attributes and compare them with expected values.

Recommended implementation pattern:

```html
<section data-testid="kpi-total-revenue" data-value="1842500">
  $1.84M revenue
</section>

<svg data-testid="monthly-revenue-chart">
  <g data-series="monthly-revenue" data-points="12"></g>
</svg>
```

The UI text can stay beautiful, while `data-*` attributes make deterministic checks simple.

Suggested scoring:

```text
0.30 headline KPI values
0.25 chart/group values
0.20 filter recomputation
0.15 ranks/top-bottom claims
0.10 missing-data handling
```

### 6.4 Visualization and Interaction

Checks:

- main chart type matches the question
- trend uses line or bar by time bucket
- categorical comparison uses bars, not pie by default
- x/y axes have labels or accessible names
- legends are present when multiple series exist
- filters update KPIs and charts
- hover/focus or detail table lets the user inspect values
- desktop and mobile layouts remain usable

This can be mostly deterministic:

- assert chart containers and labels exist
- assert number of marks/buckets
- assert filter changes visible values
- assert no horizontal overflow
- assert color scale uses enough contrast

Keep subjective visual judging optional for MVP.

### 6.5 Story, Layout, and Accessibility

Checks:

- headline insight matches the strongest data finding
- supporting charts explain the headline
- sections are ordered as a narrative
- no fake or unsupported insight appears
- all controls have accessible names
- chart summaries exist for screen readers
- mobile screenshot has no overlapping text
- visual hierarchy is clear: headline, KPIs, main chart, breakdown, notes

For MVP, use a rubric plus deterministic layout checks. Later, add screenshot and LLM-judge review against the rubric.

## 7. Data Story Template

The `vite-data-story` template should include a small, dependency-light foundation:

- plain Vite-compatible HTML/JS/CSS
- optional lightweight chart helper built with SVG
- data loading helper for CSV or JSON
- utility functions for grouping, formatting, sorting, and filtering
- CSS layout tokens for report width, chart grid, KPI strip, controls, and mobile behavior

Avoid heavy chart libraries in the first eval template unless pinned dependencies are already available offline. A simple SVG chart layer is enough for deterministic tests:

- bar chart
- line chart
- horizontal ranking bars
- stacked bars if needed later

If a library is added later, prefer one stable, pinned dependency and make the eval image contain it.

## 8. UI Requirements For Generated Reports

Every generated data story should include:

- `<main>` landmark
- clear report title
- headline insight section
- KPI strip with 3-5 cards
- primary chart with title, axes/legend, and accessible summary
- at least one breakdown chart
- filters when the prompt requests comparison or exploration
- details table or grouped values section
- data notes with calculation definitions

Every numeric claim should be either:

- directly visible as a computed metric
- reflected in a chart mark
- included in the notes as a calculation

The agent should avoid:

- hardcoded values that are not derived from data
- placeholder chart marks
- unexplained percentages
- generic "insights" that do not cite a metric
- decorative landing-page composition
- one-color dashboards with weak hierarchy

## 9. Runner Changes

The current harness in `core/src/eval/runner.py` is **not yet scenario-agnostic** and the agent trace it passes to `evaluate()` is **not the trace the agent actually produced**. The data-viz scenario cannot work end-to-end until those gaps are closed. §9.0 lists the harness prerequisites with concrete code snippets for the functions/branches that do not exist yet. §9.1 lists the additive edits. §9.2 lists the new files. §9.3 gives the result row shape.

### 9.0 Harness prerequisites (must land before `data_viz`)

Five concrete pieces must exist in the harness before `DataVizScenario` can be implemented. None exist today; all snippets below are *new code*.

#### 9.0.1 Extend `Trace` with typed managed-event surfaces

The current `Trace` (`core/src/agent/runtime/trace.py`) has `steps`, `tool_calls`, `artifacts`, `gate_results`, plus four ledgers, but no fields for the managed events the data-viz evaluator needs. Add them:

```python
# core/src/agent/runtime/trace.py — additions to class Trace

class Trace(BaseModel):
    # ... existing fields ...
    plan_updates: list[dict[str, Any]] = Field(default_factory=list)
    skill_loads: list[dict[str, Any]] = Field(default_factory=list)
    artifact_events: list[dict[str, Any]] = Field(default_factory=list)

    def record_plan_update(self, *, step: int, kind: str, payload: dict[str, Any]) -> None:
        self.plan_updates.append({
            "step": step,
            "kind": kind,
            "payload": payload,
            "timestamp": time.time(),
        })

    def record_skill_load(self, *, step: int, skill_name: str) -> None:
        self.skill_loads.append({
            "step": step,
            "skill_name": skill_name,
            "timestamp": time.time(),
        })

    def record_artifact_event(self, *, step: int, event_type: str, data: dict[str, Any]) -> None:
        # event_type in {"artifact_created", "artifact_patch", "artifact_finalized"}
        self.artifact_events.append({
            "step": step,
            "event_type": event_type,
            "data": data,
            "timestamp": time.time(),
        })
```

#### 9.0.2 Route managed and artifact events into the trace

There are **two separate event channels** in the runtime, and the harness must bridge both into `Trace`:

- **`ManagedEvent` channel** — the runtime's own event generator. Emits `skill_auto_loaded` (`managed_runtime.py:270`, `:750`) and `plan_updated` (`:681`), among many others. `_map_event_to_trace` (line ~1102) already reads this channel.
- **`event_bus` channel** — used by `tool_context.emit_artifact_created`, `emit_artifact_replace`, and `emit_artifact_finalized` (`core/src/agent/runtime/tool_context.py:78`). These are awaited inside tool functions, publish to `ctx.event_bus`, and **never appear as `ManagedEvent`s**. So a naive `elif event.type == "artifact_created"` branch in `_map_event_to_trace` would never fire.

**Fix A — add `ManagedEvent` branches for what the runtime already yields:**

```python
# core/src/agent/runtime/managed_runtime.py — additions inside _map_event_to_trace

elif event.type == "plan_updated":
    trace.record_plan_update(
        step=self.session.step,
        kind=event.data.get("kind", "task"),
        payload=event.data,
    )
elif event.type == "skill_auto_loaded":
    trace.record_skill_load(
        step=self.session.step,
        skill_name=event.data.get("skill_name", ""),
    )
```

**Fix B — bridge artifact events from `tool_context` into the trace.** Extend `ToolContext` to optionally carry a `Trace`, and have the artifact helpers append to it directly:

```python
# core/src/agent/runtime/tool_context.py — additions

class ToolContext:
    # ... existing fields ...
    trace: "Trace | None" = None  # set by ManagedAgentRuntime when running under harness


async def emit_artifact_created(*, spec: Any) -> str | None:
    from agent.events.schema import ArtifactCreated
    ctx = get_tool_context()
    if ctx is None or ctx.artifact_store is None:
        return None
    artifact = await ctx.artifact_store.create(ctx.session_id, spec)
    payload = {
        "artifact_id": artifact.id,
        "kind": getattr(spec.kind, "value", str(spec.kind)),
        "name": spec.name,
        "language": getattr(spec, "language", None),
        "mime": getattr(spec, "mime", None),
        "description": getattr(spec, "description", None),
    }
    await ctx.event_bus.publish(ctx.session_id, ArtifactCreated(
        session_id=ctx.session_id, turn_id=ctx.turn_id, **payload,
    ))
    if ctx.trace is not None:
        ctx.trace.record_artifact_event(
            step=getattr(ctx, "current_step", 0),
            event_type="artifact_created",
            data=payload,
        )
    return artifact.id


async def emit_artifact_replace(artifact_id: str, content: str) -> None:
    # ... existing replace + ArtifactPatch publish call ...
    ctx = get_tool_context()
    if ctx is not None and ctx.trace is not None and artifact_id:
        ctx.trace.record_artifact_event(
            step=getattr(ctx, "current_step", 0),
            event_type="artifact_patch",
            data={
                "artifact_id": artifact_id,
                "op": "replace",
                "content": content,
            },
        )


async def emit_artifact_finalized(artifact_id: str) -> None:
    # ... existing publish call ...
    ctx = get_tool_context()
    if ctx is not None and ctx.trace is not None and artifact_id:
        ctx.trace.record_artifact_event(
            step=getattr(ctx, "current_step", 0),
            event_type="artifact_finalized",
            data={"artifact_id": artifact_id},
        )
```

`ManagedAgentRuntime` sets `ctx.trace = trace` and updates `ctx.current_step = self.session.step` whenever it enters tool dispatch. Existing non-harness callers (server, tests) leave `trace = None` and pay no overhead.

This is the cleanest option given the current architecture: artifact helpers can't `yield ManagedEvent` because they are not generators and are called from deep inside tool functions. The alternative — having the runtime subscribe to its own `event_bus` and reconvert events into `ManagedEvent`s — adds a feedback loop and double-publishes to TUI consumers.

#### 9.0.3 `run_single` must return the trace and skip premature evaluation

Today `run_single` (`core/src/eval/runner.py:210`) returns only the score dict and saves the trace internally. The coding path constructs a *fresh* local `Trace` and never receives the agent's. Two problems must be fixed at once:

1. The agent's real trace must reach the caller.
2. For app-artifact scenarios, `scenario.evaluate(...)` must run **after** install/build/test gates and artifact capture — not inside `run_single`. Calling `evaluate()` twice (once with empty `gate_results`, once with full ones) silently masks bugs.

```python
# core/src/eval/runner.py — replace the current signature and body

async def run_single(
    scenario: Any,
    model: str,
    test_case: dict[str, Any],
    strategy: str,
    timeout: int = 300,
    use_mock: bool = False,
    evaluate: bool = True,
) -> tuple[Trace, dict[str, Any] | None]:
    """Run a single agent execution.

    `evaluate=True` (default): caller is the simple benchmark path; trace contains
    everything needed to score, so we evaluate inline.

    `evaluate=False`: caller is `run_app_artifact_benchmark`, which will run gates
    and capture artifacts before scoring. Returning None for the score signals
    that the trace is not yet complete.
    """
    # ... existing setup, LT1/mock branches return (trace, score|None) the same way ...
    trace = await run_agent(
        user_input=test_case["input"],
        model=model,
        context_strategy=strategy,
        runtime_config=runtime,
    )
    if not evaluate:
        return trace, None
    score = scenario.evaluate(trace, test_case)
    score["model"] = model
    score["context_strategy"] = strategy
    score["scenario"] = scenario.name
    return trace, score
```

`_run_lt1` and `_run_mock` must return the same `(trace, score | None)` tuple and honor `evaluate`. Existing callers of `run_single` (the generic `run_benchmark` path) unpack the tuple; the app-artifact runner passes `evaluate=False` and re-evaluates after gates.

#### 9.0.4 Generalize the coding runner into an app-artifact runner

`run_coding_benchmark`, `_prepare_coding_workspace`, `_case_path`, `_run_coding_gates`, and `_capture_coding_artifacts` all hardcode `scenarios/coding`. Replace them with parameterized versions that take a scenario directory:

```python
# core/src/eval/runner.py — new generalized runner (replaces run_coding_benchmark)

async def run_app_artifact_benchmark(
    scenario: Any,
    test_cases: list[dict[str, Any]],
    *,
    model: str,
    strategy: str,
    timeout: int,
    replay: str | None,
    use_mock: bool,
    output_dir: str,
) -> list[dict[str, Any]]:
    """Generic gated-app benchmark for any scenario whose cases declare
    install/build/test gates. Used by `coding`, `data_viz`, and any future
    scenario that produces an app_preview artifact."""
    scenario_dir = Path(__file__).resolve().parents[1] / "scenarios" / scenario.name
    out_root = Path(output_dir).resolve()
    sandbox_root = out_root / "sandbox"
    trace_root = out_root / "traces"
    sandbox_root.mkdir(parents=True, exist_ok=True)
    trace_root.mkdir(parents=True, exist_ok=True)

    selected = test_cases
    if replay and replay != "all":
        selected = [c for c in test_cases if c.get("id") == replay]
    if replay and not selected:
        raise ValueError(f"Replay case not found: {replay}")

    results: list[dict[str, Any]] = []
    for case in selected:
        case_id = case["id"]
        start = time.time()
        workspace = _prepare_app_artifact_workspace(case, scenario_dir, sandbox_root)
        try:
            if replay or use_mock:
                trace = Trace(
                    run_id=f"{scenario.name}_{case_id}_{int(start)}",
                    model=model, scenario=scenario.name,
                    prompt=case["input"], context_strategy=strategy,
                )
                _apply_patch_file(workspace, _case_path(scenario_dir, case["golden_patch"]))
            else:
                old_cwd = Path.cwd()
                try:
                    os.chdir(workspace)
                    # evaluate=False: gates and artifact capture happen below;
                    # we evaluate exactly once after the trace is fully populated.
                    trace, _ = await run_single(
                        scenario, model, case, strategy,
                        timeout=timeout, use_mock=False, evaluate=False,
                    )
                finally:
                    os.chdir(old_cwd)

            trace.gate_results = await _run_app_artifact_gates(
                case, scenario_dir, workspace, timeout=timeout,
            )
            trace.artifacts.extend(
                _capture_app_artifact_outputs(case, scenario_dir, workspace, out_root)
            )
            trace.finish(output=f"{scenario.name} gates completed")
            score = scenario.evaluate(trace, case)
            score.update({
                "model": model, "context_strategy": strategy,
                "scenario": scenario.name, "mock_mode": use_mock,
                "replay_mode": bool(replay),
                "duration_seconds": time.time() - start,
            })
            results.append(score)
        except Exception as exc:
            results.append({
                "test_case_id": case_id, "model": model,
                "scenario": scenario.name, "total_score": 0.0,
                "error": str(exc),
                "duration_seconds": time.time() - start,
            })
        finally:
            trace.save(str(trace_root))
    return results


def _prepare_app_artifact_workspace(
    case: dict[str, Any], scenario_dir: Path, sandbox_root: Path
) -> Path:
    template = scenario_dir / "templates" / case["template"]
    workspace = sandbox_root / case["id"]
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(template, workspace)

    # Copy public fixtures into the workspace so the agent can read them
    case_root = _case_path(scenario_dir, case["golden_patch"]).parents[1]
    public_src = case_root / "public"
    if public_src.exists():
        shutil.copytree(public_src, workspace, dirs_exist_ok=True)

    # Copy the case manifest dir under cases/ for backward compat with coding cases
    cases_dst = workspace / "cases"
    cases_dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(case_root, cases_dst / case_root.name, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("hidden"))

    _git(workspace, "init")
    _git(workspace, "config", "user.email", "apex@example.invalid")
    _git(workspace, "config", "user.name", "Apex Eval")
    _git(workspace, "add", ".")
    _git(workspace, "commit", "-m", "baseline")
    return workspace


def _case_path(scenario_dir: Path, rel: str) -> Path:
    return scenario_dir / rel
```

#### 9.0.5 Hidden-asset injection at the test gate

Mount hidden fixtures only for the `test` stage and remove them after, so the agent's earlier steps never see them.

**Path contract:** the manifest references hidden assets by their *post-injection* path, e.g. `cases/case_001/hidden/dashboard.spec.mjs` and `cases/case_001/hidden/expected_metrics.json` (see §5). The injection target must mirror that layout — `<workspace>/cases/<case_dir>/hidden/` — not workspace root. The earlier `_prepare_app_artifact_workspace` step copies the case directory under `<workspace>/cases/<case_dir>/` with `ignore_patterns("hidden")`, so the parent path already exists at gate time.

```python
# core/src/eval/runner.py — replaces _run_coding_gates

async def _run_app_artifact_gates(
    case: dict[str, Any],
    scenario_dir: Path,
    workspace: Path,
    *,
    timeout: int,
) -> dict[str, bool]:
    manifest = case["manifest"]
    sandbox = _make_gate_sandbox(workspace)  # existing helper, runner.py:654
    log_dir = workspace / ".gate_logs"
    log_dir.mkdir(exist_ok=True)
    results: dict[str, bool] = {}

    case_root = _case_path(scenario_dir, case["golden_patch"]).parents[1]
    hidden_src = case_root / "hidden"
    hidden_dst = workspace / "cases" / case_root.name / "hidden"

    for stage in ("install", "build", "test"):
        cmd_parts = manifest.get(stage)
        if not cmd_parts:
            results[stage] = True
            continue

        injected = False
        if stage == "test" and hidden_src.exists():
            shutil.copytree(hidden_src, hidden_dst, dirs_exist_ok=True)
            injected = True
        try:
            cmd = _stage_command(cmd_parts, case)  # existing helper, runner.py:686
            network = "bridge" if stage == "install" else "none"
            result = await sandbox.run_oneshot(cmd, timeout=timeout, network=network)

            # Persist stdout and stderr separately so §9.0.6 can extract the
            # Playwright JSON payload from test.stdout. The existing runner
            # writes a single combined <stage>.log; we replace that here.
            (log_dir / f"{stage}.stdout").write_text(
                result.stdout or "", encoding="utf-8",
            )
            (log_dir / f"{stage}.stderr").write_text(
                result.stderr or "", encoding="utf-8",
            )
            results[stage] = result.exit_code == 0 and not result.timed_out
            if not results[stage]:
                # Match existing behavior: stop the chain on first failure.
                break
        finally:
            if injected and hidden_dst.exists():
                shutil.rmtree(hidden_dst, ignore_errors=True)
    return results
```

This replaces both the gate loop and the log persistence in one body — there is no separate `_run_gate_stage` helper. `sandbox.run_oneshot` and `_make_gate_sandbox` already exist in `runner.py`.

The Phase 0 audit (§11) must verify `<workspace>/cases/<case_dir>/hidden/` does not exist before or after the test gate, only during it.

#### 9.0.6 Capture logs, Playwright JSON, screenshots, preview URL

Today `_capture_coding_artifacts` (`runner.py:690`) writes only the patch diff. Replace with a richer capture:

```python
# core/src/eval/runner.py — replaces _capture_coding_artifacts

def _capture_app_artifact_outputs(
    case: dict[str, Any],
    scenario_dir: Path,
    workspace: Path,
    out_root: Path,
) -> list[dict[str, Any]]:
    case_id = case["id"]
    out_dir = out_root / "artifacts" / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    captured: list[dict[str, Any]] = []

    # 1. Patch diff
    patch = subprocess.run(
        ["git", "diff"], cwd=workspace, capture_output=True, text=True,
    ).stdout
    patch_path = out_dir / "patch.diff"
    patch_path.write_text(patch, encoding="utf-8")
    captured.append({"kind": "code", "language": "diff", "path": str(patch_path)})

    # 2. Stage logs (written by _run_app_artifact_gates; see §9.0.5)
    for stage in ("install", "build", "test"):
        for stream in ("stdout", "stderr"):
            log = workspace / f".gate_logs/{stage}.{stream}"
            if log.exists():
                dst = out_dir / f"{stage}.{stream}.log"
                shutil.copy2(log, dst)
                captured.append({"kind": "log", "stage": stage, "stream": stream,
                                 "path": str(dst)})

    # 3. Playwright JSON report.
    # Bare `--reporter=json` writes to stdout, NOT to playwright-report/report.json
    # (that path is only populated by the html reporter). The reliable source is
    # the captured test stdout from §9.0.7. We persist it as a separate file.
    test_stdout = workspace / ".gate_logs" / "test.stdout"
    if test_stdout.exists():
        text = test_stdout.read_text(encoding="utf-8", errors="replace")
        # Playwright JSON starts with `{` and contains "config" + "suites" keys.
        # Tolerate noise lines emitted by the wrapper before/after the payload.
        json_blob = _extract_playwright_json(text)
        if json_blob is not None:
            dst = out_dir / "playwright.json"
            dst.write_text(json_blob, encoding="utf-8")
            captured.append({"kind": "test_report", "format": "playwright_json",
                             "path": str(dst)})

    # 4. Screenshots
    screenshots = workspace / "test-results"
    if screenshots.exists():
        for png in screenshots.rglob("*.png"):
            dst = out_dir / "screenshots" / png.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(png, dst)
            captured.append({"kind": "screenshot", "path": str(dst)})

    # 5. Preview URL — derived from trace.artifact_events (§9.0.2), NOT a marker file.
    # Captured by the runner via _resolve_preview_url() below; this function is
    # intentionally trace-free.

    return captured


def _resolve_preview_url(trace: "Trace") -> str | None:
    """Walk trace.artifact_events to recover the preview URL.

    `start_app_preview` emits:
      1. `artifact_created` with kind="app_preview"  -> we capture the artifact_id
      2. `artifact_patch` op="replace"               -> content is the URL string
      3. `artifact_finalized`                        -> only carries final metadata

    The URL is in the patch payload (op=replace), not artifact_finalized.
    """
    preview_id: str | None = None
    for ev in trace.artifact_events:
        data = ev.get("data", {})
        if (ev["event_type"] == "artifact_created"
                and data.get("kind") == "app_preview"):
            preview_id = data.get("artifact_id")
        elif (ev["event_type"] == "artifact_patch"
                and preview_id is not None
                and data.get("artifact_id") == preview_id
                and data.get("op") == "replace"):
            return data.get("content")
    return None
```

The runner calls `_resolve_preview_url(trace)` after `_capture_app_artifact_outputs` and appends the result to `trace.artifacts` if non-null:

```python
# core/src/eval/runner.py — inside run_app_artifact_benchmark, after capture

trace.artifacts.extend(
    _capture_app_artifact_outputs(case, scenario_dir, workspace, out_root)
)
preview_url = _resolve_preview_url(trace)
if preview_url:
    trace.artifacts.append({"kind": "app_preview", "url": preview_url})
```

Replay mode skips this — the patch never calls `start_app_preview`, so `trace.artifact_events` is empty and `_resolve_preview_url` returns `None`.

```python
def _extract_playwright_json(text: str) -> str | None:
    """Pull the JSON payload out of Playwright's stdout.

    With --reporter=json, Playwright writes a single JSON object to stdout. The
    sandbox/test wrapper may add lines before/after. Find the first { and the
    matching closing brace.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                blob = text[start:i + 1]
                try:
                    json.loads(blob)
                    return blob
                except Exception:
                    return None
    return None
```

##### 9.0.7 Gate stdout/stderr persistence

This is no longer a separate helper — it is folded into `_run_app_artifact_gates` above (§9.0.5). The relevant change versus today's `_run_coding_gates` (`runner.py:631`):

- old: writes a single combined `<workspace>/<stage>.log` with `\n[stderr]\n` separating streams
- new: writes `<workspace>/.gate_logs/<stage>.stdout` and `<workspace>/.gate_logs/<stage>.stderr` as separate files

The split is required so §9.0.6's `_extract_playwright_json` can read pure stdout (Playwright's `--reporter=json` output) without having to strip a stderr suffix.

#### 9.0.8 CLI dispatch by scenario shape

Replace the hardcoded `if args.scenario == "coding":` branch (`runner.py:818`) with a check on whether the scenario declares the gated-app shape:

```python
# core/src/eval/runner.py — main()

if scenario.is_app_artifact():  # new method on Scenario, default False
    if len(models) > 1 or len(strategies) > 1:
        console.print("[yellow]App-artifact runner uses the first model and strategy only.[/yellow]")
    results = await run_app_artifact_benchmark(
        scenario, test_cases,
        model=models[0], strategy=strategies[0],
        timeout=args.timeout, replay=args.replay,
        use_mock=use_mock, output_dir=args.output,
    )
    ...
    return
```

`Scenario.is_app_artifact(self) -> bool` defaults to `False` in `scenarios/base.py`; `CodingScenario` and `DataVizScenario` override it to return `True`.

#### 9.0.9 Replay flag naming

The runner CLI flag is `--replay`, not `--apply-golden`. Use `--replay <case_id>` (or `--replay` alone for all cases) wherever this guide mentions replaying a golden patch.

### 9.1 Edits to existing files (after §9.0 lands)

- **`core/src/eval/runner.py`** — already covered by §9.0 (signature, generalized runner, gate, capture, CLI dispatch).
- **`core/src/eval/comparator.py`** — extend the row diff to also diff per-layer scores (`skill_planner_score`, `artifact_runtime_score`, `data_correctness_score`, `viz_interaction_score`, `story_layout_accessibility_score`) when present, so `format_regression_gate` names which layer regressed. Treat missing layer fields as backwards-compatible (skip the diff).
- **`core/src/scenarios/base.py`** — add `is_app_artifact(self) -> bool` returning `False`.
- **`core/src/scenarios/coding/scenario.py`** — override `is_app_artifact` to return `True` (so the existing coding scenario also routes through the generalized runner).
- **`core/src/scenarios/registry.py`** — register `DataVizScenario` next to `CodingScenario` in both the dict and `list_scenarios()`.
- **`core/src/skill_packs/registry.py`** — add a `DataVizSkill` import block in `discover_skills()` mirroring the `CodingSkill` block.

### 9.2 New files

- `core/src/skill_packs/data_viz/{SKILL.md, REFERENCE.md, skill.py, tools.py, __init__.py}`
- `core/src/scenarios/data_viz/{scenario.py, __init__.py, cases/, templates/vite-data-story/}`
- `core/src/eval/baselines/data_viz.json` (added once Phase 1 replay is green; consumed by the existing `compare_against_baseline` path)

### 9.3 Result row shape

`DataVizScenario.evaluate` returns:

```json
{
  "test_case_id": "sales_story_001",
  "total_score": 0.86,
  "skill_planner_score": 0.9,
  "artifact_runtime_score": 1.0,
  "data_correctness_score": 0.82,
  "viz_interaction_score": 0.8,
  "story_layout_accessibility_score": 0.78,
  "install_passed": true,
  "build_passed": true,
  "test_passed": true,
  "app_preview_created": true,
  "details": {
    "missing_plan_concepts": [],
    "metric_failures": ["region_share_apac"],
    "layout_failures": []
  }
}
```

## 10. First Three Eval Cases

### DV1: Sales Growth Story

Dataset:

- monthly orders
- region
- product category
- revenue
- gross margin

Prompt:

```text
Build an interactive data story showing what drove 2025 revenue growth.
Include monthly revenue, revenue by region, top product categories, margin trend,
and filters for year and region.
```

Core checks:

- total revenue equals expected
- monthly chart has 12 buckets
- top category matches expected
- region filter recomputes KPIs
- headline mentions the correct growth driver

### DV2: Support Operations Story

Dataset:

- ticket created date
- priority
- team
- status
- first response time
- resolution time
- SLA breached

Prompt:

```text
Create a support operations story that explains where SLA breaches are coming from.
Show breach rate by priority and team, weekly ticket volume, and a filter for team.
```

Core checks:

- breach rate is calculated correctly
- highest breach team is correct
- priority breakdown includes all priorities
- team filter updates rate and charts
- data notes define breach rate

### DV3: Marketing ROI Story

Dataset:

- campaign
- channel
- spend
- impressions
- clicks
- conversions
- revenue

Prompt:

```text
Build a marketing performance story comparing channels by CAC and ROAS.
Show spend, conversions, CAC, ROAS, and identify the best and worst channel.
```

Core checks:

- CAC and ROAS formulas are correct
- best/worst channel claims match expected metrics
- chart labels include units
- channel filter updates KPIs
- no division-by-zero display errors

## 11. Implementation Phases

### Phase 0: Harness Prerequisites

Land §9.0 before any data-viz code. Without these, the agent's trace is discarded, the runner can only target `coding`, and Playwright/log/screenshot artifacts are not captured.

- Add `Trace.plan_updates`, `Trace.skill_loads`, `Trace.artifact_events` and the corresponding `record_*` methods (§9.0.1).
- Add managed-event branches in `_map_event_to_trace` for `plan_updated` and `skill_auto_loaded`, and record artifact helper events (`artifact_created`, `artifact_patch`, `artifact_finalized`) through `ToolContext.trace` (§9.0.2).
- Change `run_single` to return `(trace, score)`; update all callers (§9.0.3).
- Replace `run_coding_benchmark` with `run_app_artifact_benchmark(scenario, ...)` and parameterize workspace/case helpers on `scenario_dir` (§9.0.4).
- Add hidden-asset injection at the `test` gate only (§9.0.5).
- Capture install/build/test stdout+stderr logs, Playwright JSON, screenshots, and preview URL alongside the patch diff (§9.0.6, §9.0.7).
- Add `Scenario.is_app_artifact()`; route `coding` and `data_viz` through the generalized runner via the new CLI dispatch (§9.0.8).

Exit criteria:

- `coding` cases still pass on `--replay` after migration to the generalized runner (regression check)
- a `Trace` round-tripped through JSON contains non-empty `plan_updates`, `skill_loads`, and `artifact_events` for any agent run that exercised those paths
- gate logs and Playwright JSON appear under `results/artifacts/<case_id>/` for the existing coding case

### Phase 1: Fast Signal

- Scaffold `data_viz` skill and register in `skill_packs/registry.py`.
- Scaffold `DataVizScenario` (override `is_app_artifact` to `True`) and register in `scenarios/registry.py`.
- Add the `vite-data-story` template under `scenarios/data_viz/templates/`.
- Add `sales_story_001` with `public/`, `hidden/`, and `golden/` subdirs.
- Implement layered `evaluate()` reading `trace.gate_results`, `trace.plan_updates`, `trace.skill_loads`, `trace.artifact_events`, and the Playwright JSON from `trace.artifacts`. Implement the replay-mode branch from §6.0.
- Run replay via `--replay sales_story_001`.

Exit criteria (replay mode only — agent-execution-only layers are waived per §6.0):

- `data_correctness_score`, `viz_interaction_score`, `story_layout_accessibility_score`, and the install/build/test sub-signals of `artifact_runtime_score` are all 1.0
- `replay_total_score` is 1.0 (the renormalized total under replay-mode scoring)
- `skill_planner_score` and the artifact-event sub-signals of `artifact_runtime_score` are reported as `null` or `n/a` in replay output, never as 0
- mutating an `expected_metrics` value flips `data_correctness_score` and adds the field name to `details.metric_failures`
- hidden assets are absent from `<workspace>/cases/<case_dir>/hidden/` before and after the test gate; present only during it (verified by directory listings at each phase)

### Phase 2: Agent Runs

- Run the actual agent on the first case via the benchmark CLI (no `--replay`).
- Tune `SKILL.md` only when failure traces show a clear workflow gap.
- Add DV2 and DV3 cases following the same `public/`/`hidden/`/`golden/` layout.
- Confirm `_capture_app_artifact_outputs` stores patch diff, install/build/test logs, Playwright JSON, screenshots, and preview URL; extend if any are missing.

Exit criteria:

- at least 3 cases run end-to-end
- every failure can be attributed to exactly one layer score from the evaluator output
- workspace audit confirms hidden Playwright specs and `expected_metrics.json` are not visible to the agent before completion

### Phase 3: Product Polish

- Extend `eval/comparator.py` to diff per-layer scores and surface them in `format_regression_gate`.
- Freeze `eval/baselines/data_viz.json` from the latest stable run; consumed by the existing `compare_against_baseline` path.
- Add frontend side-panel validation for `app_preview`.
- Add responsive screenshot checks (mobile + desktop) into the Playwright spec.
- Optionally add LLM-judge rubric scoring as a sixth, low-weight layer.

Exit criteria:

- candidate agent can be compared against baseline by total and per-layer scores
- regression gate names the specific layer (process, runtime, data, viz, story) that dropped
- baseline is regenerated only via an explicit `--update-baseline` action, not silently

## 12. Open Decisions

- Whether `data_viz` should require loading `coding` explicitly or duplicate the coding workflow in its own `SKILL.md`.
- Whether to use pure SVG charts in the template or pin one chart dependency.
- Whether screenshots should be mandatory in Phase 1 or added in Phase 3.
- Whether the eval runner should generalize coding/data-viz as one "app artifact" scenario class.

Recommended defaults:

- Keep `data_viz` separate from `coding`, but repeat the required coding workflow in the `data_viz` skill prompt.
- Use pure SVG for the first template.
- Defer subjective visual judging until deterministic correctness is stable.
- Generalize the runner only after two scenarios share enough code to justify it.
