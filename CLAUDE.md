# Apex Agent — repo context for Claude

A monorepo for an autonomous agent platform: agent runtime + tools + skills (`core/`), FastAPI backend with SSE (`backend/apex_server/`), Textual TUI (`tui/`), React/Vite web UI (`frontend/`), and an eval harness (`core/src/eval/`).

## Workspace layout

```
core/                 Python lib: runtime, tools, skills, scenarios, eval harness
  src/agent/          Runtime, session, events, context assembler, policy, sandbox
  src/skill_packs/    Domain skills (coding, data_viz, stock_strategy, wealth_guide)
  src/scenarios/      Eval scenarios (coding, data_viz, core_agent, …)
  src/eval/           Benchmark runner, comparator, baselines
  src/tools/          Filesystem, shell, web, rag — typed ToolDef registry
  src/services/       RAG client, retrieval policy, web search
backend/apex_server/  FastAPI app: routes, deps, SSE streaming
backend/tests/        Server-level tests
tui/                  Textual TUI client
frontend/             React + Vite + Zustand + pnpm
doc/                  Design docs and implementation guides
datasets/             Sample data for ad-hoc agent testing (not eval fixtures)
results/              Eval output (sandboxes, traces, CSVs, baselines)
rag-service/          Standalone RAG microservice (separate process)
docker-compose.yml    Postgres for sessions/auth
```

## How to run things

```bash
# Workspace install (uv workspace, members: core, backend, tui)
uv sync

# Postgres for sessions/auth
docker compose up -d postgres

# TUI
uv run python -m tui

# FastAPI backend (dev)
uv run uvicorn apex_server.app:app --reload --port 8000

# Frontend (from frontend/)
pnpm install && pnpm dev

# Eval suite — see also doc/eval-suite.md
uv run python -m eval --scenario data_viz --replay sales_story_001
uv run python -m eval --scenario coding --replay
```

Tests:

```bash
uv run pytest                       # core/tests + backend/tests (configured in pyproject.toml)
uv run pytest core/tests/test_t1_managed_agent_properties.py -k <name>
```

`pyproject.toml` sets `asyncio_mode = "auto"`, so async tests don't need explicit decorators.

## Key concepts that bite if you don't know them

### Skill packs vs scenarios

- **Skill pack** (`core/src/skill_packs/`): a runtime capability (set of tools + `SKILL.md` instructions). Loaded by the agent at runtime, often auto-loaded from prompt intent.
- **Scenario** (`core/src/scenarios/`): an eval target (cases + manifest + evaluator). Used by the harness, not the live runtime.

Both have their own registry: `skill_packs/registry.py` (`discover_skills()`) and `scenarios/registry.py` (`get_scenario()`). New skill or scenario → add to its registry.

### App-artifact scenarios

Scenarios that build a runnable app (coding, data_viz) override `Scenario.is_app_artifact()` to return `True`. The eval runner dispatches them through `run_app_artifact_benchmark` (`core/src/eval/runner.py`), which:

1. Sets up a sandbox workspace (template + `cases/<id>/public/`).
2. Runs the agent (or applies golden patch on `--replay`).
3. Runs install/build/test gates with hidden assets injected only at the test stage.
4. Captures patch diff, gate logs, Playwright JSON, screenshots, preview URL.
5. Calls `scenario.evaluate(trace, case)` exactly once after artifacts are captured.

Don't add gate logic to the regular `run_benchmark` path — gated app scenarios go through `is_app_artifact()` dispatch.

### Events table is per-session-sequenced, not global

Schema (`core/src/agent/session/archive.py`):

```sql
events(id BIGSERIAL, session_id, seq INTEGER, event_type, timestamp, payload, content_text,
       UNIQUE(session_id, seq))
```

`seq` is per-session, assigned by `SessionArchive.emit_event` under `with self._lock`. Don't read or increment `_seq_cache` outside the lock — concurrent writers will hit the UNIQUE constraint.

### `_publish(event, persist=True)` contract

`ManagedAgentRuntime._publish` is the canonical UI-event path. **Default: `persist=True`** (writes to events table + fans out to event bus). Use `persist=False` for high-volume live-only events:

- `AssistantToken` → `persist=False` (every LLM token would otherwise create a row)
- Everything else → `persist=True` (default)

Adding a new live-only event? Match the AssistantToken call site at `managed_runtime.py:612`. Adding a new durable event? Default behavior is correct.

### `stream_end` is **not** the same as `turn_finished`

`StreamEnd` (`core/src/agent/events/schema.py:109`) carries `final_state ∈ {completed, waiting_approval, failed, cancelled}` and fires at every SSE burst boundary, including approval pauses. Composer unlock and "turn over" logic must branch on `final_state`:

| `final_state` | turn over? | composer unlocks? |
|---|---|---|
| `completed` / `cancelled` / `failed` | yes | yes |
| `waiting_approval` | **no** | **no** — show approval UI |

`turn_finished` finalizes assistant text; `stream_end` is the source of truth for unlock.

### SSE: named events, no client cursor

Server emits **named** SSE events (`event: assistant_token`, etc.). Frontend in `frontend/src/hooks/useSSE.ts` registers `addEventListener(name, ...)` per event type — adding a new typed event requires adding it to that list, otherwise it's silently dropped.

No localStorage cursor. Refresh replays from `last_turn_started.seq - 1` (server-side cutoff). Brief disconnects use `Last-Event-ID` automatically. See `doc/agent-refresh-reconnect-implementation-guide.md`.

### Session/run model

- `POST /sessions/{id}/turns` — owns the run; spawns a background asyncio task (`SharedTurnRunner.start_turn_background`).
- The background task is the run owner. Killing the SSE connection does **not** stop the agent.
- `active_run` dict on `ManagedAgentRuntime` tracks status + heartbeat. **In-memory only**: no backend-restart recovery yet (Phase 1).

## Code conventions

### Python

- `from __future__ import annotations` at top of every module.
- Pydantic v2 models for typed events and structured payloads.
- Async by default at the runtime/server boundary; sync at the storage layer (`SessionArchive` uses sync psycopg). Bridge with `asyncio.to_thread` when calling sync from async.
- No comments narrating *what* code does. Comments only for non-obvious *why*: hidden invariant, workaround, surprising constraint.
- Don't add backwards-compat shims, fallbacks for impossible cases, or feature flags unless explicitly asked.

### TypeScript / React

- Zustand store in `frontend/src/store.ts` is the single source of UI state.
- `useStore((s) => s.field)` selectors only — don't `useStore()` without a selector.
- SSE is the input to the store; reducers are idempotent and key on stable ids (`turn_id`, `tool_call_id`, `artifact_id`, `seq`).
- Tailwind utilities for layout; `index.css` for tokens/themes; one component per file.

### Tests

- Pytest with `asyncio_mode = "auto"` — write `async def test_…` directly.
- Mocks for LLM responses live in `core/src/eval/mock_brain.py` and friends; avoid mocking the database — use the real Postgres in `docker-compose.yml`.
- Eval mock mode: `APEX_MOCK_LLM=1` env var or `--mock` flag.

## Writing design docs (`doc/*.md`)

Design and implementation docs in this repo have a recurring failure mode: snippets reference functions that don't exist, fields that were renamed, or line numbers that drifted. The architecture is right, but the doc isn't implementable without grepping. The rules below exist to make verification cheaper than guessing.

### Read before write

Before writing any code snippet, **read the file you're referencing**. Copy the actual function signature/header verbatim into the snippet. Do not write a snippet whose function name you have not seen in `grep` output. Retyping signatures from memory is where drift enters.

If the function or field doesn't exist yet, that's fine — but the snippet must be marked as new (see "Anchor every snippet" below).

### Anchor every snippet

Every code block in a design doc must cite the file it lives in (or will live in) **and** declare one of three states:

```python
# core/src/agent/runtime/trace.py:Trace.record_tool_call (existing)
def record_tool_call(self, *, step: int, name: str, ...): ...

# core/src/agent/runtime/managed_runtime.py:_map_event_to_trace — TO ADD
elif event.type == "plan_updated":
    trace.record_plan_update(...)

# core/src/eval/runner.py:_run_coding_gates — REPLACES
async def _run_app_artifact_gates(...): ...
```

States: **existing** (must match current code), **TO ADD** (new code, name and signature must not collide), **REPLACES** (names current code being replaced, with old function/range cited). No fourth state — no orphan snippets without an anchor.

### Function name > line number

Line numbers drift every refactor. Function names drift only when intentionally renamed.

- ✅ `managed_runtime.py:_map_event_to_trace`
- ⚠️ `managed_runtime.py:1102` (will go stale)
- ✅ best: `managed_runtime.py:_map_event_to_trace ~line 1102` (both anchors, line as rough orientation)

When citing a specific recorder/helper, also cite its definition file — `record_tool_call` is in `trace.py:134` and called from `managed_runtime.py:1257`. Pair the call site with the definition site so a single move doesn't break both halves of the citation.

### Recurring drift patterns to watch for

Specific failures that have already cost time on this repo:

- **Field renames silently propagate.** `trace.events` (does not exist) vs `trace.skill_loads`/`plan_updates`/`artifact_events` (real). `event_type` (real DB column) vs `type` (intuitive but wrong). `final_state` (real on `StreamEnd`) vs `stop_reason` (different concept).
- **Invented helper signatures.** `event_bus.subscribe(into=queue)` and `sub.close()` do not exist; the real API is an async iterator from `subscribe(session_id, *, since_seq=None)`. `db.transaction()` and `fetchval` are not the storage layer; the real one is sync `SessionArchive.emit_event(session_id, event_type, payload) → int`.
- **CLI flag drift.** `--apply-golden` does not exist; the real flag is `--replay`.
- **Generic SSE listener.** `addEventListener("message", ...)` does not catch named events. The frontend uses one listener per event name.

### Doc-review checklist

Before merging any design doc that contains code snippets, verify:

1. Every field name appears in `grep` of the cited file.
2. Every function called in a snippet has the cited signature in current code.
3. Every line number — if used at all — points at the line claimed.
4. Every snippet is anchored as **existing**, **TO ADD**, or **REPLACES**.

If all four pass, the doc is implementable as-is. If any one fails, the doc isn't.

## Where to find existing patterns

| Need to … | Look at |
|---|---|
| Add a new skill pack | `core/src/skill_packs/coding/` (skill.py, tools.py, SKILL.md) + register in `skill_packs/registry.py` |
| Add a new scenario | `core/src/scenarios/coding/scenario.py` (regular) or `data_viz/scenario.py` (app-artifact) + register in `scenarios/registry.py` |
| Add a typed event | `core/src/agent/events/schema.py` — add class + add to the union, then add to `useSSE.ts` listener list and `store.ts` reducer |
| Emit an event from runtime | `await self._publish(MyEvent(...))` (or `persist=False` for live-only) in `managed_runtime.py` |
| Add a tool | `core/src/tools/<name>.py` with `ToolDef`; register in `tools/__init__.py`; or scope it to a skill pack |
| Add a sample dataset | `datasets/<name>/sales.csv` + README — keep separate from `core/src/scenarios/<name>/cases/*/public/` (eval fixtures) |
| Add a backend route | `backend/apex_server/routes/<area>_routes.py`, mount in `backend/apex_server/app.py` |

## Important docs

- `doc/design-spec.md` — full architecture
- `doc/design-checklist.md` — design principles enumerated
- `doc/agent-refresh-reconnect-implementation-guide.md` — SSE/reconnect contract (canonical for any session/streaming work)
- `doc/data-viz-story-agent-implementation-guide.md` — data-viz scenario + harness extensions
- `doc/coding-ability-regression-harness.md` — coding scenario eval contract
- `doc/eval-suite.md` — running and interpreting the eval harness
- `doc/wealth-guide-implementation-plan.md` — wealth-guide scenario design

## Things to avoid

- **Don't** persist every event indiscriminately — respect `persist=False` for tokens. A 4000-token response would otherwise create 4000 rows.
- **Don't** read or increment `_seq_cache` outside `with self._lock` in `archive.py`.
- **Don't** treat `stream_end` and `turn_finished` as interchangeable — `waiting_approval` is a stream_end with no turn_finished.
- **Don't** use generic `addEventListener("message", …)` for SSE — server emits named events.
- **Don't** add localStorage for SSE cursor; the design relies on per-turn server-side replay + EventSource's automatic `Last-Event-ID`.
- **Don't** modify `core/src/scenarios/<scenario>/cases/<id>/public/` if you need a test dataset — those are eval fixtures with hidden expected_metrics. Use `datasets/` instead.
- **Don't** bypass the skill registry by importing skill modules directly from outside — go through `discover_skills()` so loaders see them consistently.
- **Don't** add comments narrating what code does. If a reader can derive the *what* from the names, the comment is noise.

## Current state of major in-flight work

- **Refresh/reconnect** (per-turn replay, snapshots, persist flag): runtime + SSE + frontend reducer all landed (see `doc/agent-refresh-reconnect-implementation-guide.md` step status). Remaining: `GET /sessions/{id}/turns` summary endpoint and `TurnNavigator.tsx`.
- **Data-viz scenario**: skill pack, scenario, evaluator, vite-data-story template, `case_001` all present. Generalized `run_app_artifact_benchmark` is wired. Remaining: comparator per-layer diff (Phase 3 of that guide).
- **Backend-restart recovery**: not implemented; Phase 2 of refresh guide. UI surfaces "agent stopped unexpectedly" instead.

## Plain-text contract for any AI agent editing this repo

1. Read the relevant doc in `doc/` before changing the matching subsystem.
2. Use `Read` before `Edit`; `Edit` requires a prior `Read` of the file.
3. For UI features, run `pnpm dev` and verify in a browser before declaring done.
4. For runtime changes, prefer adding a focused test in `core/tests/` over a one-off script.
5. Match conventions of the file you're in. Don't impose a different style across the boundary.
