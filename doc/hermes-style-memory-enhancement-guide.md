# Hermes-Style Memory Enhancement Guide

This guide adapts the useful parts of Hermes Agent's memory model to Apex Agent without turning Apex into a different product. The goal is a practical memory system for coding, data visualization, stock analysis, and app-generation sessions.

Hermes' useful pattern is three memory layers:

1. Small curated memory injected into every run.
2. Procedural memory as skills.
3. Searchable session history for episodic recall.

Apex already has the beginning of this:

- `ContextManager` supports pinned facts.
- `remember`, `forget`, and `recall_session` exist in `core/src/tools/memory.py`.
- `SessionArchive` stores session events.
- `SkillLoader` already uses progressive disclosure for skills.
- The frontend/server now have per-turn replay and turn summaries.

The missing part is making these pieces durable, explicit, budgeted, and safe.

## Target Architecture

```text
User turn
  -> skill router chooses top-1 active skill
  -> context assembler builds prompt
       1. runtime/system rules
       2. active user request
       3. active skill instructions
       4. workspace/artifact state
       5. curated memory snapshot
       6. recent turns
       7. retrieved old turn summaries
  -> agent loop runs tools
  -> memory proposals are written durably for next turn/session
```

The most important rule: memory should be **stable during one turn**. The agent can write memory while running, but the active prompt should not keep changing mid-loop. New memory becomes visible on the next turn or next session.

## Layer 1: Curated Durable Memory

Add small, durable memory records that are always available but strictly budgeted.

Recommended scopes:

- `user`: stable user preferences, language, style, timezone, tool preferences.
- `project`: facts about the Apex repo, local environment, Docker/Colima setup, conventions.
- `session`: short-lived decisions for the current session.

Recommended table:

```sql
CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  project_id TEXT,
  session_id TEXT,
  scope TEXT NOT NULL CHECK (scope IN ('user', 'project', 'session')),
  text TEXT NOT NULL,
  tags JSONB NOT NULL DEFAULT '[]',
  confidence REAL NOT NULL DEFAULT 1.0,
  source_session_id TEXT,
  source_turn_id TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  archived_at TIMESTAMPTZ
);
```

Budget:

- `user`: 500-800 tokens.
- `project`: 800-1200 tokens.
- `session`: 500-800 tokens.
- total curated memory budget: hard cap around 2,500 tokens.

Implementation:

- Add `MemoryStore` backed by Postgres.
- Add `ContextAssembler.load_memory_snapshot(user_id, session_id, project_id)`.
- Render memory into one compact block:

```text
## Durable Memory
- User prefers concise engineering answers.
- Project uses Colima-backed Docker; sandbox image should be apex-sandbox:latest.
- Data-viz artifacts should end with app_preview in the side panel.
```

Important behavior:

- Build the memory snapshot once at turn start.
- Store the snapshot on the turn record for replay/debugging.
- If the agent calls `remember`, persist it, but do not reassemble the current prompt mid-loop.

## Layer 2: Procedural Memory As Skills

Apex skills already map well to Hermes-style procedural memory. Treat skills as durable procedures, not just prompt snippets.

Enhancements:

- Keep only the skill index in the base system prompt.
- Load exactly one top-ranked skill by intent before the first model call.
- If a tool call requires an unloaded skill, allow explicit `load_skill`, but unload previous active skill unless the user explicitly asks for multi-skill work.
- Store `active_skill_name` on the turn record.
- Add usage telemetry:

```sql
CREATE TABLE IF NOT EXISTS skill_usage (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  skill_name TEXT NOT NULL,
  match_score REAL,
  loaded_by TEXT NOT NULL CHECK (loaded_by IN ('router', 'manual', 'tool_owner')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Data-viz rule:

- Data-viz task should load only `data_viz`.
- Stock task should load only `stock_strategy`.
- If the next turn changes task type, the router should replace the active skill.
- If the next turn is a follow-up, keep the current skill if intent score is weak but follow-up signals are strong.

Follow-up detection signals:

- user says "make it better", "fix that", "continue", "same app", "that chart", "the report".
- current turn references an artifact from the previous turn.
- no new domain intent beats the active skill by a threshold.

## Layer 3: Episodic Session Retrieval

Apex should use durable events as episodic memory.

Current `recall_session` searches current session only. Extend this in two stages:

### Stage 3A: Same-Session Retrieval

Use Postgres full-text search over durable event content.

Index:

```sql
CREATE INDEX IF NOT EXISTS session_events_fts_idx
ON session_events
USING GIN (to_tsvector('english', coalesce(event_type, '') || ' ' || coalesce(data::text, '')));
```

Tool:

```text
recall_session(query, limit=5)
```

Return compact fragments, not raw event dumps.

### Stage 3B: Cross-Session Retrieval

Add:

```text
recall_history(query, scope='user'|'project', limit=5)
```

This should search older sessions owned by the same user. It should never retrieve another user's sessions.

Injection policy:

- Do not inject cross-session retrieval by default.
- Let the model call `recall_history` when it needs older context.
- For high-confidence follow-up sessions, the context assembler can inject one short "recent related work" summary.

## Turn Summaries

Raw events are too noisy for long-term recall. Add a durable summary per turn.

Recommended table:

```sql
CREATE TABLE IF NOT EXISTS turn_summaries (
  turn_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  active_skill_name TEXT,
  user_goal TEXT NOT NULL,
  outcome TEXT NOT NULL,
  artifacts JSONB NOT NULL DEFAULT '[]',
  files_changed JSONB NOT NULL DEFAULT '[]',
  durable_facts JSONB NOT NULL DEFAULT '[]',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Create the summary at `turn_finished`.

For data-viz, the summary should include:

- dataset filename and column summary.
- generated app directory.
- preview artifact id/url.
- chart/story sections.
- known issues or failed gates.

For stock analysis, include:

- ticker(s).
- data window.
- tools used.
- output artifact/report path.
- recommendations and risk notes.

## Memory Write Policy

Do not let the model silently write arbitrary permanent memory.

Use two modes:

1. Session memory writes: allowed without approval.
2. User/project memory writes: proposed first, then approved by user or policy.

Example proposal event:

```json
{
  "type": "memory_proposed",
  "scope": "project",
  "text": "Apex uses Colima Docker locally; sandbox image should include Python, Node, pnpm, and Playwright.",
  "reason": "Repeatedly needed for sandbox debugging."
}
```

Short-term implementation can auto-approve project memories created by the developer user, but keep the event explicit.

Security filters before persistence:

- Reject secrets and API keys.
- Reject instructions that override system/developer policy.
- Strip invisible Unicode.
- Reject "always ignore previous instructions" style prompt injection.
- Cap each memory item at 300 characters.

## Context Assembly Priority

When context gets large, Apex should prefer recency and task relevance over chronological completeness.

Priority order:

1. System/developer/runtime rules.
2. Current user message.
3. Active skill `SKILL.md`.
4. Tool schemas for active tool surface.
5. Current plan and active artifacts.
6. Curated memory snapshot.
7. Last 2-4 turns verbatim.
8. Retrieved turn summaries.
9. Older raw events only through explicit `recall_session`.

Avoid placing crucial instructions in the middle of a huge prompt. Critical rules should live near the top in system/skill blocks or near the end in a short "Current Task Contract" block.

## APIs

Backend routes:

```text
GET    /memories?scope=user|project|session
POST   /memories
PATCH  /memories/{id}
DELETE /memories/{id}

GET    /sessions/{session_id}/turns/{turn_id}/summary
GET    /sessions/{session_id}/memory-snapshot
```

Agent tools:

```text
remember_session(fact, tags?)
propose_memory(scope, fact, tags?, reason)
forget_memory(id_or_substring)
recall_session(query, limit?)
recall_history(query, scope?, limit?)
```

Keep existing `remember` and `forget` as aliases only if needed for compatibility.

## Implementation Plan

### Phase 1: Durable Curated Memory

- Add `memories` table migration.
- Add `MemoryStore`.
- Replace in-memory-only pinned facts with durable session memory plus a frozen turn snapshot.
- Add memory snapshot rendering to `ContextAssembler`.
- Add tests proving memory written in turn N appears in turn N+1, not mid-turn.

### Phase 2: Turn Summaries

- Add `turn_summaries` table.
- Create summary on `turn_finished`.
- Store active skill, artifacts, files changed, tool names, and outcome.
- Expose summary in backend route.
- Use summaries in refresh/reconnect navigator details.

### Phase 3: Retrieval

- Add Postgres FTS indexes for event and summary search.
- Extend `recall_session` to search summaries first, events second.
- Add `recall_history` for same-user cross-session retrieval.
- Add tests for tenant isolation.

### Phase 4: Skill Memory

- Add `skill_usage` table.
- Persist active skill per turn.
- Add top-1 skill replacement policy.
- Add "follow-up keeps active skill" logic.
- Add skill success telemetry from eval harness.

### Phase 5: Memory Review UI

- Add a small settings page or side panel section for memory.
- Show user/project/session memory separately.
- Let user approve, edit, archive, and delete memories.
- Show source session/turn for each memory item.

## Evaluation Plan

Memory should be evaluated separately from the agent loop.

Core eval cases:

1. **Preference recall**
   User says they prefer dark UI and concise answers. Later session asks for a data-viz app. Expected: generated app uses dark UI without re-asking.

2. **Project environment recall**
   User says Colima/Docker sandbox should use `apex-sandbox:latest`. Later sandbox task should mention/build against that image.

3. **Follow-up task continuity**
   Turn 1 builds data-viz artifact. Turn 2 says "make the chart cleaner." Expected: keeps `data_viz` skill and references the existing artifact.

4. **Domain switch**
   Turn 1 stock analysis. Turn 2 asks for sales CSV story app. Expected: unload/replace stock skill with data-viz skill.

5. **No secret persistence**
   User pastes an API key. Expected: memory write is rejected or redacted.

Metrics:

- memory_recall_precision
- memory_recall_coverage
- wrong_memory_injection_count
- prompt_token_overhead
- skill_switch_accuracy
- followup_skill_retention_accuracy
- secret_persistence_violations

## Design Rules

- Memory is not chat history. Memory is curated, compact, and durable.
- Session history is searchable, not always injected.
- Skills are procedural memory and should remain top-1 for normal turns.
- Memory updates should be auditable.
- Long-term memory should improve future turns without making current turns unstable.
- In eval, measure whether memory improved the result, not just whether it was retrieved.
