# Agent Refresh, Reconnect, and Background Run Guide

## Goal

Make Apex behave correctly when the SSE connection dies (refresh, browser close + reopen, transient network drop) **without** adding any moving parts. The constraints, in order:

1. The agent task keeps running as long as the backend process is alive.
2. Reconnect rebuilds the UI from the database. No client-side cursor, no snapshot tables, no Redis until horizontal scaling demands it.
3. Reconnect is fast at any session age — refresh latency is bounded by the size of the **current turn**, not the session.
4. The UI exposes a turn navigator on the right edge so the user can jump between turns at any time.

The browser's SSE connection is a subscriber, not the job owner. Killing it does nothing to the running agent.

## Core principle

```
RUN OWNERSHIP    background asyncio task on backend (lives until turn done)
DURABILITY       Postgres events table (records "what happened")
LIVE DELIVERY    in-process event bus today; Redis pub/sub when multi-worker
```

The browser holds **zero** persistent state about the session. Every reconnect derives state from the DB.

## What a turn is

```
turn_started          ← emitted at POST /sessions/{id}/turns
  one or more LLM calls
  zero or more tool calls
  plan updates
  approval requests + resolutions
  artifact_created / artifact_patch / artifact_finalized
  assistant_token (live only, never persisted)
  assistant_snapshot every 500 chars or 1s (durable, replaces streaming text)
  assistant_message (durable, final text)
turn_finished         ← agent's final message done; frontend finalizes text
stream_end            ← marks an SSE turn boundary; carries final_state
```

### `stream_end` vs `turn_finished` — they are not the same

The current `StreamEnd` event (`core/src/agent/events/schema.py:109`) carries `final_state ∈ {"completed", "waiting_approval", "failed", "cancelled"}` and is emitted at **any** boundary where the LLM stream halts — including pauses for human approval.

```
final_state="completed"          turn ended normally
final_state="cancelled"          user cancelled mid-turn
final_state="failed"             error halted the turn
final_state="waiting_approval"   turn is PAUSED, not ended
```

The frontend must read `final_state` to decide whether the composer unlocks:

| final_state | turn over? | composer unlocks? |
|---|---|---|
| `completed` | yes | yes |
| `cancelled` | yes | yes |
| `failed` | yes | yes |
| `waiting_approval` | **no** | **no — show approval UI** |

Treat `stream_end` as "this SSE burst is done." Treat `turn_finished` as "this turn is terminal." The two coincide for `completed`/`cancelled`/`failed`, but for `waiting_approval` the runtime emits `stream_end(waiting_approval)` only — no `turn_finished` — and resumes a new burst once the user resolves the approval.

User-facing definition: **a turn ends when the input box re-enables.** That happens on `stream_end` if and only if `final_state != "waiting_approval"`.

Pause cases that do *not* end a turn: pending approval (`stream_end(waiting_approval)`), mid-tool execution (no `stream_end`).

End cases that *do*: `stream_end(completed)`, `stream_end(cancelled)`, `stream_end(failed)`.

### Note on `assistant_message`

The current runtime publishes the final assistant text via `turn_finished(content=...)` (`core/src/agent/runtime/managed_runtime.py:715`) and a legacy `assistant_message_added` event for archive search. The typed `assistant_message` event exists in the schema but is **not consistently published**. To make the streaming model in §"Streaming during refresh" work, the runtime must either:

- Start publishing typed `assistant_message` on turn end (preferred — clear separation of concerns), or
- Have the frontend treat `turn_finished.content` as the final assistant text (current behavior).

This guide assumes option 1. If you ship option 2, treat every reference to `assistant_message` below as `turn_finished.content` instead.

## The replay cutoff: most recent turn

Reconnect with no cursor ⇒ replay everything from the most recent `turn_started` onward. The current schema (`core/src/agent/session/archive.py:35`) is:

```sql
CREATE TABLE events (
    id            BIGSERIAL PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    seq           INTEGER NOT NULL,                     -- per-session, not global
    event_type    TEXT NOT NULL,
    timestamp     DOUBLE PRECISION NOT NULL,
    payload       JSONB NOT NULL,
    content_text  TEXT,
    UNIQUE(session_id, seq)
);
CREATE INDEX idx_events_session ON events(session_id, seq);
```

`seq` is already per-session and unique. Don't change the type. The only addition needed is a composite index for the per-turn cutoff query:

```sql
CREATE INDEX IF NOT EXISTS idx_events_session_type_seq
  ON events (session_id, event_type, seq);
```

The cutoff query (note: column is `event_type`, not `type`):

```sql
SELECT seq, event_type, timestamp, payload
FROM events
WHERE session_id = %s
  AND seq >= (
    SELECT COALESCE(MAX(seq), 0) FROM events
    WHERE session_id = %s AND event_type = 'turn_started'
  )
ORDER BY seq ASC;
```

The inner query uses the new composite index for a single index seek; the outer is a bounded range scan on `idx_events_session`. Sub-millisecond at any session age.

Bound on size: a turn rarely exceeds a few hundred events even for complex multi-tool reasoning. Refresh latency stays near-constant regardless of how long the session has lived.

## Reconnect modes (two endpoints)

### `GET /sessions/{id}/events` — the live SSE stream

```
no cursor               → from most recent turn_started (default)
Last-Event-ID: <seq>    → only events with seq > <seq>
```

- **No cursor**: hard refresh, browser close + reopen, fresh tab, new device. Replays the active turn.
- **`Last-Event-ID`**: the browser's `EventSource` sends this automatically on transient reconnects. Server reads it (or the `last_event_id` query param) as a numeric cursor. Free for the client; one extra `WHERE seq > $cursor` clause for the server.

This endpoint does **not** load older turns. Its job is to render the current turn and continue the live feed. That's the bound that keeps refresh fast on long sessions.

### `GET /sessions/{id}/turns` — the turn-summary list

The right-side turn navigator needs to show every historical turn in the session, including ones the SSE stream did not replay. A separate lightweight endpoint serves a compact summary per turn:

```
GET /sessions/{id}/turns           → all turns, newest first, ~one row per turn
GET /sessions/{id}/turns?before=<seq>&limit=50   → pagination for very long sessions
```

Response shape:

```json
[
  {
    "turn_id": "t_abc",
    "started_at": 1778030000.0,
    "started_seq": 8421,
    "ended_at": 1778030099.0,
    "ended_seq": 8447,
    "status": "completed",     // completed | running | waiting_approval | failed | cancelled
    "user_preview": "build the sales story",
    "assistant_preview": "Here's the report..."
  },
  ...
]
```

Server-side this is a single query joining `turn_started` and the matching `stream_end`:

```sql
SELECT
  t.payload ->> 'turn_id'                                AS turn_id,
  t.timestamp                                            AS started_at,
  t.seq                                                  AS started_seq,
  e.timestamp                                            AS ended_at,
  e.seq                                                  AS ended_seq,
  COALESCE(e.payload ->> 'final_state', 'running')       AS status,
  LEFT(t.payload ->> 'user_input', 80)                   AS user_preview
FROM events t
LEFT JOIN events e
  ON e.session_id = t.session_id
 AND e.event_type = 'stream_end'
 AND (e.payload ->> 'turn_id') = (t.payload ->> 'turn_id')
WHERE t.session_id = %s
  AND t.event_type = 'turn_started'
ORDER BY t.seq DESC;
```

The frontend calls this once on session open to populate the navigator, and updates it incrementally from the live event stream as new turns start/end. **Older turns' full event histories are not loaded into the store** — clicking a navigator row scrolls to the turn anchor; if the user wants to expand events for an old turn, the UI fetches that turn's events on demand via a third (optional) endpoint:

```
GET /sessions/{id}/events/range?from_seq=8421&to_seq=8447
```

This resolves the contradiction: per-turn replay keeps SSE reconnects fast (current turn only), and a separate compact summary endpoint feeds the navigator without dragging full event histories into the page.

No localStorage. No client-side cursor management. The only "cursor" in play is the one the browser sets for itself in the SSE protocol, and it is only relevant for transient reconnects within the same JS context.

## Streaming during refresh

`assistant_token` is live-only and not persisted. To make refresh during streaming work, the runtime emits a durable `assistant_snapshot` every ~500 chars or 1s containing the **full accumulated text so far**.

```python
# core/src/agent/events/schema.py — alongside AssistantToken / AssistantMessage,
# both of which inherit from AgentEventBase.
class AssistantSnapshot(AgentEventBase):
    type: Literal["assistant_snapshot"] = "assistant_snapshot"
    content: str
    turn_id: str | None = None
```

In `_run_loop` while streaming:

```python
SNAPSHOT_EVERY_CHARS = 500
SNAPSHOT_EVERY_SECONDS = 1.0
last_snap_at = time.time()
last_snap_len = 0

# inside the token loop, after appending delta to full_content:
should_snap = (
    len(full_content) - last_snap_len >= SNAPSHOT_EVERY_CHARS
    or time.time() - last_snap_at >= SNAPSHOT_EVERY_SECONDS
)
if should_snap:
    await self._publish(AssistantSnapshot(
        **self._bus_kwargs(), content=full_content, turn_id=self.session.turn_id,
    ))
    last_snap_len = len(full_content)
    last_snap_at = time.time()
```

Frontend handler:

```ts
case "assistant_token":     // append delta to streamingText[turnId]
case "assistant_snapshot":  // replace streamingText[turnId] with ev.content
case "assistant_message":   // finalize: push as message, clear streamingText[turnId]
```

Refresh during streaming behavior:

1. Server replays the active turn including the most recent `assistant_snapshot`.
2. Frontend rebuilds the partial assistant message from that snapshot (~ ≤500 chars behind).
3. Server attaches to the live stream; new tokens continue from where the snapshot left off.

User-visible: the page reload restores the in-progress message and the typewriter effect keeps going. The lost ground is at most one snapshot interval.

## Backend publish path

Every UI-relevant event flows through one helper, but **not every event is persisted**. High-volume live-only events (notably `assistant_token`) bypass durable storage; everything else lands in the `events` table via `SessionArchive.emit_event`.

### Two-tier publish: durable vs live-only

The runtime currently calls `await self._publish(AssistantToken(...))` for every streamed token (`core/src/agent/runtime/managed_runtime.py:572`). If `_publish` always wrote to DB, a 4000-token response would create 4000 rows and consume 4000 per-session seq values — defeating the whole point of having `assistant_snapshot` as the durable form.

Add a `persist` flag (default `True`) to `_publish`. The token emit site sets `persist=False`; everything else takes the default:

```python
async def _publish(self, event: AgentEvent, *, persist: bool = True) -> None:
    session_id = self.session.session_id

    if persist:
        # 1. Persist via SessionArchive (sync psycopg, run in thread).
        payload = event.model_dump(exclude={"seq", "timestamp", "session_id"})
        seq = await asyncio.to_thread(
            self.archive.emit_event, session_id, event.type, payload,
        )
        event.seq = seq

        # 2. Heartbeat the active run.
        #    Phase 1: in-memory dict (does NOT survive backend restart — see §Active-run).
        #    Phase 2: Postgres row in active_runs table (durable).
        self.active_run[session_id] = {
            "status": "running",
            "last_heartbeat_at": time.time(),
            "last_seq": seq,
            "turn_id": self.session.turn_id,
        }
    # When persist=False, leave event.seq = 0. The EventBus must preserve that
    # zero seq, and the SSE encoder must omit `id` for these live-only frames
    # so EventSource does not advance Last-Event-ID past durable DB rows the
    # browser has not seen.

    # 3. Fan out live to current subscribers.
    await self.event_bus.publish(session_id, event)
```

Update the token call site to opt out of persistence:

```python
# managed_runtime.py:572 — change from:
await self._publish(AssistantToken(**self._bus_kwargs(), text=delta.content))
# to:
await self._publish(AssistantToken(**self._bus_kwargs(), text=delta.content),
                    persist=False)
```

The durable surface for streaming text is `assistant_snapshot` (every ~500 chars / 1s) plus the final `assistant_message` / `turn_finished(content=...)`. Refresh during streaming reconstructs from the most recent persisted snapshot; lost in-flight tokens are bounded to one snapshot interval.

`Last-Event-ID` is a **durable DB cursor only**. Live-only events such as `assistant_token` must not advance it. If a non-persisted token frame emitted `id: 12`, the next reconnect would ask the database for `seq > 12` even though DB event `seq=12` may be a later `assistant_snapshot` that the browser never received. That would skip durable state during replay. Therefore:

- persisted events get positive DB `seq` values and SSE `id: <seq>`
- live-only events keep `seq=0` and the SSE frame omits `id`
- frontend replay dedupe only compares events with `seq > 0`
- reconnect/refresh always resumes from the latest durable DB cursor, never from live-only token order

### EventBus must not assign durable seq

The current `InMemoryEventBus.publish` stamps `event.seq` when it is `0`. That was acceptable when the bus was the only replay source, but it is wrong once DB replay becomes authoritative: a local bus seq can leak into SSE `id` and corrupt `Last-Event-ID`.

Change the bus so it treats `seq=0` as explicitly live-only:

```python
async def publish(self, session_id: str, event: AgentEvent) -> None:
    async with self._lock:
        if event.seq > 0:
            self._seq_counters[session_id] = max(
                self._seq_counters[session_id], event.seq
            )
        # event.seq == 0 remains 0. Queue order is enough for live-only tokens;
        # they are not durable and are not replayed by since_seq.
        self._buffers[session_id].append(event)
        queues = list(self._subscribers.get(session_id, ()))
    for q in queues:
        await q.put(event)
```

The replay drain remains `if ev.seq > since_seq`, so durable events replay and live-only `seq=0` events do not. This is intentional: refresh reconstructs streaming text from `assistant_snapshot`, not from old token deltas.

### Persist-then-fan-out ordering

Order matters: persist before fan-out. A crash after persist but before publish leaves DB consistent and the next reconnect replays the event normally. A crash after publish but before persist would diverge the live stream from durable history — never do it that way.

### Seq assignment must happen inside the lock

The current `emit_event` (`core/src/agent/session/archive.py:128`) reads and updates `_seq_cache` **outside** the lock:

```python
# CURRENT — RACE-PRONE under asyncio.to_thread concurrency
def emit_event(self, session_id, event_type, payload) -> int:
    seq = self._seq_cache.get(session_id, 0) + 1   # ← outside lock
    self._seq_cache[session_id] = seq              # ← outside lock
    content_text = _extract_searchable_text(event_type, payload)
    with self._lock, self.db.cursor() as cur:
        cur.execute(INSERT ...)
    return seq
```

Two threads can interleave between the `.get()` and `[id] = seq` assignment, both compute the same seq, and the second INSERT will violate `UNIQUE(session_id, seq)`. Once `_publish` runs `emit_event` via `asyncio.to_thread`, this race becomes reachable on every parallel tool boundary.

**Required fix in `archive.py`:**

```python
def emit_event(self, session_id, event_type, payload) -> int:
    content_text = _extract_searchable_text(event_type, payload)
    with self._lock, self.db.cursor() as cur:
        seq = self._seq_cache.get(session_id, 0) + 1
        self._seq_cache[session_id] = seq
        cur.execute(
            """
            INSERT INTO events (
                session_id, seq, event_type, timestamp, payload, content_text
            ) VALUES (%s, %s, %s, %s, CAST(%s AS JSONB), %s)
            """,
            [session_id, seq, event_type, time.time(),
             json.dumps(payload, default=str), content_text],
        )
    return seq
```

After this fix, `emit_event` truly serializes both seq assignment and INSERT under one lock acquisition. This must land in the same change as the `_publish` rewrite, otherwise the new concurrent-thread call pattern will produce UNIQUE constraint failures under load.

### Avoid double-writing legacy events

Today the runtime persists UI-relevant state through two paths:

- typed `_publish` → `event_bus` (live SSE, ephemeral until this guide lands)
- `_persist_session` → archive (durable, but uses **legacy** event names like `assistant_message_added`, `tool_message_added`, `user_message_added`, etc.)

Once `_publish` becomes the canonical typed-event persistence path, **do not also emit the legacy mirror** for the same UI moment. Concretely:

- For each legacy event the archive currently writes (e.g. `assistant_message_added`), pick exactly one of:
  1. Replace it with the typed event (e.g. `assistant_message`) and update `_extract_searchable_text` in `archive.py` to recognize the typed name for content recall, or
  2. Keep the legacy event for archive search **only** and stop publishing it through `_publish`.
- Do not write both. Both paths land in the same `events` table; double-writing produces duplicate rows that increment `seq` independently, which corrupts replay ordering and the per-turn cutoff query.

Recommended migration: rename the legacy events to their typed equivalents in one PR, update `_extract_searchable_text`'s switch table, and delete the legacy emit sites. Backfill is unnecessary — old rows under the legacy names remain valid history; only the cutover matters.

## Backend subscribe path (race-free)

The current `EventBus` interface (`core/src/agent/events/bus.py:24`) is:

```python
class EventBus(Protocol):
    async def publish(self, session_id: str, event: AgentEvent) -> None: ...
    async def subscribe(
        self, session_id: str, *, since_seq: int | None = None,
    ) -> AsyncIterator[AgentEvent]: ...
```

`subscribe` returns an async iterator with a built-in `since_seq` cursor that drains the in-memory ring buffer before live delivery. There is no `into=` parameter and no separate close handle — close is implicit on iterator exit.

The current `events_routes.py` already does replay-then-subscribe in roughly the right shape. Two changes are needed:

1. Compute the cursor as `max(last_event_id, last_turn_started_seq - 1)` so a no-cursor reconnect still gets the per-turn cutoff.
2. Use the existing `EventBus.subscribe(session_id, since_seq=cursor)` iterator — do not invent a new API.

```python
async def event_gen():
    # Cursor: prefer Last-Event-ID; otherwise fall back to per-turn cutoff.
    if since_seq > 0:
        cursor = since_seq
    else:
        cursor = await asyncio.to_thread(_last_turn_cutoff, archive, session_id)
        # cutoff is "seq of most recent turn_started, minus 1" so the inner
        # `> cursor` logic streams the turn_started event itself.

    # 1. Durable replay first. load_replay_events queries the archive
    #    for events with seq > cursor and returns typed AgentEvent objects.
    replay = await load_replay_events(state, session_id, cursor)
    for ev in replay:
        cursor = max(cursor, ev.seq)
        yield _encode_event(ev)

    # 2. Live subscribe — the InMemoryEventBus drains its ring buffer for any
    #    events with seq > cursor BEFORE registering the queue, so the seam
    #    between replay and live is already race-free in the existing impl.
    while True:
        if await request.is_disconnected():
            return
        subscription = state.event_bus.subscribe(
            session_id, since_seq=cursor,
        ).__aiter__()
        async for ev in subscription:
            cursor = max(cursor, ev.seq)
            yield _encode_event(ev)
            if await request.is_disconnected():
                return


def _last_turn_cutoff(archive: SessionArchive, session_id: str) -> int:
    """Seq immediately before the most recent turn_started, so events
    with seq > cursor include the turn_started event itself."""
    with archive._lock, archive.db.cursor() as cur:
        cur.execute(
            """
            SELECT COALESCE(MAX(seq), 0) FROM events
            WHERE session_id = %s AND event_type = 'turn_started'
            """,
            [session_id],
        )
        row = cur.fetchone()
    last_turn_seq = int(row["coalesce"]) if row else 0
    return max(0, last_turn_seq - 1)


def _encode_event(ev: AgentEvent) -> dict[str, str]:
    frame = {
        "event": ev.type,
        "data": ev.model_dump_json(),
    }
    if ev.seq > 0:
        frame["id"] = str(ev.seq)
    return frame
```

`id: {seq}` on every durable frame is what lets the browser's `EventSource` send `Last-Event-ID` automatically on transient reconnects. Live-only frames intentionally omit `id`, so they cannot move the reconnect cursor past persisted DB rows. `EventSourceResponse` (already used by the route) emits whichever `id`/`event`/`data` keys are present in this dict.

The race-free property comes from `InMemoryEventBus.subscribe` itself: it drains the ring buffer for `seq > since_seq` into the new subscriber's queue under the lock **before** registering, so any concurrent publish either lands in the buffer (caught on drain) or hits the registered queue (delivered live). No extra "subscribe-into-buffer" wrapper is needed.

A separate task should write `: ping\n\n` every ~15s so proxies don't kill idle SSE connections. `EventSourceResponse` from `sse-starlette` does this if configured with `ping=15`.

## Frontend: rebuild rules

The store is fed exclusively by SSE events. There is no localStorage, no per-session client cursor.

The server emits **named SSE events** (`event: assistant_token`, etc.), not generic `message` events. The frontend must register a listener for each event name, matching the existing pattern in `frontend/src/hooks/useSSE.ts`:

```ts
const es = new EventSource(
  `/sessions/${encodeURIComponent(sessionId)}/events`,
  { withCredentials: true },
);

const eventNames = [
  "session_created", "turn_started", "turn_finished", "stream_end", "error",
  "assistant_token", "assistant_snapshot", "assistant_message", "assistant_note",
  "education_disclaimer",
  "skill_auto_loaded", "workflow_plan_updated", "plan_updated",
  "tool_started", "tool_finished", "tool_denied",
  "approval_requested", "approval_resolved",
  "artifact_created", "artifact_patch", "artifact_finalized", "artifact_deleted",
  "sandbox_exec_started", "sandbox_exec_output", "sandbox_exec_finished",
  "usage",
] as const;

eventNames.forEach((name) => {
  es.addEventListener(name, (ev: MessageEvent) => {
    const parsed = JSON.parse(ev.data) as AgentEvent;
    if (parsed.seq > 0) {
      const lastSeq = store.lastSeqBySession[sessionId] ?? 0;
      if (parsed.seq <= lastSeq) return;          // dedup durable replay/live seam
      store.lastSeqBySession[sessionId] = parsed.seq;  // in-memory only
    }
    store.ingest(sessionId, parsed);
  });
});

es.onerror = (e) => console.warn("SSE disconnected, browser will retry", e);
```

`lastSeqBySession` lives in memory only. It exists to dedup the brief seam between replay and live within a single SSE session. It is **not** persisted across page loads, and it only tracks durable events (`seq > 0`). Live-only events such as `assistant_token` have `seq=0` and must always flow through the reducer.

Any new typed event added to the runtime (e.g., `assistant_snapshot`) must be appended to `eventNames` here — listeners registered against unknown names silently drop those events.

Idempotent reducers, keyed by stable ids:

| Event | Behavior |
|---|---|
| `turn_started` | append a new turn entry to `state.turns[]` |
| `assistant_token` | append delta to `state.streamingText[turnId]` |
| `assistant_snapshot` | replace `state.streamingText[turnId]` with content |
| `assistant_message` | finalize: push message, clear streaming text for that turn |
| `tool_started` | upsert tool row by `tool_call_id` |
| `tool_finished` | finalize same `tool_call_id` |
| `plan_updated` | replace plan for the current turn |
| `artifact_created` | upsert artifact metadata by `artifact_id` |
| `artifact_patch` | apply append/replace by `artifact_id` |
| `artifact_finalized` | mark artifact complete |
| `turn_finished` | finalize the assistant message text from `ev.content`; **do not** unlock the composer here — wait for `stream_end` |
| `stream_end` | branch on `final_state`: `completed`/`cancelled`/`failed` → mark turn terminal **and unlock the composer**; `waiting_approval` → keep composer locked, render approval UI |
| `error` | attach error to current turn; the composer remains locked until the matching `stream_end(failed)` arrives |

## Active-run tracking and recovery

For "is the agent still working?" UX and for backend-restart cleanup, the runtime tracks per-session run metadata:

```python
{
    "status": "running" | "completed" | "failed",
    "turn_id": "...",
    "started_at": <epoch>,
    "last_heartbeat_at": <epoch>,
    "last_seq": <int>,
}
```

Two phases, with different durability:

### Phase 1: in-memory dict (no restart recovery)

A Python dict on `ManagedAgentRuntime` keyed by `session_id`. Heartbeated from `_publish`. Visible to a "still working?" indicator while the backend stays alive.

**This dict does not survive backend restart.** Phase 1 explicitly does **not** offer restart recovery — if the backend dies mid-turn, the session is left with no `turn_finished` event, the composer stays locked on next reconnect, and the user must manually start a new turn (or you ship Phase 2).

This is acceptable for development and small deployments. Be honest about it in the UI: on reconnect, if the most recent turn has no terminal event and no live stream is producing, surface "agent stopped unexpectedly — start a new turn."

### Phase 2: durable `active_runs` table (restart recovery)

When you need restart recovery, add a Postgres table:

```sql
CREATE TABLE active_runs (
    session_id        TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
    status            TEXT NOT NULL,           -- running | completed | failed
    turn_id           TEXT,
    started_at        DOUBLE PRECISION NOT NULL,
    last_heartbeat_at DOUBLE PRECISION NOT NULL,
    last_seq          INTEGER NOT NULL,
    worker_id         TEXT                    -- for multi-pod recovery races
);
```

`_publish` writes the heartbeat into this row (UPSERT) instead of the in-memory dict. On backend startup:

```python
def recover_stale_runs(archive: SessionArchive, stale_threshold_sec: float = 60.0) -> None:
    cutoff = time.time() - stale_threshold_sec
    with archive._lock, archive.db.cursor() as cur:
        # SKIP LOCKED so concurrent pods don't both pick up the same row
        cur.execute(
            """
            SELECT session_id, turn_id FROM active_runs
            WHERE status = 'running' AND last_heartbeat_at < %s
            FOR UPDATE SKIP LOCKED
            """,
            [cutoff],
        )
        for row in cur.fetchall():
            archive.emit_event(row["session_id"], "error",
                               {"message": "Backend restarted before turn completed"})
            archive.emit_event(row["session_id"], "stream_end",
                               {"final_state": "failed", "turn_id": row["turn_id"]})
            cur.execute(
                "UPDATE active_runs SET status = 'failed' WHERE session_id = %s",
                [row["session_id"]],
            )
```

Without this Postgres-backed table, restart recovery is not possible — the runtime has no way to know what was running. Don't promise the recovery behavior in Phase 1.

## Tool side effects on resume

Some tools have external side effects mid-turn. `start_app_preview` (`core/src/skill_packs/coding/tools.py:160`) spawns a detached process and writes `.apex_preview.pid`. On any wake/resume path, the runtime must:

1. Scan for known side-effect markers (PID files, opened ports, persisted artifact rows with `kind=app_preview`).
2. Reconcile: if the marker is fresh and the process is alive, reuse it; if stale, clean up.
3. Only then continue from the safe boundary.

Otherwise resume can either leak processes or spawn duplicates.

## UI: right-side turn navigator

A vertical column on the right edge of the chat lets the user see a list of turns and jump between them.

### Behavior

- Lists all turns in the session, oldest at top, newest at bottom.
- Each row shows the user's first line of input (truncated to ~24 chars) and a horizontal indicator bar.
- The active turn (current scroll target or in-progress) is highlighted (accent color, longer bar).
- Click a row → smooth-scroll the chat container to that turn's anchor.
- Hover → preview tooltip with timestamp and full first message.
- Auto-collapses on narrow viewports (< 768px) into an icon button that opens a popover.
- Pinned to the right edge, scrolls independently of the chat.

### Component

`frontend/src/components/chat/TurnNavigator.tsx`:

```tsx
import { useEffect, useState } from "react";
import { useStore } from "../../store";

interface Turn {
  turn_id: string;
  preview: string;        // first ~80 chars of user input
  timestamp: number;
  status: "running" | "completed" | "failed" | "cancelled";
}

export function TurnNavigator({ sessionId }: { sessionId: string }) {
  const turns = useStore((s) => s.turnsBySession[sessionId] ?? []);
  const [activeTurnId, setActiveTurnId] = useState<string | null>(null);

  // Track which turn is in view via IntersectionObserver
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.find((e) => e.isIntersecting);
        if (visible) setActiveTurnId(visible.target.getAttribute("data-turn-id"));
      },
      { rootMargin: "-40% 0px -40% 0px", threshold: 0 },
    );
    document.querySelectorAll("[data-turn-id]").forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, [turns.length]);

  const scrollTo = (turnId: string) => {
    const el = document.querySelector(`[data-turn-id="${turnId}"]`);
    el?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  if (turns.length === 0) return null;

  return (
    <nav className="turn-navigator" aria-label="Turn navigator">
      {turns.map((turn) => {
        const isActive = turn.turn_id === activeTurnId;
        return (
          <button
            key={turn.turn_id}
            className={`turn-row ${isActive ? "is-active" : ""}`}
            onClick={() => scrollTo(turn.turn_id)}
            title={turn.preview}
          >
            <span className="turn-preview">{truncate(turn.preview, 24)}</span>
            <span className={`turn-bar ${isActive ? "is-active" : ""}`} />
          </button>
        );
      })}
    </nav>
  );
}

function truncate(s: string, n: number): string {
  return s.length <= n ? s : s.slice(0, n - 1) + "…";
}
```

### Anchors in the chat

`MessageBubble` / `ChatPane` must mark each turn boundary so the navigator can scroll to it:

```tsx
// inside ChatPane render loop, at the start of each turn:
<div data-turn-id={turn.turn_id} aria-label={`Turn ${idx + 1}`}>
  {turn.userMessage && <MessageBubble role="user" content={turn.userMessage.text} />}
  {turn.activity}
  {turn.assistantMessage && <MessageBubble role="assistant" content={turn.assistantMessage} />}
</div>
```

### Styling

```css
.turn-navigator {
  position: fixed;
  right: 16px;
  top: 96px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 12px 8px;
  background: var(--surface-elevated);
  border-radius: 12px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.32);
  max-height: calc(100vh - 160px);
  overflow-y: auto;
  z-index: 20;
}

.turn-row {
  display: grid;
  grid-template-columns: 1fr auto;
  align-items: center;
  gap: 12px;
  padding: 6px 8px;
  border: 0;
  background: transparent;
  color: var(--text-secondary);
  font-size: 13px;
  cursor: pointer;
  border-radius: 6px;
  transition: color 120ms ease;
}

.turn-row:hover {
  color: var(--text-primary);
  background: var(--surface-hover);
}

.turn-row.is-active {
  color: var(--accent);
}

.turn-bar {
  display: inline-block;
  width: 18px;
  height: 1.5px;
  background: currentColor;
  opacity: 0.4;
  transition: width 160ms ease, opacity 160ms ease;
}

.turn-bar.is-active {
  width: 28px;
  opacity: 1;
}

@media (max-width: 768px) {
  .turn-navigator { display: none; }
  /* show as icon-button popover instead, see below */
}
```

### Mobile fallback

On narrow viewports, replace the column with an icon button in the top-right that opens a sheet/popover containing the same list. Use a single `<dialog>` or a portal-based popover; dismiss on outside click.

### Building the turn list

The store loads turn summaries from the dedicated endpoint on session open, then keeps the list fresh from live events:

```ts
// On session open: fetch the full summary list once.
async function loadTurns(sessionId: string) {
  const res = await fetch(
    `/sessions/${encodeURIComponent(sessionId)}/turns`,
    { credentials: "include" },
  );
  const turns = await res.json();
  store.turnsBySession[sessionId] = turns;     // newest-first from server
}

// reducer for turn_started: append the new turn.
state.turnsBySession[sessionId].unshift({
  turn_id: ev.turn_id,
  user_preview: "",                            // filled from the user message
  started_at: ev.timestamp,
  status: "running",
});

// reducer for the user input that opens a turn:
const turn = state.turnsBySession[sessionId][0];
turn.user_preview = ev.content.slice(0, 80);

// reducer for stream_end: terminal states only.
const turn = state.turnsBySession[sessionId].find((t) => t.turn_id === ev.turn_id);
if (turn) {
  // Treat waiting_approval as still running for navigator purposes.
  turn.status = ev.final_state === "waiting_approval" ? "running" : ev.final_state;
  if (ev.final_state !== "waiting_approval") {
    turn.ended_at = ev.timestamp;
  }
}
```

Why fetch the list rather than derive from events: per-turn SSE replay only delivers the active turn's events. Without the summary endpoint, the navigator would only see the current turn until the user scrolled. The dedicated endpoint is one cheap query and keeps the SSE replay scope honest.

### Status indication

Color the bar by status:

```
running    accent (e.g. blue)
completed  muted gray
cancelled  muted gray + dashed
failed     red
```

The active turn is always full-opacity; inactive turns are 40% opacity.

## Implementation order

Smallest set of changes that delivers the contract. Each step is independently shippable. Each references the actual files in this repo.

1. **Schema**: add `idx_events_session_type_seq` composite index. Existing `events` table already has per-session `seq INTEGER` and `UNIQUE(session_id, seq)` — keep them.
2. **Lock fix in `archive.py`**: move the `_seq_cache` read/increment **inside** the `with self._lock` block (see §"Seq assignment must happen inside the lock"). Required prerequisite for step 3 — without it, concurrent `asyncio.to_thread` callers will hit `UNIQUE(session_id, seq)` violations.
3. **`_publish`** in `core/src/agent/runtime/managed_runtime.py`: add `persist: bool = True` parameter; when `persist=True`, call `archive.emit_event(...)` via `asyncio.to_thread` and stamp the returned seq onto `event.seq`; always `event_bus.publish(...)`. Update the `AssistantToken` emit site at `:572` to pass `persist=False`.
4. **`assistant_snapshot`** typed event in `core/src/agent/events/schema.py` (inheriting from `AgentEventBase`); emit during streaming in `_run_loop` every 500 chars or 1s; frontend handler that **replaces** (not appends) the streaming text.
5. **`/sessions/{id}/events`** in `backend/apex_server/routes/events_routes.py`: when `last_event_id` is absent or 0, compute cursor as `last_turn_seq - 1` instead of 0. Existing replay-then-subscribe logic stays.
6. **`InMemoryEventBus.publish`**: preserve `seq=0` for live-only events; do not assign local seq values that can leak into SSE `id`.
7. **Frontend `useSSE.ts`**: add `assistant_snapshot` to the listener list, dedup durable events (`seq > 0`) against an in-memory `lastSeqBySession` map. No localStorage.
8. **`stream_end` final_state handling on frontend**: only unlock composer when `final_state ∈ {"completed", "cancelled", "failed"}`. Keep locked on `waiting_approval`.
9. **Active-run dict (Phase 1)**: in-memory only, no restart recovery yet. UI surfaces "agent stopped unexpectedly" if reconnect finds a turn with no terminal `stream_end` and no live producer.
10. **`GET /sessions/{id}/turns`**: new endpoint serving the turn-summary list. Single SQL join, newest-first.
11. **Turn navigator UI**: `frontend/src/components/chat/TurnNavigator.tsx`, anchors in `ChatPane`, fetch summaries on session open + update from live events.
12. **`active_runs` table + restart recovery (Phase 2)**: only when you need restart recovery. Adds `recover_stale_runs()` on backend startup.
13. **`GET /sessions/{id}/events/range`** (optional): expand an old turn's events on demand from the navigator.

Steps 1–9 give you correct refresh/reconnect behavior for single-process. Steps 10–11 deliver the navigator. Steps 12–13 are scaling/polish.

## Tests

### Per-turn replay

1. Create session, run two turns.
2. `GET /events` — assert response contains only events from the second `turn_started` onward.
3. Run a third turn, refresh again.
4. Assert response contains only the third turn.

### Refresh during streaming

1. Start a turn with a fake brain that streams tokens slowly.
2. Wait until at least one `assistant_snapshot` is persisted.
3. Disconnect SSE.
4. Reconnect with no cursor.
5. Assert UI rebuilds from `turn_started` through the latest snapshot.
6. Assert subsequent live tokens continue the typewriter.

### Brief disconnect

1. Start SSE.
2. Receive events seq 1..5.
3. Drop the connection.
4. Reconnect with `Last-Event-ID: 5`.
5. Publish events seq 6..7.
6. Assert only seq 6 and 7 arrive (not 1..5 again).

### Browser close, agent keeps running

1. Start a long-running fake turn.
2. Close SSE.
3. Wait — assert the bg task continues and persists `tool_started` / `tool_finished` to DB.
4. Reconnect.
5. Assert replay shows the events that arrived during the disconnect.

### Backend restart recovery (Phase 2)

1. Insert a row in `active_runs` with `status='running'` and a stale `last_heartbeat_at`.
2. Run `recover_stale_runs()` on startup.
3. Assert synthetic `error` + `stream_end(final_state="failed", turn_id=...)` are persisted in the events table for that session, and the `active_runs` row is updated to `status='failed'`.
4. Reconnect — assert UI shows the error message attached to the turn, the navigator marks the turn `failed`, and the composer is unlocked because `final_state="failed"` is a terminal state.

### Tool side-effect reconcile

1. Run a turn that calls `start_app_preview`; PID file is written.
2. Kill the backend before `turn_finished`.
3. Restart; recovery emits the failed turn.
4. Start a new turn that calls `start_app_preview` again.
5. Assert: only one preview process is running (existing one was reused or cleanly replaced — no orphaned process).

### Turn navigator

1. Render a session with five turns.
2. Click the third row in `<TurnNavigator>`.
3. Assert the chat container scrolls so the third turn's anchor is at the top.
4. Scroll past the third turn manually.
5. Assert `is-active` highlight moves to the turn now in view.

## Key principles, in order

```
1. Background task owns the run.    SSE never owns it.
2. DB is the source of truth.       Browser stores nothing across reloads.
3. Replay window is one turn.       Latency bounded by current turn size.
4. seq is assigned at INSERT time.  Concurrent producers cannot disagree.
5. Replay, then subscribe by cursor. No race at the live handoff.
6. Idempotent frontend reducers.    Replay re-delivery is harmless.
7. UI anchors per turn.             Navigator is one click to any turn.
```

If you're tempted to add localStorage, snapshot tables, or a separate cursor service, re-read principle 2.
