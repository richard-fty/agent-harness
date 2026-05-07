"""Turn kickoff and approval resolution routes."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from apex_server.deps import AppState, User, get_state, require_user
from apex_server.runner import get_or_build_runner
from apex_server.routes.session_support import (
    ApprovalIn,
    TurnIn,
    owned_session,
)


router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("/{session_id}/turns")
async def list_turns(
    session_id: str,
    before: int | None = Query(default=None, ge=1),
    limit: int = Query(default=100, ge=1, le=200),
    user: User = Depends(require_user),
    state: AppState = Depends(get_state),
) -> list[dict[str, Any]]:
    await owned_session(session_id, user, state)

    def _query() -> list[dict[str, Any]]:
        params: list[Any] = [session_id]
        before_clause = ""
        if before is not None:
            before_clause = "AND t.seq < %s"
            params.append(before)
        params.append(limit)
        with state.archive._lock, state.archive.db.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                  COALESCE(t.payload ->> 'turn_id', '') AS turn_id,
                  t.timestamp AS started_at,
                  t.seq AS started_seq,
                  e.timestamp AS ended_at,
                  e.seq AS ended_seq,
                  COALESCE(e.payload ->> 'final_state', 'running') AS status,
                  LEFT(COALESCE(t.payload ->> 'user_input', ''), 120) AS user_preview,
                  LEFT(COALESCE(a.payload ->> 'content', f.payload ->> 'content', ''), 160) AS assistant_preview
                FROM events t
                LEFT JOIN LATERAL (
                  SELECT timestamp, seq, payload
                  FROM events
                  WHERE session_id = t.session_id
                    AND event_type = 'stream_end'
                    AND payload ->> 'turn_id' = t.payload ->> 'turn_id'
                  ORDER BY seq DESC
                  LIMIT 1
                ) e ON TRUE
                LEFT JOIN LATERAL (
                  SELECT payload
                  FROM events
                  WHERE session_id = t.session_id
                    AND event_type = 'assistant_message'
                    AND payload ->> 'turn_id' = t.payload ->> 'turn_id'
                  ORDER BY seq DESC
                  LIMIT 1
                ) a ON TRUE
                LEFT JOIN LATERAL (
                  SELECT payload
                  FROM events
                  WHERE session_id = t.session_id
                    AND event_type = 'turn_finished'
                    AND payload ->> 'turn_id' = t.payload ->> 'turn_id'
                  ORDER BY seq DESC
                  LIMIT 1
                ) f ON TRUE
                WHERE t.session_id = %s
                  AND t.event_type = 'turn_started'
                  {before_clause}
                ORDER BY t.seq DESC
                LIMIT %s
                """,
                params,
            )
            rows = cur.fetchall()
        return [
            {
                "turn_id": row["turn_id"] or f"seq-{row['started_seq']}",
                "started_at": row["started_at"],
                "started_seq": row["started_seq"],
                "ended_at": row["ended_at"],
                "ended_seq": row["ended_seq"],
                "status": row["status"],
                "user_preview": row["user_preview"] or "",
                "assistant_preview": row["assistant_preview"] or "",
            }
            for row in rows
        ]

    return await asyncio.to_thread(_query)


@router.post("/{session_id}/turns", status_code=202)
async def post_turn(
    session_id: str,
    payload: TurnIn,
    user: User = Depends(require_user),
    state: AppState = Depends(get_state),
) -> dict[str, str]:
    sess = await owned_session(session_id, user, state)
    stored = await state.session_store.get(session_id)
    uploads = _metadata_uploads(stored.metadata if stored is not None else {})
    effective_input = _with_upload_context(payload.user_input, uploads)
    runner = get_or_build_runner(state, session_id, sess.model)
    assert runner.session_id == session_id
    runner.start_turn_background(effective_input, display_user_input=payload.user_input)
    return {"status": "accepted", "session_id": session_id}


@router.post("/{session_id}/approvals", status_code=202)
async def post_approval(
    session_id: str,
    payload: ApprovalIn,
    user: User = Depends(require_user),
    state: AppState = Depends(get_state),
) -> dict[str, str]:
    await owned_session(session_id, user, state)
    runner = state.runners.get(session_id)
    if runner is None:
        raise HTTPException(status_code=409, detail="No active session runner")
    runner.resume_pending_background(payload.action)
    return {"status": "accepted"}


def _metadata_uploads(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    raw = metadata.get("uploaded_files")
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    session_meta = metadata.get("session_metadata")
    if isinstance(session_meta, dict):
        nested = session_meta.get("uploaded_files")
        if isinstance(nested, list):
            return [item for item in nested if isinstance(item, dict)]
    return []


def _with_upload_context(user_input: str, uploads: list[dict[str, Any]]) -> str:
    if not uploads:
        return user_input
    lines = [
        "",
        "[Uploaded files available to inspect]",
        "Use read_file with these paths before deriving metrics or building visualizations:",
    ]
    for item in uploads[-8:]:
        filename = str(item.get("filename") or "uploaded file")
        path = str(item.get("absolute_path") or item.get("path") or "")
        rel_path = str(item.get("path") or "")
        size = item.get("size")
        detail = f"{filename}: {path}"
        if rel_path and rel_path != path:
            detail += f" (repo path: {rel_path})"
        if isinstance(size, int):
            detail += f" ({size} bytes)"
        lines.append(f"- {detail}")
    return user_input.rstrip() + "\n" + "\n".join(lines)
