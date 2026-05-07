"""Session-scoped dataset upload routes."""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel

from agent.session.store import SessionPatch
from apex_server.deps import AppState, User, get_state, require_user
from apex_server.routes.session_support import owned_session


router = APIRouter(prefix="/sessions", tags=["sessions"])

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_ALLOWED_EXTENSIONS = {".csv", ".tsv", ".json", ".jsonl", ".txt"}


class UploadOut(BaseModel):
    id: str
    filename: str
    path: str
    size: int
    content_type: str


@router.post("/{session_id}/uploads", response_model=UploadOut, status_code=status.HTTP_201_CREATED)
async def upload_file(
    session_id: str,
    request: Request,
    filename: str = Query(min_length=1, max_length=160),
    user: User = Depends(require_user),
    state: AppState = Depends(get_state),
) -> UploadOut:
    await owned_session(session_id, user, state)
    safe_name = _safe_filename(filename)
    ext = Path(safe_name).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(_ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type. Allowed: {allowed}",
        )

    body = await request.body()
    if not body:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")
    if len(body) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File is larger than 10 MB",
        )

    upload_id = uuid.uuid4().hex
    upload_dir = _upload_root() / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{int(time.time())}-{upload_id[:8]}-{safe_name}"
    path = upload_dir / stored_name
    path.write_bytes(body)

    rel_path = path.relative_to(_REPO_ROOT).as_posix()
    record = {
        "id": upload_id,
        "filename": safe_name,
        "path": rel_path,
        "absolute_path": str(path),
        "size": len(body),
        "content_type": request.headers.get("content-type") or "application/octet-stream",
        "uploaded_at": time.time(),
    }
    await _record_upload(state, session_id, record)
    return UploadOut(
        id=upload_id,
        filename=safe_name,
        path=rel_path,
        size=len(body),
        content_type=record["content_type"],
    )


def _upload_root() -> Path:
    return _REPO_ROOT / "results" / "uploads"


def _safe_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    if not name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename")
    return name[:160]


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


async def _record_upload(state: AppState, session_id: str, record: dict[str, Any]) -> None:
    session = await state.session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    metadata = dict(session.metadata)
    uploads = [*_metadata_uploads(metadata), record]
    metadata["uploaded_files"] = uploads
    session_meta = metadata.get("session_metadata")
    if isinstance(session_meta, dict):
        session_meta = dict(session_meta)
        session_meta["uploaded_files"] = uploads
        metadata["session_metadata"] = session_meta
    await state.session_store.update(session_id, SessionPatch(metadata=metadata))

    runner = state.runners.get(session_id)
    if runner is not None:
        runner.runtime.session.metadata["uploaded_files"] = uploads


__all__ = ["router"]
