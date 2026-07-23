"""Small bounded job history with optional Supabase persistence."""

from __future__ import annotations

import logging
import os
import threading
import uuid
from collections import deque
from datetime import UTC, datetime
from typing import Any

from app.config import HISTORY_LIMIT

logger = logging.getLogger(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
_client: Any | None = None
if SUPABASE_URL and SUPABASE_KEY and "your-project" not in SUPABASE_URL:
    try:
        from supabase import create_client

        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        logger.exception("Supabase initialization failed; using memory history")

_memory: deque[dict[str, Any]] = deque(maxlen=HISTORY_LIMIT)
_memory_lock = threading.RLock()


def storage_mode() -> str:
    return "supabase" if _client else "memory"


def _save_memory(row: dict[str, Any]) -> dict[str, Any]:
    saved = {
        **row,
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
    }
    with _memory_lock:
        _memory.appendleft(saved)
    return saved


def save_job(original_text: str, result: dict[str, Any]) -> dict[str, Any]:
    row = {
        "original_text": original_text,
        "corrected_text": result.get("corrected_text", ""),
        "changes": result.get("changes", []),
        "summary": result.get("summary", ""),
        "stats": result.get("stats", {}),
    }
    if _client:
        try:
            response = _client.table("proofread_jobs").insert(row).execute()
            if response.data:
                return dict(response.data[0])
        except Exception:
            logger.exception("Supabase save failed; retaining job in memory")
    return _save_memory(row)


def get_job(job_id: str) -> dict[str, Any] | None:
    if _client:
        try:
            response = (
                _client.table("proofread_jobs").select("*").eq("id", job_id).execute()
            )
            if response.data:
                return dict(response.data[0])
        except Exception:
            logger.exception("Supabase read failed; checking memory history")
    with _memory_lock:
        return next((dict(job) for job in _memory if job["id"] == job_id), None)


def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
    if _client:
        try:
            response = (
                _client.table("proofread_jobs")
                .select("id, summary, stats, created_at, original_text")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return [dict(row) for row in response.data]
        except Exception:
            logger.exception("Supabase list failed; returning memory history")
    with _memory_lock:
        jobs = list(_memory)[:limit]
    return [
        {
            "id": job["id"],
            "summary": job.get("summary", ""),
            "stats": job.get("stats", {}),
            "created_at": job.get("created_at", ""),
            "original_text": str(job.get("original_text", ""))[:1_000],
        }
        for job in jobs
    ]


def clear_memory_for_tests() -> None:
    with _memory_lock:
        _memory.clear()
