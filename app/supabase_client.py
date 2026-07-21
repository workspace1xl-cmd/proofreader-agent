"""Job storage. Uses Supabase when configured, otherwise an in-memory store
so the app works end to end locally with just a Cerebras key."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

_client = None
if SUPABASE_URL and SUPABASE_KEY and "your-project" not in SUPABASE_URL:
    try:
        from supabase import create_client

        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        _client = None

_memory: list[dict] = []
_MEMORY_LIMIT = 200


def storage_mode() -> str:
    return "supabase" if _client else "memory"


def _save_memory(row: dict) -> dict:
    row = {
        **row,
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _memory.insert(0, row)
    del _memory[_MEMORY_LIMIT:]
    return row


def save_job(original_text: str, result: dict) -> dict:
    row = {
        "original_text": original_text,
        "corrected_text": result.get("corrected_text", ""),
        "changes": result.get("changes", []),
        "summary": result.get("summary", ""),
        "stats": result.get("stats", {}),
    }
    if _client:
        try:
            res = _client.table("proofread_jobs").insert(row).execute()
            if res.data:
                return res.data[0]
        except Exception:
            # Older schema without the stats column — retry without it.
            try:
                slim = {k: v for k, v in row.items() if k != "stats"}
                res = _client.table("proofread_jobs").insert(slim).execute()
                if res.data:
                    return {**res.data[0], "stats": row["stats"]}
            except Exception:
                pass
    return _save_memory(row)


def get_job(job_id: str) -> dict | None:
    if _client:
        try:
            res = (
                _client.table("proofread_jobs").select("*").eq("id", job_id).execute()
            )
            if res.data:
                return res.data[0]
        except Exception:
            pass
    return next((j for j in _memory if j["id"] == job_id), None)


def list_jobs(limit: int = 20) -> list:
    if _client:
        try:
            res = (
                _client.table("proofread_jobs")
                .select("id, summary, stats, created_at, original_text")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return res.data
        except Exception:
            pass
    return [
        {
            "id": j["id"],
            "summary": j.get("summary", ""),
            "stats": j.get("stats", {}),
            "created_at": j.get("created_at", ""),
            "original_text": j.get("original_text", ""),
        }
        for j in _memory[:limit]
    ]
