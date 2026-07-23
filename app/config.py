"""Application-wide limits and tunables.

Environment variables make the Render deployment adjustable without code changes,
while conservative defaults keep local and test runs predictable.
"""

from __future__ import annotations

import os


def _positive_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


MAX_CHARS = _positive_int("MAX_CHARS", 100_000)
MAX_UPLOAD_BYTES = _positive_int("MAX_UPLOAD_BYTES", 2_000_000)
MAX_DOCX_UNCOMPRESSED_BYTES = _positive_int("MAX_DOCX_UNCOMPRESSED_BYTES", 20_000_000)
MAX_DOCX_ENTRIES = _positive_int("MAX_DOCX_ENTRIES", 5_000)
CHUNK_TARGET = _positive_int("CHUNK_TARGET", 6_000, minimum=1_000)
CHUNK_CONTEXT = _positive_int("CHUNK_CONTEXT", 500, minimum=0)
PIPELINE_CONCURRENCY = _positive_int("PIPELINE_CONCURRENCY", 4)
PIPELINE_TIMEOUT_SECONDS = _positive_int("PIPELINE_TIMEOUT_SECONDS", 420, minimum=30)
MODEL_TIMEOUT_SECONDS = _positive_int("MODEL_TIMEOUT_SECONDS", 120, minimum=10)
SSE_HEARTBEAT_SECONDS = _positive_int("SSE_HEARTBEAT_SECONDS", 12, minimum=3)
HISTORY_LIMIT = _positive_int("HISTORY_LIMIT", 50)
