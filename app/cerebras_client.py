"""Resilient structured-output client for Cerebras inference."""

from __future__ import annotations

import asyncio
import email.utils
import json
import logging
import os
import random
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import httpx

from app.config import MODEL_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")
CEREBRAS_URL = os.environ.get(
    "CEREBRAS_URL", "https://api.cerebras.ai/v1/chat/completions"
)
MODEL = os.environ.get("CEREBRAS_MODEL", "gpt-oss-120b")

SYSTEM_PROMPT = """You are a precision proofreading engine. Correct only genuine
grammar, spelling, punctuation, clarity, and internal-consistency errors without
changing meaning, tone, formatting, names, defined terms, or optional style.

Return ONLY this JSON object:
{
  "corrected_text": "<the full TARGET text with listed corrections applied>",
  "changes": [{
    "original": "<smallest exact verbatim snippet from TARGET>",
    "corrected": "<exact replacement>",
    "category": "<grammar|spelling|punctuation|clarity|readability|style|consistency>",
    "evidence": "<exact source evidence>",
    "rule": "<specific language or consistency rule>",
    "reason": "<why the evidence violates the rule>",
    "suggested_fix": "<the exact replacement and why it is minimal>",
    "supporting_context": "<short surrounding context>",
    "severity": "<minor|major>"
  }],
  "summary": "<one sentence>"
}

Rules:
- Review only TARGET. CONTEXT is read-only and must never appear in corrected_text.
- Every original must occur verbatim in TARGET and every listed replacement must
  be reflected in corrected_text.
- Preserve paragraph breaks, markdown syntax, list markers, tables, code, URLs,
  email addresses, identifiers, product names, headings, and Unicode characters.
- Apply format rules only for the declared DOCUMENT TYPE. Never apply Markdown
  heading rules to DOCX, TXT, HTML, or PDF text.
- Do not alter text inside fenced/inline code, URLs, email addresses, or file paths
  unless it is unmistakably malformed.
- Do not rename roles, titles, departments, products, headings, or defined terms.
- Do not impose Oxford commas, heading punctuation, title capitalization, or other
  optional preferences unless the document is internally inconsistent.
- Prefer no change over a debatable change. Never fabricate a correction.
- If clean, return TARGET unchanged with an empty changes array.
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
Sleep = Callable[[float], Awaitable[None]]


def _parse_response(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    candidates = [raw]
    match = _JSON_RE.search(raw)
    if match and match.group(0) != raw:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return max(0.0, (parsed - datetime.now(UTC)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def _backoff(attempt: int, retry_after: str | None = None) -> float:
    requested = _retry_after_seconds(retry_after)
    exponential = min(2.0 * (2**attempt), 24.0)
    base = max(exponential, requested or 0.0)
    return min(30.0, base + random.uniform(0.0, min(1.0, base * 0.15)))


async def call_json(
    client: httpx.AsyncClient,
    system: str,
    user: str,
    max_tokens: int = 4096,
    temperature: float = 0.1,
    attempts: int = 4,
    *,
    sleep: Sleep = asyncio.sleep,
) -> dict[str, Any] | None:
    """Call the model and return a JSON object.

    Retry transient HTTP failures, timeouts, invalid response envelopes, and
    malformed model JSON. Cancellation always propagates immediately.
    """

    if not CEREBRAS_API_KEY:
        raise RuntimeError(
            "CEREBRAS_API_KEY is not set. Create a key at cloud.cerebras.ai."
        )
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {CEREBRAS_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(max(1, attempts)):
        retry_after: str | None = None
        try:
            response = await client.post(
                CEREBRAS_URL,
                headers=headers,
                json=payload,
                timeout=MODEL_TIMEOUT_SECONDS,
            )
            retry_after = response.headers.get("retry-after")
            if response.status_code in _RETRYABLE_STATUS:
                raise httpx.HTTPStatusError(
                    "retryable model response",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()
            envelope = response.json()
            if not isinstance(envelope, dict):
                raise ValueError("model response envelope is not an object")
            choices = envelope.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError("model response has no choices")
            message = (
                choices[0].get("message") if isinstance(choices[0], dict) else None
            )
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str):
                raise ValueError("model response has no text content")
            parsed = _parse_response(content)
            if parsed is not None:
                return parsed
            raise ValueError("model returned malformed JSON")
        except asyncio.CancelledError:
            raise
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in _RETRYABLE_STATUS:
                logger.warning(
                    "Non-retryable model HTTP status %s", exc.response.status_code
                )
                return None
            logger.warning(
                "Transient model HTTP status %s (attempt %s/%s)",
                exc.response.status_code,
                attempt + 1,
                attempts,
            )
        except (
            httpx.TimeoutException,
            httpx.TransportError,
            ValueError,
            KeyError,
        ) as exc:
            logger.warning(
                "Model call failed (attempt %s/%s): %s",
                attempt + 1,
                attempts,
                type(exc).__name__,
            )
        if attempt + 1 < max(1, attempts):
            await sleep(_backoff(attempt, retry_after))
    return None


async def proofread_chunk(
    client: httpx.AsyncClient,
    text: str,
    *,
    context_before: str = "",
    context_after: str = "",
    document_type: str = "txt",
    attempts: int = 4,
) -> dict[str, Any]:
    """Proofread a target chunk with read-only boundary context."""

    user = (
        f"DOCUMENT TYPE: {document_type}\n\n"
        f"CONTEXT BEFORE (read-only):\n{context_before}\n\n"
        f"TARGET:\n{text}\n\n"
        f"CONTEXT AFTER (read-only):\n{context_after}"
    )
    parsed = await call_json(
        client,
        SYSTEM_PROMPT,
        user,
        max_tokens=8192,
        attempts=attempts,
    )
    if parsed is None:
        return {
            "corrected_text": text,
            "changes": [],
            "summary": "This section could not be analysed.",
            "failed": True,
        }
    changes = parsed.get("changes")
    return {
        "corrected_text": (
            parsed.get("corrected_text")
            if isinstance(parsed.get("corrected_text"), str)
            else text
        ),
        "changes": changes if isinstance(changes, list) else [],
        "summary": str(parsed.get("summary") or ""),
        "failed": False,
    }
