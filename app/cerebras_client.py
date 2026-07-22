from __future__ import annotations

import os
import json
import re
import asyncio

import httpx

CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
MODEL = os.environ.get("CEREBRAS_MODEL", "gpt-oss-120b")

SYSTEM_PROMPT = """You are a precision proofreading engine. You correct grammar, spelling, \
punctuation, clarity, and consistency issues WITHOUT changing the author's meaning, tone, or voice. \
You never rewrite for style unless it is a genuine error. You never add opinions or content.

Return ONLY valid JSON matching this exact schema, nothing else, no markdown fences:
{
  "corrected_text": "<the full corrected text>",
  "changes": [
    {
      "original": "<exact original snippet, copied verbatim from the input>",
      "corrected": "<exact replacement snippet>",
      "category": "<grammar|spelling|punctuation|clarity|consistency>",
      "reason": "<short plain-language explanation of the fix>",
      "severity": "<minor|major>"
    }
  ],
  "summary": "<one sentence summary of overall quality>"
}

Rules:
- "original" must be copied character-for-character from the input so it can be located programmatically. Keep snippets short — the smallest span that contains the error plus a word of context if needed.
- If there are no errors, return corrected_text identical to input and an empty changes array.
- Never fabricate changes. Only list changes you actually made.
- Preserve original paragraph breaks and formatting exactly.
- List changes in the order they appear in the text.
- NEVER suggest renaming roles, job titles, department names, product names, section headings, or defined terms (e.g. do not shorten "Graphic Design Manager" to "Design Manager"). If the text uses two variants of the same name, align the minority variant to the dominant one and categorise it as "consistency".
- Do NOT impose optional style preferences. The serial (Oxford) comma, colons after headings, capitalisation style in titles, and similar choices are only errors if the text is internally inconsistent about them.
- Do NOT add punctuation to headings, labels, table cells, or list items to match your own style.
- Prioritise genuine grammar, agreement, and word-choice errors over marginal punctuation tweaks.
- Prefer fewer, high-confidence corrections over many debatable ones. If a change is debatable, omit it.
"""

DOC_REVIEW_PROMPT = """You are a senior documentation reviewer (ISO 9001 / QMS style). You receive a \
full document that has already been through mechanical proofreading — do NOT list spelling, grammar, \
or punctuation fixes. Instead, review the document as a whole for:

1. Terminology and role consistency — the same role, department, or defined term should be named \
identically everywhere. Flag variants and state which form dominates.
2. Heading and formatting consistency — numbering, colon usage, capitalisation patterns across \
sections should match.
3. Procedural logic — steps should be in a workable order; every decision point (e.g. "Approved?") \
must have all branches (a Yes path needs a No path); no dead ends; referenced sections, tables, or \
attachments must exist in the document.
4. Cross-section consistency — if a process is described both as prose steps and in a table, \
checklist, or flowchart, they must describe the same process, roles, and order.

Return ONLY valid JSON, nothing else, no markdown fences:
{
  "findings": [
    {
      "title": "<short name of the issue>",
      "detail": "<what is wrong and where, in plain language, with a concrete recommendation>",
      "category": "<terminology|structure|logic|consistency>",
      "severity": "<minor|major>"
    }
  ],
  "verdict": "<one sentence on overall document quality from a structural point of view>"
}

Only report real findings you can point to in the text. An empty findings array is a valid answer \
for a clean document. Never invent issues to seem thorough.
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_response(raw: str) -> dict | None:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON_RE.search(raw)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


async def call_json(
    client: httpx.AsyncClient,
    system: str,
    user: str,
    max_tokens: int = 4096,
    temperature: float = 0.1,
    attempts: int = 4,
) -> dict | None:
    """One structured-JSON model call with retries and 429/503 backoff.
    Returns None if every attempt fails — callers degrade gracefully."""
    if not CEREBRAS_API_KEY:
        raise RuntimeError(
            "CEREBRAS_API_KEY is not set. Create a key at cloud.cerebras.ai and export it."
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
    for attempt in range(attempts):
        try:
            resp = await client.post(CEREBRAS_URL, headers=headers, json=payload)
            if resp.status_code in (429, 503):
                # Rate-limited or at capacity: honour Retry-After, capped.
                try:
                    delay = float(resp.headers.get("retry-after") or 0)
                except ValueError:
                    delay = 0
                await asyncio.sleep(min(max(delay, 2.5 * (attempt + 1)), 25))
                continue
            resp.raise_for_status()
            data = resp.json()
            parsed = _parse_response(data["choices"][0]["message"]["content"])
            if parsed is not None:
                return parsed
        except (httpx.HTTPError, KeyError, IndexError):
            pass
        if attempt < attempts - 1:
            await asyncio.sleep(2.0 * (attempt + 1))
    return None


async def proofread_chunk(client: httpx.AsyncClient, text: str, attempts: int = 4) -> dict:
    """Proofread one chunk of text. Falls back to a no-op result (input
    returned unchanged) if every attempt fails, so one bad chunk never sinks
    a whole document."""
    parsed = await call_json(
        client, SYSTEM_PROMPT, text, max_tokens=8192, attempts=attempts
    )
    if parsed is not None and "corrected_text" in parsed:
        parsed.setdefault("changes", [])
        parsed.setdefault("summary", "")
        return parsed
    return {
        "corrected_text": text,
        "changes": [],
        "summary": "This section could not be analysed (model unavailable).",
    }


DOC_REVIEW_MAX_CHARS = 30_000


async def review_document(client: httpx.AsyncClient, text: str, attempts: int = 3) -> dict:
    """Legacy single-call structural review, kept for API compatibility.
    Degrades to an empty review on failure — never sinks the job."""
    truncated = len(text) > DOC_REVIEW_MAX_CHARS
    doc = text[:DOC_REVIEW_MAX_CHARS]
    parsed = await call_json(
        client, DOC_REVIEW_PROMPT, doc, max_tokens=4096, temperature=0.2, attempts=attempts
    )
    if parsed is not None and isinstance(parsed.get("findings"), list):
        parsed.setdefault("verdict", "")
        parsed["truncated"] = truncated
        return parsed
    return {"findings": [], "verdict": "", "truncated": truncated, "failed": True}
