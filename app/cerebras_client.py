from __future__ import annotations

import os
import json
import re
import asyncio

import httpx

CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
MODEL = os.environ.get("CEREBRAS_MODEL", "llama-3.3-70b")

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


async def proofread_chunk(client: httpx.AsyncClient, text: str, attempts: int = 3) -> dict:
    """Proofread one chunk of text. Returns the parsed model response.

    Retries on transient HTTP errors and on malformed JSON. Falls back to a
    no-op result (input returned unchanged) if every attempt fails to parse,
    so one bad chunk never sinks a whole document.
    """
    if not CEREBRAS_API_KEY:
        raise RuntimeError(
            "CEREBRAS_API_KEY is not set. Create a key at cloud.cerebras.ai and export it."
        )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0.1,
        "max_tokens": 8192,
    }
    headers = {
        "Authorization": f"Bearer {CEREBRAS_API_KEY}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = await client.post(CEREBRAS_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"]
            parsed = _parse_response(raw)
            if parsed is not None and "corrected_text" in parsed:
                parsed.setdefault("changes", [])
                parsed.setdefault("summary", "")
                return parsed
            last_error = ValueError("model returned non-JSON output")
        except (httpx.HTTPError, KeyError, IndexError) as e:
            last_error = e
        if attempt < attempts - 1:
            await asyncio.sleep(1.5 * (attempt + 1))

    # Every attempt failed — degrade gracefully rather than erroring the job.
    return {
        "corrected_text": text,
        "changes": [],
        "summary": f"This section could not be analysed ({last_error}).",
    }
