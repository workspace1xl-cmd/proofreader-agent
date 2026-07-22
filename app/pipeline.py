"""End-to-end proofreading pipeline.

Splits long documents into chunks on natural boundaries, proofreads chunks
concurrently, anchors every change to a character offset in the original text
(so the UI can render inline markup and apply accept/reject decisions), and
computes document statistics.
"""

from __future__ import annotations

import asyncio
import re

import httpx

from app.cerebras_client import proofread_chunk, review_document

MAX_CHARS = 100_000
CHUNK_TARGET = 6_000
CONCURRENCY = 4

CATEGORIES = ("grammar", "spelling", "punctuation", "clarity", "consistency")

_WORD_RE = re.compile(r"[\w'’-]+")
_SENTENCE_RE = re.compile(r"[.!?]+(?:\s|$)")


def split_spans(text: str) -> list[tuple[int, int]]:
    """Split text into (start, end) spans of ~CHUNK_TARGET chars.

    Spans tile the whole string with no gaps, so global character offsets are
    preserved. Cuts prefer paragraph breaks, then sentence ends, then spaces.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    n = len(text)
    while n - start > CHUNK_TARGET:
        window_end = start + CHUNK_TARGET
        cut = text.rfind("\n\n", start + 1, window_end)
        if cut > start:
            cut += 2
        else:
            m = text.rfind(". ", start + 1, window_end)
            if m > start:
                cut = m + 2
            else:
                sp = text.rfind(" ", start + 1, window_end)
                cut = sp + 1 if sp > start else window_end
        spans.append((start, cut))
        start = cut
    spans.append((start, n))
    return spans


def _normalise_category(value) -> str:
    v = str(value or "").strip().lower()
    return v if v in CATEGORIES else "clarity"


def _normalise_severity(value) -> str:
    return "major" if str(value or "").strip().lower() == "major" else "minor"


def anchor_changes(chunk_text: str, changes: list, offset: int) -> list[dict]:
    """Attach global start/end character offsets to each change.

    Snippets are searched for in document order (a moving cursor), falling back
    to a whole-chunk search. Changes that cannot be located get start/end None
    and are shown in the list but not in the inline markup.
    """
    anchored = []
    cursor = 0
    for ch in changes:
        if not isinstance(ch, dict):
            continue
        original = str(ch.get("original") or "")
        corrected = str(ch.get("corrected") or "")
        if not original or original == corrected:
            continue
        idx = chunk_text.find(original, cursor)
        if idx == -1:
            idx = chunk_text.find(original)
        entry = {
            "original": original,
            "corrected": corrected,
            "category": _normalise_category(ch.get("category") or ch.get("reason")),
            "reason": str(ch.get("reason") or ""),
            "severity": _normalise_severity(ch.get("severity")),
        }
        if idx == -1:
            entry["start"] = None
            entry["end"] = None
        else:
            entry["start"] = offset + idx
            entry["end"] = offset + idx + len(original)
            cursor = idx + len(original)
        anchored.append(entry)
    return anchored


def drop_overlaps(changes: list[dict]) -> list[dict]:
    """Sort anchored changes by position and drop any that overlap a kept one."""
    anchored = sorted(
        (c for c in changes if c["start"] is not None), key=lambda c: c["start"]
    )
    unanchored = [c for c in changes if c["start"] is None]
    kept: list[dict] = []
    prev_end = -1
    for c in anchored:
        if c["start"] >= prev_end:
            kept.append(c)
            prev_end = c["end"]
        else:
            c = {**c, "start": None, "end": None}
            unanchored.append(c)
    return kept + unanchored


def build_stats(text: str, changes: list[dict], chunk_count: int) -> dict:
    words = len(_WORD_RE.findall(text))
    sentences = max(1, len(_SENTENCE_RE.findall(text)))
    by_category = {c: 0 for c in CATEGORIES}
    by_severity = {"minor": 0, "major": 0}
    for ch in changes:
        by_category[ch["category"]] += 1
        by_severity[ch["severity"]] += 1
    issues = len(changes)
    return {
        "words": words,
        "sentences": sentences,
        "chars": len(text),
        "chunks": chunk_count,
        "issues": issues,
        "issues_per_100_words": round(issues * 100 / words, 1) if words else 0.0,
        "by_category": by_category,
        "by_severity": by_severity,
    }


REVIEW_CATEGORIES = ("terminology", "structure", "logic", "consistency")


def _normalise_review(review: dict) -> dict:
    findings = []
    for f in review.get("findings") or []:
        if not isinstance(f, dict):
            continue
        cat = str(f.get("category") or "").strip().lower()
        findings.append(
            {
                "title": str(f.get("title") or "Finding").strip(),
                "detail": str(f.get("detail") or "").strip(),
                "category": cat if cat in REVIEW_CATEGORIES else "consistency",
                "severity": _normalise_severity(f.get("severity")),
            }
        )
    return {
        "findings": findings,
        "verdict": str(review.get("verdict") or "").strip(),
        "truncated": bool(review.get("truncated")),
        "failed": bool(review.get("failed")),
    }


async def run_pipeline(text: str, progress=None) -> dict:
    """Proofread a full document. `progress(done, total)` is awaited per unit of
    work: one unit per chunk plus one for the whole-document structural review."""
    spans = split_spans(text)
    total = len(spans) + 1  # +1 for the document-level review pass
    results: list[dict | None] = [None] * len(spans)
    review: dict = {}
    done = 0
    sem = asyncio.Semaphore(CONCURRENCY)

    if progress:
        await progress(0, total)

    async with httpx.AsyncClient(timeout=90) as client:

        async def work(i: int):
            nonlocal done
            s, e = spans[i]
            async with sem:
                res = await proofread_chunk(client, text[s:e])
            results[i] = res
            done += 1
            if progress:
                await progress(done, total)

        async def doc_review():
            nonlocal done, review
            review = await review_document(client, text)
            done += 1
            if progress:
                await progress(done, total)

        await asyncio.gather(*(work(i) for i in range(len(spans))), doc_review())

    all_changes: list[dict] = []
    corrected_parts: list[str] = []
    for i, (s, e) in enumerate(spans):
        chunk_text = text[s:e]
        res = results[i] or {}
        corrected_parts.append(str(res.get("corrected_text") or chunk_text))
        all_changes.extend(anchor_changes(chunk_text, res.get("changes") or [], s))

    changes = drop_overlaps(all_changes)
    n_chunks = len(spans)
    stats = build_stats(text, changes, n_chunks)
    stats["review"] = _normalise_review(review)

    if n_chunks == 1 and (results[0] or {}).get("summary"):
        summary = str(results[0]["summary"]).strip()
    elif stats["issues"] == 0:
        summary = f"No issues found across {stats['words']:,} words."
    else:
        summary = (
            f"{stats['issues']:,} suggested correction"
            f"{'s' if stats['issues'] != 1 else ''} "
            f"across {stats['words']:,} words in {n_chunks} section"
            f"{'s' if n_chunks != 1 else ''}."
        )

    return {
        "corrected_text": "".join(corrected_parts),
        "changes": changes,
        "summary": summary,
        "stats": stats,
    }
