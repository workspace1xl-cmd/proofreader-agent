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

from app.agents import (
    DOC_AGENTS,
    run_doc_agent,
    run_summary,
    run_verifier,
)
from app.cerebras_client import proofread_chunk, review_document  # noqa: F401 (review_document kept for API compat)

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


def _norm_title(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def dedupe_findings(findings: list[dict]) -> list[dict]:
    """Drop findings whose normalised title duplicates an earlier one (across
    agents too — two agents reporting the same defect is one finding)."""
    seen: set[str] = set()
    out = []
    for f in findings:
        key = _norm_title(f.get("title", ""))
        if key and key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


# Deterministic scoring: each area starts at 100 and loses points per verified
# issue (minor=2, major=6). Inline-correction areas are scaled by density
# (issues per 100 words) so long documents are not punished for length.
_SEV_PENALTY = {"minor": 2, "major": 6}
_SCORE_WEIGHTS = {
    "grammar": 0.20, "spelling": 0.10, "punctuation": 0.10, "consistency": 0.15,
    "structure": 0.10, "procedure": 0.15, "logic": 0.10, "iso": 0.10,
}


def compute_scores(changes: list[dict], findings: list[dict], words: int) -> dict:
    words = max(1, words)

    def change_area_score(cats: tuple) -> int:
        pen = sum(
            _SEV_PENALTY[c["severity"]]
            for c in changes
            if c["category"] in cats and c.get("verified", True)
        )
        density_pen = pen * 100 / words * 8  # scale by document length
        return max(0, round(100 - min(pen * 1.5, density_pen) if words > 200 else 100 - pen * 1.5))

    def finding_area_score(agent_keys: tuple) -> int:
        pen = sum(
            _SEV_PENALTY[f["severity"]]
            for f in findings
            if f["agent"] in agent_keys and f.get("verified", True)
        )
        return max(0, round(100 - pen * 1.5))

    scores = {
        "grammar": change_area_score(("grammar", "clarity")),
        "spelling": change_area_score(("spelling",)),
        "punctuation": change_area_score(("punctuation",)),
        "consistency": min(
            change_area_score(("consistency",)), finding_area_score(("terminology",))
        ),
        "structure": finding_area_score(("structure",)),
        "procedure": finding_area_score(("procedure",)),
        "logic": finding_area_score(("logic",)),
        "iso": finding_area_score(("iso",)),
    }
    overall = sum(scores[k] * w for k, w in _SCORE_WEIGHTS.items())
    scores["overall"] = round(overall)
    return scores


async def run_pipeline(text: str, progress=None, agent_status=None) -> dict:
    """Multi-agent document review.

    Units of work for `progress(done, total)`: one per chunk (corrections),
    one per document-level agent, one for verification, one for the summary.
    `agent_status(key, label, state)` is awaited on agent transitions
    (state: "running" | "done" | "failed").
    """
    spans = split_spans(text)
    n_chunks = len(spans)
    total = n_chunks + len(DOC_AGENTS) + 2  # + verifier + summary
    results: list[dict | None] = [None] * n_chunks
    agent_results: dict[str, dict] = {}
    done = 0
    sem = asyncio.Semaphore(CONCURRENCY)

    async def status(key: str, label: str, state: str):
        if agent_status:
            await agent_status(key, label, state)

    async def tick():
        nonlocal done
        done += 1
        if progress:
            await progress(done, total)

    if progress:
        await progress(0, total)

    async with httpx.AsyncClient(timeout=120) as client:
        # ---- Stage A: corrections per chunk + doc-level agents, all parallel
        await status("corrections", "Corrections (grammar, spelling, punctuation)", "running")
        chunks_left = n_chunks

        async def work(i: int):
            nonlocal chunks_left
            s, e = spans[i]
            async with sem:
                res = await proofread_chunk(client, text[s:e])
            results[i] = res
            chunks_left -= 1
            if chunks_left == 0:
                await status("corrections", "Corrections", "done")
            await tick()

        async def doc_work(agent: dict):
            await status(agent["key"], agent["label"], "running")
            async with sem:
                res = await run_doc_agent(client, agent, text[:30000])
            agent_results[agent["key"]] = res
            await status(
                agent["key"], agent["label"],
                "failed" if res.get("failed") else "done",
            )
            await tick()

        await asyncio.gather(
            *(work(i) for i in range(n_chunks)),
            *(doc_work(a) for a in DOC_AGENTS),
        )

        # ---- Stage B: merge + anchor + dedupe
        all_changes: list[dict] = []
        corrected_parts: list[str] = []
        for i, (s, e) in enumerate(spans):
            chunk_text = text[s:e]
            res = results[i] or {}
            corrected_parts.append(str(res.get("corrected_text") or chunk_text))
            all_changes.extend(anchor_changes(chunk_text, res.get("changes") or [], s))
        changes = drop_overlaps(all_changes)

        findings: list[dict] = []
        for agent in DOC_AGENTS:
            res = agent_results.get(agent["key"]) or {}
            for f in res.get("findings") or []:
                if not isinstance(f, dict):
                    continue
                findings.append(
                    {
                        "agent": agent["key"],
                        "agent_label": agent["label"],
                        "category": agent["category"],
                        "title": str(f.get("title") or "Finding").strip(),
                        "detail": str(f.get("detail") or "").strip(),
                        "location": str(f.get("location") or "").strip(),
                        "severity": _normalise_severity(f.get("severity")),
                        "verified": True,
                        "confidence": None,
                    }
                )
        findings = dedupe_findings(findings)

        # ---- Stage C: false-positive verification over everything
        await status("verifier", "False-positive verification", "running")
        items = []
        for idx, ch in enumerate(changes[:150]):
            items.append(
                {
                    "id": f"c{idx}",
                    "kind": ch["category"],
                    "text": f'change "{ch["original"]}" to "{ch["corrected"]}" — {ch["reason"]}',
                }
            )
        for idx, f in enumerate(findings[:60]):
            items.append(
                {"id": f"f{idx}", "kind": f["agent"], "text": f'{f["title"]}: {f["detail"][:180]}'}
            )
        try:
            verdicts = await run_verifier(client, text, items)
        except RuntimeError:
            raise
        except Exception:
            verdicts = {}
        for idx, ch in enumerate(changes):
            v = verdicts.get(f"c{idx}")
            if v:
                ch["verified"] = v["keep"]
                ch["confidence"] = v["confidence"]
                if not v["keep"] and v["note"]:
                    ch["verifier_note"] = v["note"]
            else:
                ch.setdefault("verified", True)
                ch.setdefault("confidence", None)
        for idx, f in enumerate(findings):
            v = verdicts.get(f"f{idx}")
            if v:
                f["verified"] = v["keep"]
                f["confidence"] = v["confidence"]
                if not v["keep"] and v["note"]:
                    f["verifier_note"] = v["note"]
        await status("verifier", "False-positive verification", "done")
        await tick()

        # ---- Stage D: scores + executive summary
        stats = build_stats(text, changes, n_chunks)
        scores = compute_scores(changes, findings, stats["words"])
        await status("summary", "Executive summary", "running")
        top = sorted(
            [f for f in findings if f["verified"]],
            key=lambda f: (f["severity"] != "major",),
        )
        try:
            exec_summary = await run_summary(client, stats, scores, top)
        except RuntimeError:
            raise
        except Exception:
            exec_summary = {"summary": "", "risk_level": "medium", "top_issues": [], "readability": None}
        await status("summary", "Executive summary", "done")
        await tick()

    iso_res = agent_results.get("iso") or {}
    report = {
        "agents": [
            {
                "key": a["key"],
                "label": a["label"],
                "status": "failed" if (agent_results.get(a["key"]) or {}).get("failed") else "done",
                "verdict": str((agent_results.get(a["key"]) or {}).get("verdict") or ""),
                "findings": sum(1 for f in findings if f["agent"] == a["key"]),
            }
            for a in DOC_AGENTS
        ],
        "findings": findings,
        "scores": scores,
        "summary": exec_summary,
        "iso": {
            "present_sections": [str(s) for s in iso_res.get("present_sections") or []],
            "missing_sections": [str(s) for s in iso_res.get("missing_sections") or []],
            "is_sop": bool(iso_res.get("is_sop")),
        },
    }
    stats["report"] = report

    verified_findings = sum(1 for f in findings if f["verified"])
    if stats["issues"] == 0 and verified_findings == 0:
        summary = f"No issues found across {stats['words']:,} words."
    else:
        summary = (
            f"{stats['issues']:,} suggested correction"
            f"{'s' if stats['issues'] != 1 else ''} and "
            f"{verified_findings} document-level finding"
            f"{'s' if verified_findings != 1 else ''} "
            f"across {stats['words']:,} words. Overall score: {scores['overall']}/100."
        )

    return {
        "corrected_text": "".join(corrected_parts),
        "changes": changes,
        "summary": summary,
        "stats": stats,
    }
