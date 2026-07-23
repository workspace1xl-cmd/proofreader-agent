"""Concurrent, deterministic document-review orchestration."""

from __future__ import annotations

import asyncio
import logging
import re
import weakref
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from app.agents import (
    CORRECTION_REVIEWERS,
    DOC_AGENTS,
    run_doc_agent,
    run_summary,
    run_verifier,
)
from app.cerebras_client import proofread_chunk
from app.config import (
    CHUNK_CONTEXT,
    CHUNK_TARGET,
    MODEL_TIMEOUT_SECONDS,
    PIPELINE_CONCURRENCY,
)
from app.document_intelligence import (
    detect_document_type,
    is_protected_change,
    is_reviewer_applicable,
    run_rule_reviewer,
)

logger = logging.getLogger(__name__)

CATEGORIES = (
    "grammar",
    "spelling",
    "punctuation",
    "clarity",
    "readability",
    "style",
    "consistency",
)
_WORD_RE = re.compile(r"[\w'\u2019-]+", re.UNICODE)
_SENTENCE_RE = re.compile(r"[.!?]+(?:\s|$)")
_SEV_PENALTY = {"minor": 2, "major": 6}
_SCORE_WEIGHTS = {
    "grammar": 0.15,
    "spelling": 0.08,
    "punctuation": 0.08,
    "readability": 0.08,
    "style": 0.06,
    "consistency": 0.12,
    "structure": 0.10,
    "procedure": 0.13,
    "logic": 0.10,
    "iso": 0.10,
}
_MODEL_SEMAPHORES: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, asyncio.Semaphore
] = weakref.WeakKeyDictionary()

ProgressCallback = Callable[[int, int], Awaitable[None]]
StatusCallback = Callable[[str, str, str], Awaitable[None]]


def _model_semaphore() -> asyncio.Semaphore:
    """Share the provider concurrency budget across requests on one event loop."""

    loop = asyncio.get_running_loop()
    semaphore = _MODEL_SEMAPHORES.get(loop)
    if semaphore is None:
        semaphore = asyncio.Semaphore(PIPELINE_CONCURRENCY)
        _MODEL_SEMAPHORES[loop] = semaphore
    return semaphore


def _verify_rule_finding(finding: dict[str, Any]) -> bool:
    """Independently validate the complete contract of a rule-engine finding."""

    required = (
        "title",
        "evidence",
        "rule",
        "reason",
        "suggested_fix",
        "supporting_context",
    )
    return (
        finding.get("verification_basis") == "deterministic_rule"
        and finding.get("severity") in {"minor", "major"}
        and all(str(finding.get(field) or "").strip() for field in required)
    )


def split_spans(text: str) -> list[tuple[int, int]]:
    """Tile text with natural-boundary chunks and no lost characters."""

    if not text:
        return [(0, 0)]
    spans: list[tuple[int, int]] = []
    start = 0
    while len(text) - start > CHUNK_TARGET:
        window_end = start + CHUNK_TARGET
        minimum = start + max(1, CHUNK_TARGET // 2)
        candidates = (
            text.rfind("\n\n", minimum, window_end),
            text.rfind(". ", minimum, window_end),
            text.rfind("! ", minimum, window_end),
            text.rfind("? ", minimum, window_end),
            text.rfind("\n", minimum, window_end),
            text.rfind(" ", minimum, window_end),
        )
        cut = max(candidates)
        if cut < minimum:
            cut = window_end
        elif text[cut : cut + 2] in {"\n\n", ". ", "! ", "? "}:
            cut += 2
        else:
            cut += 1
        spans.append((start, cut))
        start = cut
    spans.append((start, len(text)))
    return spans


def _normalise_category(value: Any) -> str:
    category = str(value or "").strip().lower()
    return category if category in CATEGORIES else "clarity"


def _normalise_severity(value: Any) -> str:
    return "major" if str(value or "").strip().lower() == "major" else "minor"


def _find_flexible(text: str, snippet: str, start: int) -> tuple[int, int] | None:
    """Locate a snippet despite harmless whitespace drift in model output."""

    parts = re.split(r"(\s+)", snippet)
    pattern = "".join(r"\s+" if part.isspace() else re.escape(part) for part in parts)
    try:
        match = re.search(pattern, text[start:])
        if match is None and start:
            match = re.search(pattern, text)
            base = 0
        else:
            base = start
    except re.error:
        return None
    return None if match is None else (base + match.start(), base + match.end())


def anchor_changes(
    chunk_text: str, changes: list[Any], offset: int
) -> list[dict[str, Any]]:
    """Normalize model changes and attach exact offsets into the original document."""

    anchored: list[dict[str, Any]] = []
    cursor = 0
    for raw in changes:
        if not isinstance(raw, dict):
            continue
        original = str(raw.get("original") or "")
        corrected = str(raw.get("corrected") or "")
        if not original or original == corrected:
            continue
        start = chunk_text.find(original, cursor)
        end = start + len(original) if start >= 0 else -1
        if start < 0:
            start = chunk_text.find(original)
            end = start + len(original) if start >= 0 else -1
        if start < 0 and any(char.isspace() for char in original):
            flexible = _find_flexible(chunk_text, original, cursor)
            if flexible is not None:
                start, end = flexible
                original = chunk_text[start:end]
        entry: dict[str, Any] = {
            "original": original,
            "corrected": corrected,
            "category": _normalise_category(raw.get("category")),
            "reason": str(raw.get("reason") or "").strip(),
            "evidence": str(raw.get("evidence") or original).strip(),
            "rule": str(raw.get("rule") or "Language correctness").strip(),
            "suggested_fix": str(
                raw.get("suggested_fix") or f"Replace with {corrected!r}."
            ).strip(),
            "supporting_context": str(raw.get("supporting_context") or "").strip(),
            "severity": _normalise_severity(raw.get("severity")),
            "start": None,
            "end": None,
            "verified": None,
            "confidence": None,
        }
        if start >= 0:
            entry["start"] = offset + start
            entry["end"] = offset + end
            cursor = end
        anchored.append(entry)
    return anchored


def dedupe_changes(changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    output: list[dict[str, Any]] = []
    for change in changes:
        key = (
            change.get("start"),
            change.get("end"),
            change["original"].casefold(),
            change["corrected"].casefold(),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(change)
    return output


def resolve_overlaps(changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the strongest verified correction from each overlapping group."""

    anchored = sorted(
        (change for change in changes if change.get("start") is not None),
        key=lambda change: (int(change["start"]), int(change["end"])),
    )
    unanchored = [change for change in changes if change.get("start") is None]
    groups: list[list[dict[str, Any]]] = []
    for change in anchored:
        if not groups or int(change["start"]) >= max(
            int(item["end"]) for item in groups[-1]
        ):
            groups.append([change])
        else:
            groups[-1].append(change)

    def rank(change: dict[str, Any]) -> tuple[int, float, int, int]:
        verification = 2 if change.get("verified") is True else 1
        confidence = float(change.get("confidence") or 0.0)
        severity = 1 if change["severity"] == "major" else 0
        span = int(change["end"]) - int(change["start"])
        return verification, confidence, severity, span

    return [max(group, key=rank) for group in groups] + unanchored


def _norm_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def dedupe_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove strong duplicates using normalized title plus location."""

    seen: set[tuple[str, str]] = set()
    output: list[dict[str, Any]] = []
    for finding in findings:
        title = _norm_title(str(finding.get("title") or ""))
        location = _norm_title(str(finding.get("location") or ""))
        key = (title, location)
        if title and key in seen:
            continue
        seen.add(key)
        output.append(finding)
    return output


def build_stats(
    text: str, changes: list[dict[str, Any]], chunk_count: int
) -> dict[str, Any]:
    words = len(_WORD_RE.findall(text))
    sentences = max(1, len(_SENTENCE_RE.findall(text)))
    by_category = {category: 0 for category in CATEGORIES}
    by_severity = {"minor": 0, "major": 0}
    for change in changes:
        by_category[change["category"]] += 1
        by_severity[change["severity"]] += 1
    issues = len(changes)
    return {
        "words": words,
        "sentences": sentences,
        "chars": len(text),
        "chunks": chunk_count,
        "issues": issues,
        "verified_issues": sum(change.get("verified") is True for change in changes),
        "unverified_issues": sum(change.get("verified") is None for change in changes),
        "issues_per_100_words": round(issues * 100 / words, 1) if words else 0.0,
        "by_category": by_category,
        "by_severity": by_severity,
    }


def compute_scores(
    changes: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    words: int,
) -> dict[str, int]:
    words = max(1, words)

    def correction_score(categories: tuple[str, ...]) -> int:
        penalty = sum(
            _SEV_PENALTY[change["severity"]]
            for change in changes
            if change["category"] in categories and change.get("verified") is True
        )
        scaled = penalty * 1.5
        if words > 200:
            scaled = min(scaled, penalty * 800 / words)
        return max(0, round(100 - scaled))

    def finding_score(agents: tuple[str, ...]) -> int:
        penalty = sum(
            _SEV_PENALTY[finding["severity"]]
            for finding in findings
            if finding["agent"] in agents and finding.get("verified") is True
        )
        return max(0, round(100 - penalty * 1.5))

    scores = {
        "grammar": correction_score(("grammar", "clarity")),
        "spelling": correction_score(("spelling",)),
        "punctuation": correction_score(("punctuation",)),
        "readability": correction_score(("readability",)),
        "style": correction_score(("style",)),
        "consistency": min(
            correction_score(("consistency",)), finding_score(("terminology",))
        ),
        "structure": finding_score(("structure",)),
        "procedure": finding_score(("procedure",)),
        "logic": finding_score(("logic",)),
        "iso": finding_score(("iso",)),
    }
    scores["overall"] = round(
        sum(scores[key] * weight for key, weight in _SCORE_WEIGHTS.items())
    )
    return scores


def _context_for_span(text: str, start: int | None, end: int | None) -> str:
    if start is None or end is None:
        return ""
    left = max(0, start - 220)
    right = min(len(text), end + 220)
    return text[left:right].replace("\x00", "")


def _context_for_finding(text: str, finding: dict[str, Any]) -> str:
    location = str(finding.get("location") or "").strip()
    if location:
        index = text.find(location)
        if index >= 0:
            return _context_for_span(text, index, index + len(location))
    if len(text) <= 900:
        return text
    return f"{text[:450]}\n…\n{text[-450:]}"


def apply_changes(text: str, changes: list[dict[str, Any]]) -> str:
    """Apply only anchored, verifier-approved, non-overlapping changes."""

    approved = sorted(
        (
            change
            for change in changes
            if change.get("verified") is True and change.get("start") is not None
        ),
        key=lambda change: int(change["start"]),
    )
    output: list[str] = []
    cursor = 0
    for change in approved:
        start = int(change["start"])
        end = int(change["end"])
        if start < cursor or text[start:end] != change["original"]:
            continue
        output.extend((text[cursor:start], change["corrected"]))
        cursor = end
    output.append(text[cursor:])
    return "".join(output)


async def run_pipeline(
    text: str,
    progress: ProgressCallback | None = None,
    agent_status: StatusCallback | None = None,
    document_type: str = "auto",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_type = detect_document_type(text, document_type)
    document_metadata = metadata if isinstance(metadata, dict) else {}
    spans = split_spans(text)
    total = len(spans) + len(DOC_AGENTS) + 2
    completed = 0
    semaphore = _model_semaphore()
    chunk_results: list[dict[str, Any] | None] = [None] * len(spans)
    agent_results: dict[str, dict[str, Any]] = {}

    async def emit_status(key: str, label: str, state: str) -> None:
        if agent_status:
            await agent_status(key, label, state)

    async def tick() -> None:
        nonlocal completed
        completed += 1
        if progress:
            await progress(completed, total)

    if progress:
        await progress(0, total)

    limits = httpx.Limits(
        max_connections=PIPELINE_CONCURRENCY,
        max_keepalive_connections=PIPELINE_CONCURRENCY,
    )
    timeout = httpx.Timeout(MODEL_TIMEOUT_SECONDS)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        await emit_status("corrections", "Corrections", "running")

        async def chunk_work(index: int) -> None:
            start, end = spans[index]
            before = text[max(0, start - CHUNK_CONTEXT) : start]
            after = text[end : min(len(text), end + CHUNK_CONTEXT)]
            try:
                async with semaphore:
                    result = await proofread_chunk(
                        client,
                        text[start:end],
                        context_before=before,
                        context_after=after,
                        document_type=resolved_type,
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Correction chunk %s failed", index)
                result = {"changes": [], "failed": True}
            chunk_results[index] = result
            await tick()

        async def doc_work(agent: dict[str, str]) -> None:
            await emit_status(agent["key"], agent["label"], "running")
            try:
                if agent.get("engine") == "rules":
                    result = await asyncio.to_thread(
                        run_rule_reviewer,
                        agent["key"],
                        text,
                        resolved_type,
                        document_metadata,
                    )
                elif not is_reviewer_applicable(agent["key"], text):
                    result = {
                        "findings": [],
                        "verdict": "Not applicable to this document.",
                        "failed": False,
                    }
                else:
                    rule_result = (
                        await asyncio.to_thread(
                            run_rule_reviewer,
                            agent["key"],
                            text,
                            resolved_type,
                            document_metadata,
                        )
                        if agent.get("engine") == "hybrid"
                        else None
                    )
                    async with semaphore:
                        review_input = (
                            f"DOCUMENT TYPE: {resolved_type}\n"
                            f"FORMAT METADATA: {document_metadata}\n\n"
                            f"DOCUMENT:\n{text}"
                        )
                        result = await run_doc_agent(client, agent, review_input)
                    if rule_result:
                        result["findings"] = [
                            *(rule_result.get("findings") or []),
                            *(result.get("findings") or []),
                        ]
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Document agent %s failed", agent["key"])
                result = {"findings": [], "verdict": "", "failed": True}
            agent_results[agent["key"]] = result
            await emit_status(
                agent["key"],
                agent["label"],
                "failed" if result.get("failed") else "done",
            )
            await tick()

        await asyncio.gather(
            *(chunk_work(index) for index in range(len(spans))),
            *(doc_work(agent) for agent in DOC_AGENTS),
        )
        correction_failed = any(
            result is None or result.get("failed") for result in chunk_results
        )
        await emit_status(
            "corrections", "Corrections", "failed" if correction_failed else "done"
        )

        changes: list[dict[str, Any]] = []
        for index, (start, end) in enumerate(spans):
            result = chunk_results[index] or {}
            changes.extend(
                anchor_changes(text[start:end], result.get("changes") or [], start)
            )
        changes = dedupe_changes(changes)
        eligible_changes: list[dict[str, Any]] = []
        prefiltered_changes = 0
        for change in changes:
            rejection = is_protected_change(
                text,
                change.get("start"),
                change.get("end"),
                change["original"],
                change["corrected"],
                resolved_type,
            )
            if rejection:
                prefiltered_changes += 1
                continue
            eligible_changes.append(change)
        changes = eligible_changes

        findings: list[dict[str, Any]] = []
        for agent in DOC_AGENTS:
            result = agent_results.get(agent["key"]) or {}
            rows = result.get("findings")
            if not isinstance(rows, list):
                continue
            for raw in rows:
                if not isinstance(raw, dict):
                    continue
                name = str(raw.get("finding") or raw.get("title") or "Finding").strip()
                evidence = str(raw.get("evidence") or raw.get("location") or "").strip()
                reason = str(raw.get("reason") or raw.get("detail") or "").strip()
                context = str(
                    raw.get("supporting_context") or raw.get("location") or evidence
                ).strip()
                findings.append(
                    {
                        "agent": agent["key"],
                        "agent_label": agent["label"],
                        "category": agent["category"],
                        "verification_basis": str(
                            raw.get("verification_basis") or "model"
                        ),
                        "finding": name,
                        "title": name,
                        "evidence": evidence,
                        "rule": str(raw.get("rule") or "").strip(),
                        "confidence": None,
                        "severity": _normalise_severity(raw.get("severity")),
                        "reason": reason,
                        "detail": reason,
                        "suggested_fix": str(raw.get("suggested_fix") or "").strip(),
                        "supporting_context": context,
                        "location": context,
                        "verified": None,
                    }
                )
        findings = dedupe_findings(findings)

        await emit_status("verifier", "False-positive verification", "running")
        verifier_items: list[dict[str, str]] = []
        for index, change in enumerate(changes):
            verifier_items.append(
                {
                    "id": f"c{index}",
                    "kind": change["category"],
                    "text": (
                        f'Change "{change["original"]}" to "{change["corrected"]}". '
                        f"RULE: {change['rule']}. REASON: {change['reason']}"
                    ),
                    "context": _context_for_span(
                        text, change.get("start"), change.get("end")
                    ),
                }
            )
        for index, finding in enumerate(findings):
            if _verify_rule_finding(finding):
                finding["verified"] = True
                finding["confidence"] = 0.98
                finding["verifier_note"] = (
                    "Validated independently against the deterministic rule contract."
                )
                continue
            verifier_items.append(
                {
                    "id": f"f{index}",
                    "kind": finding["agent"],
                    "text": (
                        f"{finding['title']}. EVIDENCE: {finding['evidence']}. "
                        f"RULE: {finding['rule']}. REASON: {finding['reason']}. "
                        f"FIX: {finding['suggested_fix']}"
                    ),
                    "context": _context_for_finding(text, finding),
                }
            )
        try:
            verdicts = await run_verifier(client, verifier_items)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Verification stage failed")
            verdicts = {}
        for prefix, rows in (("c", changes), ("f", findings)):
            for index, row in enumerate(rows):
                verdict = verdicts.get(f"{prefix}{index}")
                if verdict is None:
                    continue
                row["verified"] = verdict.get("keep")
                row["confidence"] = verdict.get("confidence")
                threshold = (
                    0.85
                    if row.get("category")
                    in {"style", "readability", "terminology", "consistency"}
                    else 0.75
                )
                if (
                    row["verified"] is not True
                    or float(row["confidence"] or 0.0) < threshold
                ):
                    row["verified"] = False
                if verdict.get("note"):
                    row["verifier_note"] = verdict["note"]
        changes = resolve_overlaps(changes)
        verified_changes = [
            change for change in changes if change.get("verified") is True
        ]
        verified_findings_all = [
            finding for finding in findings if finding.get("verified") is True
        ]
        filtered_changes = len(changes) - len(verified_changes) + prefiltered_changes
        filtered_findings = len(findings) - len(verified_findings_all)
        await emit_status(
            "verifier",
            "False-positive verification",
            "done" if len(verdicts) == len(verifier_items) else "failed",
        )
        await tick()

        stats = build_stats(text, verified_changes, len(spans))
        scores = compute_scores(verified_changes, verified_findings_all, stats["words"])
        await emit_status("summary", "Executive summary", "running")
        verified_findings = sorted(
            verified_findings_all,
            key=lambda finding: finding["severity"] != "major",
        )
        try:
            executive_summary = await run_summary(
                client, stats, scores, verified_findings
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Executive-summary stage failed")
            executive_summary = {
                "summary": "",
                "risk_level": "medium",
                "top_issues": [],
                "readability": None,
                "failed": True,
            }
        await emit_status(
            "summary",
            "Executive summary",
            "failed" if executive_summary.get("failed") else "done",
        )
        await tick()

    iso_result = agent_results.get("iso") or {}
    report = {
        "agents": [
            {
                "key": reviewer["key"],
                "label": reviewer["label"],
                "status": "failed" if correction_failed else "done",
                "verdict": (
                    "Review completed."
                    if not correction_failed
                    else "One or more chunks failed."
                ),
                "findings": sum(
                    change["category"] == reviewer["category"]
                    for change in verified_changes
                ),
            }
            for reviewer in CORRECTION_REVIEWERS
        ]
        + [
            {
                "key": agent["key"],
                "label": agent["label"],
                "status": (
                    "failed"
                    if (agent_results.get(agent["key"]) or {}).get("failed")
                    else "done"
                ),
                "verdict": str(
                    (agent_results.get(agent["key"]) or {}).get("verdict") or ""
                ),
                "findings": sum(
                    finding["agent"] == agent["key"]
                    for finding in verified_findings_all
                ),
            }
            for agent in DOC_AGENTS
        ],
        "findings": verified_findings_all,
        "scores": scores,
        "summary": executive_summary,
        "document_type": resolved_type,
        "iso": {
            "present_sections": [
                str(section) for section in iso_result.get("present_sections") or []
            ],
            "missing_sections": [
                str(section) for section in iso_result.get("missing_sections") or []
            ],
            "is_sop": bool(iso_result.get("is_sop")),
        },
        "verification": {
            "verified_corrections": len(verified_changes),
            "filtered_corrections": filtered_changes,
            "verified_findings": len(verified_findings_all),
            "filtered_findings": filtered_findings,
        },
    }
    stats["report"] = report
    verified_count = stats["verified_issues"]
    verified_findings_count = len(verified_findings_all)
    if not verified_count and not verified_findings_count:
        summary = f"No verified issues found across {stats['words']:,} words."
    else:
        summary = (
            f"{verified_count:,} verified correction"
            f"{'s' if verified_count != 1 else ''} and "
            f"{verified_findings_count} verified document-level finding"
            f"{'s' if verified_findings_count != 1 else ''} across "
            f"{stats['words']:,} words. Overall score: {scores['overall']}/100."
        )
    return {
        "corrected_text": apply_changes(text, verified_changes),
        "changes": verified_changes,
        "summary": summary,
        "stats": stats,
    }
