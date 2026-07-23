"""Deterministic local app used for browser workflow verification."""

from __future__ import annotations

from typing import Any

import app.main as target
from app.agents import CORRECTION_REVIEWERS, DOC_AGENTS


async def fixture_pipeline(
    text: str,
    progress: Any = None,
    agent_status: Any = None,
    document_type: str = "auto",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total = 4
    if progress:
        await progress(0, total)
    if agent_status:
        await agent_status("grammar", "Grammar reviewer", "running")
        await agent_status("grammar", "Grammar reviewer", "done")
    if progress:
        await progress(2, total)
    start = text.find("teh")
    changes = (
        [
            {
                "original": "teh",
                "corrected": "the",
                "category": "spelling",
                "evidence": "teh",
                "rule": "Use the standard spelling 'the'.",
                "reason": "Misspelling.",
                "suggested_fix": "Replace 'teh' with 'the'.",
                "supporting_context": "teh",
                "severity": "minor",
                "start": start,
                "end": start + 3,
                "verified": True,
                "confidence": 0.99,
            }
        ]
        if start >= 0
        else []
    )
    agents = [
        {
            "key": reviewer["key"],
            "label": reviewer["label"],
            "status": "done",
            "verdict": "Review completed.",
            "findings": 0,
        }
        for reviewer in (*CORRECTION_REVIEWERS, *DOC_AGENTS)
    ]
    findings = [
        {
            "agent": "heading",
            "agent_label": "Heading structure",
            "category": "structure",
            "finding": "Skipped heading level",
            "title": "Skipped heading level",
            "evidence": "Heading 1 followed by Heading 3",
            "rule": "Heading levels must not skip.",
            "confidence": 0.97,
            "severity": "minor",
            "reason": "The hierarchy skips level 2.",
            "detail": "The hierarchy skips level 2.",
            "suggested_fix": "Use Heading 2.",
            "supporting_context": "Advanced setup",
            "location": "Advanced setup",
            "verified": True,
        }
    ]
    scores = {
        key: 98
        for key in (
            "grammar",
            "spelling",
            "punctuation",
            "readability",
            "style",
            "consistency",
            "structure",
            "procedure",
            "logic",
            "iso",
        )
    }
    scores["overall"] = 98
    if progress:
        await progress(total, total)
    return {
        "corrected_text": text.replace("teh", "the"),
        "changes": changes,
        "summary": "One verified correction and one structural finding.",
        "stats": {
            "words": len(text.split()),
            "issues": len(changes),
            "issues_per_100_words": 10,
            "by_category": {
                category: sum(change["category"] == category for change in changes)
                for category in (
                    "grammar",
                    "spelling",
                    "punctuation",
                    "clarity",
                    "readability",
                    "style",
                    "consistency",
                )
            },
            "report": {
                "agents": agents,
                "findings": findings,
                "scores": scores,
                "summary": {
                    "summary": "The document is nearly ready.",
                    "risk_level": "low",
                    "top_issues": ["Correct the heading hierarchy."],
                    "readability": 92,
                },
                "iso": {
                    "present_sections": [],
                    "missing_sections": [],
                    "is_sop": False,
                },
                "document_type": document_type,
                "verification": {
                    "verified_corrections": len(changes),
                    "filtered_corrections": 0,
                    "verified_findings": 1,
                    "filtered_findings": 0,
                },
            },
        },
    }


target.run_pipeline = fixture_pipeline
app = target.app
