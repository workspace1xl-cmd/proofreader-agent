"""Independent specialist, verification, and summary model calls."""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.cerebras_client import call_json

FINDING_SCHEMA = """Return ONLY a JSON object:
{"findings": [{"title": "<short issue>", "detail": "<specific evidence and
recommendation>", "location": "<verbatim quote or section>", "severity":
"<minor|major>"}], "verdict": "<one sentence>"}
Only report demonstrable issues. Empty findings is valid. Never invent issues."""

DOC_AGENTS: tuple[dict[str, str], ...] = (
    {
        "key": "terminology",
        "label": "Terminology consistency",
        "category": "terminology",
        "prompt": """Review the complete document for inconsistent roles, departments,
systems, products, abbreviations, and defined terms. Flag only variants that clearly
refer to the same entity, state the evidence, and prefer the dominant form. Flag
undefined abbreviations only when genuinely ambiguous. Consistent terminology is
correct and must not be renamed.\n\n"""
        + FINDING_SCHEMA,
    },
    {
        "key": "structure",
        "label": "Structure & formatting",
        "category": "structure",
        "prompt": """Review heading hierarchy, numbering, list conventions, duplicated
headings/fragments, visible spacing, markdown syntax, and table column/header
consistency across the complete document. Do not report sentence-level grammar,
spelling, or optional style preferences.\n\n"""
        + FINDING_SCHEMA,
    },
    {
        "key": "procedure",
        "label": "Procedure validation",
        "category": "procedure",
        "prompt": """For an actual procedure/SOP, compare every process representation:
steps, tables, responsibilities, checklists, and references. Find missing, duplicated,
misordered, contradictory, or wrongly owned steps and broken references. If the
document is not procedural, return no findings.\n\n"""
        + FINDING_SCHEMA,
    },
    {
        "key": "logic",
        "label": "Flowchart & decision logic",
        "category": "logic",
        "prompt": """For an actual workflow, verify that decisions have all required
branches, paths terminate, referenced steps exist, and there are no unreachable,
duplicated, or endless paths. Do not infer invisible diagram contents. If no workflow
is described, return no findings.\n\n"""
        + FINDING_SCHEMA,
    },
    {
        "key": "iso",
        "label": "ISO/QMS compliance",
        "category": "iso",
        "prompt": """Only when the document is clearly controlled SOP/QMS documentation,
assess Purpose, Scope, Definitions, Responsibilities, Procedure, Records, References,
Revision History, ownership/approval/effective date, and risk/escalation handling.
For ordinary prose, do not demand ISO sections.

Return ONLY a JSON object:
{"findings": [{"title":"...", "detail":"...", "location":"...",
"severity":"minor|major"}], "present_sections": [], "missing_sections": [],
"is_sop": true, "verdict": "<one sentence>"}""",
    },
)

VERIFIER_PROMPT = """You are the final false-positive verifier. Each proposed item
contains its own source context. Keep only a demonstrable and useful error.

Reject optional style, speculative claims, duplicates, changes to valid names/code/
URLs/markdown, claims unsupported by context, or anything that changes meaning.
Keep genuine grammar, spelling, internal inconsistency, structure, procedure, logic,
and controlled-document defects.

Return ONLY:
{"items":[{"id":"<exact id>","keep":true,"confidence":0.0,
"note":"<brief reason when rejected>"}]}
Return every supplied id exactly once."""

SUMMARY_PROMPT = """Write a concise executive release-readiness summary using only the
provided deterministic scores and verified issues. Risk must be low for clean/minor
documents, medium when material fixes are needed, and high only for serious release
blocking defects.

Return ONLY:
{"summary":"<2-4 factual sentences>","risk_level":"low|medium|high",
"top_issues":["<up to 5>"],"readability":<0-100 integer>}"""


async def run_doc_agent(
    client: httpx.AsyncClient, agent: dict[str, str], text: str
) -> dict[str, Any]:
    parsed = await call_json(client, agent["prompt"], text, max_tokens=4096)
    if parsed is None:
        return {"findings": [], "verdict": "", "failed": True}
    findings = parsed.get("findings")
    parsed["findings"] = findings if isinstance(findings, list) else []
    parsed["verdict"] = str(parsed.get("verdict") or "")
    parsed["failed"] = False
    return parsed


async def run_verifier(
    client: httpx.AsyncClient,
    items: list[dict[str, str]],
    *,
    batch_size: int = 40,
) -> dict[str, dict[str, Any]]:
    """Verify every item in bounded batches; missing rows stay explicitly unverified."""

    output: dict[str, dict[str, Any]] = {}
    for start in range(0, len(items), max(1, batch_size)):
        batch = items[start : start + max(1, batch_size)]
        listing = "\n\n".join(
            f"{item['id']}. [{item['kind']}] {item['text']}\n"
            f"SOURCE CONTEXT: {item['context']}"
            for item in batch
        )
        parsed = await call_json(
            client,
            VERIFIER_PROMPT,
            listing,
            max_tokens=max(2048, min(6144, len(batch) * 100)),
        )
        rows = parsed.get("items") if parsed else None
        if not isinstance(rows, list):
            continue
        expected = {item["id"] for item in batch}
        for row in rows:
            if not isinstance(row, dict) or str(row.get("id")) not in expected:
                continue
            item_id = str(row["id"])
            try:
                confidence = max(0.0, min(1.0, float(row.get("confidence", 0.5))))
            except (TypeError, ValueError):
                confidence = 0.5
            keep_value = row.get("keep")
            keep = keep_value if isinstance(keep_value, bool) else None
            output[item_id] = {
                "keep": keep,
                "confidence": confidence,
                "note": str(row.get("note") or ""),
            }
    return output


async def run_summary(
    client: httpx.AsyncClient,
    stats: dict[str, Any],
    scores: dict[str, int],
    top_findings: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "stats": {
            key: value
            for key, value in stats.items()
            if key not in {"review", "report"}
        },
        "scores": scores,
        "top_findings": [
            {
                "title": finding.get("title"),
                "severity": finding.get("severity"),
                "agent": finding.get("agent"),
            }
            for finding in top_findings[:12]
        ],
    }
    parsed = await call_json(
        client, SUMMARY_PROMPT, json.dumps(payload), max_tokens=1024
    )
    if parsed is None:
        return {
            "summary": "",
            "risk_level": _fallback_risk(scores),
            "top_issues": [],
            "readability": None,
            "failed": True,
        }
    risk = str(parsed.get("risk_level") or "").lower()
    if risk not in {"low", "medium", "high"}:
        risk = _fallback_risk(scores)
    try:
        raw_readability = parsed.get("readability")
        readability = (
            None if raw_readability is None else max(0, min(100, int(raw_readability)))
        )
    except (TypeError, ValueError):
        readability = None
    top_issues = parsed.get("top_issues")
    return {
        "summary": str(parsed.get("summary") or "").strip(),
        "risk_level": risk,
        "top_issues": (
            [str(item) for item in top_issues[:5]]
            if isinstance(top_issues, list)
            else []
        ),
        "readability": readability,
        "failed": False,
    }


def _fallback_risk(scores: dict[str, int]) -> str:
    overall = scores.get("overall", 0)
    return "low" if overall >= 90 else "medium" if overall >= 70 else "high"
