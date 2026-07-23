"""Independent specialist, verification, and summary model calls."""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.cerebras_client import call_json

FINDING_SCHEMA = """Return ONLY a JSON object:
{"findings": [{
  "finding": "<short issue name>",
  "evidence": "<verbatim evidence from the document>",
  "rule": "<specific rule or consistency principle>",
  "severity": "<minor|major>",
  "reason": "<why the evidence violates the rule>",
  "suggested_fix": "<precise recommendation, never an automatic rewrite>",
  "supporting_context": "<section name or short source excerpt>"
}], "verdict": "<one factual sentence>"}
Every field is mandatory. Report only issues directly supported by source evidence.
Empty findings is valid and preferred to speculation. Never invent an issue."""

CORRECTION_REVIEWERS: tuple[dict[str, str], ...] = (
    {"key": "grammar", "label": "Grammar reviewer", "category": "grammar"},
    {"key": "spelling", "label": "Spelling reviewer", "category": "spelling"},
    {
        "key": "punctuation",
        "label": "Punctuation reviewer",
        "category": "punctuation",
    },
    {
        "key": "readability",
        "label": "Readability reviewer",
        "category": "readability",
    },
    {"key": "style", "label": "Style reviewer", "category": "style"},
)

DOC_AGENTS: tuple[dict[str, str], ...] = (
    {
        "key": "terminology",
        "label": "Terminology consistency",
        "category": "terminology",
        "engine": "rules",
        "prompt": """Review the complete document for inconsistent roles, departments,
systems, products, abbreviations, and defined terms. Flag only variants that clearly
refer to the same entity, state the evidence, and prefer the dominant form. Flag
undefined abbreviations only when genuinely ambiguous. Consistent terminology is
correct and must not be renamed.\n\n"""
        + FINDING_SCHEMA,
    },
    {
        "key": "role",
        "label": "Role consistency",
        "category": "consistency",
        "engine": "llm",
        "prompt": """Compare role names and assigned responsibilities across the
complete document. Report only direct contradictions or a minority role-name variant
supported by quotes. Do not assume similar-sounding roles are the same role.\n\n"""
        + FINDING_SCHEMA,
    },
    {
        "key": "section",
        "label": "Section consistency",
        "category": "structure",
        "engine": "llm",
        "prompt": """Compare summaries, scope, responsibilities, procedures, records,
forms, appendices, and revision statements across sections. Report only a direct,
quoted contradiction or an explicitly promised but absent section.\n\n"""
        + FINDING_SCHEMA,
    },
    {
        "key": "procedure",
        "label": "Procedure consistency",
        "category": "procedure",
        "engine": "hybrid",
        "prompt": """For an actual procedure/SOP, compare every process representation:
steps, tables, responsibilities, checklists, and references. Find missing, duplicated,
misordered, contradictory, or wrongly owned steps and broken references. If the
document is not procedural, return no findings.\n\n"""
        + FINDING_SCHEMA,
    },
    {
        "key": "cross_reference",
        "label": "Cross-reference reviewer",
        "category": "structure",
        "engine": "rules",
        "prompt": FINDING_SCHEMA,
    },
    {
        "key": "heading",
        "label": "Heading structure",
        "category": "structure",
        "engine": "rules",
        "prompt": FINDING_SCHEMA,
    },
    {
        "key": "numbering",
        "label": "Numbering reviewer",
        "category": "structure",
        "engine": "rules",
        "prompt": FINDING_SCHEMA,
    },
    {
        "key": "table",
        "label": "Table reviewer",
        "category": "structure",
        "engine": "rules",
        "prompt": FINDING_SCHEMA,
    },
    {
        "key": "workflow",
        "label": "Workflow reviewer",
        "category": "logic",
        "engine": "llm",
        "prompt": """For a document that actually defines a workflow, map each written
step and decision. Report only evidenced dead ends, unreachable steps, uncontrolled
loops, missing start/end, or mismatches against another written process representation.
If there is no workflow, return no findings.\n\n"""
        + FINDING_SCHEMA,
    },
    {
        "key": "flowchart",
        "label": "Flowchart reviewer",
        "category": "logic",
        "engine": "hybrid",
        "prompt": """Review only flowchart content represented in the supplied text.
Verify explicit YES and NO branches, start/end nodes, orphan nodes, reachable paths,
loops with exits, and agreement with written procedure steps. Never claim to inspect
an image that is not represented in the text. If no textual flowchart exists, return
no findings.\n\n"""
        + FINDING_SCHEMA,
    },
    {
        "key": "iso",
        "label": "ISO/QMS compliance",
        "category": "iso",
        "engine": "hybrid",
        "prompt": """Only when the document is clearly controlled SOP/QMS documentation,
assess Purpose, Scope, Definitions, Responsibilities, Procedure, Records, References,
Revision History, ownership/approval/effective date, and risk/escalation handling.
For ordinary prose, do not demand ISO sections.

Also return "present_sections", "missing_sections", and "is_sop" alongside the
standard finding schema. Missing sections are findings only for a controlled SOP/QMS
document, never for ordinary prose.\n\n"""
        + FINDING_SCHEMA,
    },
)

VERIFIER_PROMPT = """You are the final false-positive verifier. Each proposed item
contains its own source context. Keep only a demonstrable and useful error.

Reject optional style, speculative claims, duplicates, changes to valid company
terminology/product names/glossary entries/code/URLs/file names/table identifiers,
claims unsupported by context, or anything that changes meaning.
Keep genuine grammar, spelling, internal inconsistency, structure, procedure, logic,
and controlled-document defects.

Return ONLY:
{"items":[{"id":"<exact id>","keep":true,"confidence":0.0,
"rule_valid":true,"evidence_valid":true,
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
            rule_valid = row.get("rule_valid") is True
            evidence_valid = row.get("evidence_valid") is True
            keep = (
                keep_value
                if isinstance(keep_value, bool) and rule_valid and evidence_valid
                else False
            )
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
