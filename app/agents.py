"""Specialist document-review agents.

Each agent is an independent LLM call with its own system prompt and a shared
structured-JSON output contract. The orchestrator (pipeline.py) runs them in
parallel, merges and deduplicates their findings, passes everything through a
false-positive verification agent, and computes deterministic scores.

Agents and what they own:
  terminology — roles/departments/systems/abbreviations named consistently
  structure   — headings, numbering, bullets, spacing, duplicate headings
  procedure   — prose steps vs tables vs responsibilities cross-validation
  logic       — decision points and text-represented flowcharts (branches,
                dead ends, unreachable steps)
  iso         — ISO 9001 / QMS documentation conventions + mandatory sections
  verifier    — reviews every suggestion from all agents, rejects likely
                false positives with a confidence score
  summary     — executive summary, risk level, top issues
"""

from __future__ import annotations

import json

import httpx

from app.cerebras_client import call_json

FINDING_SCHEMA = """Return ONLY valid JSON, nothing else, no markdown fences:
{
  "findings": [
    {
      "title": "<short name of the issue>",
      "detail": "<what is wrong and where, plain language, with a concrete recommendation>",
      "location": "<short verbatim quote or section name where the issue occurs>",
      "severity": "<minor|major>"
    }
  ],
  "verdict": "<one sentence assessment of this dimension>"
}
Only report real findings you can point to in the text. An empty findings array is a
valid answer. Never invent issues to seem thorough."""

DOC_AGENTS: list[dict] = [
    {
        "key": "terminology",
        "label": "Terminology consistency",
        "category": "terminology",
        "prompt": """You are a terminology-consistency reviewer. Extract every department, role, \
designation, system, software, product, abbreviation, and defined term in the document, and check \
that each one is named identically at every occurrence.

Rules:
- Flag variants of the same name (e.g. "Graphic Designing department" vs "Graphic Design department") \
and state which form dominates so the minority form can be aligned to it.
- Flag abbreviations that are used but never defined.
- NEVER suggest renaming a term that is used consistently — consistent usage is correct by definition.
- Check the entire document before concluding anything is inconsistent.

""" + FINDING_SCHEMA,
    },
    {
        "key": "structure",
        "label": "Structure & formatting",
        "category": "structure",
        "prompt": """You are a document structure and formatting reviewer. Check:
- Heading hierarchy and duplicate headings.
- Numbering: sequential, no skipped or repeated numbers, consistent style (1. vs 1)).
- Bullet/list style consistency across sections.
- Heading punctuation consistency (if most headings have no colon, one with a colon is the defect — \
never the other way round).
- Spacing anomalies you can see in the text (double spaces, stray blank lines, duplicated fragments \
such as "Version: R00Version: R00").
- Table formatting consistency (column counts, header rows).

Do NOT report grammar, spelling, or punctuation inside sentences — other agents own those.

""" + FINDING_SCHEMA,
    },
    {
        "key": "procedure",
        "label": "Procedure validation",
        "category": "procedure",
        "prompt": """You are a procedure validation reviewer for SOP-style documents. Extract every \
representation of the process — numbered prose steps, procedure tables, responsibility lists, \
checklists — and cross-compare them.

Detect:
- Steps present in one representation but missing from another.
- Duplicated steps; steps in a different order between representations.
- Role mismatches (a step owned by different roles in different places).
- Skipped or wrong numbering; references to steps, sections, tables, or attachments that do not exist.
- Steps with no stated input or output where the surrounding steps have them.

If the document contains no procedure, return an empty findings array.

""" + FINDING_SCHEMA,
    },
    {
        "key": "logic",
        "label": "Flowchart & decision logic",
        "category": "logic",
        "prompt": """You are a workflow-logic reviewer. Build a mental graph of the process as \
described in the text (numbered steps, decision questions, flowchart-like listings).

Validate:
- Every decision point (e.g. "Approved?", "Complete?") has ALL branches: a Yes path needs a No path.
- Every step leads somewhere; no dead ends (except a defined final step).
- No unreachable or isolated steps; no steps that loop forever with no exit condition.
- No duplicated nodes describing the same action as if it were two different steps.

Note: you can only see the text. If the document references a drawn flowchart image whose contents \
are not in the text, report ONE minor finding saying the flowchart image itself could not be \
validated and should be checked by eye.

If the document has no process flow at all, return an empty findings array.

""" + FINDING_SCHEMA,
    },
    {
        "key": "iso",
        "label": "ISO/QMS compliance",
        "category": "iso",
        "prompt": """You are an ISO 9001 / QMS documentation reviewer. Assess the document as \
controlled documentation.

Check for presence and quality of: Purpose, Scope, Definitions/Abbreviations, Responsibilities, \
Procedure, Records, References, Revision History / Version, Document Control (owner, approvals, \
effective date), Risk/Escalation handling.

Also check: undefined abbreviations, broken cross-references, improper section numbering, and \
wording consistency expected in ISO documentation (shall/should/may used deliberately).

Only treat a missing section as a finding if the document is clearly meant to be a controlled \
SOP/QMS document; for ordinary prose, return an empty findings array.

Return ONLY valid JSON, nothing else, no markdown fences:
{
  "findings": [
    {"title": "...", "detail": "...", "location": "...", "severity": "minor|major"}
  ],
  "present_sections": ["<section names found>"],
  "missing_sections": ["<mandatory sections not found — empty if not an SOP-style document>"],
  "is_sop": true,
  "verdict": "<one sentence>"
}""",
    },
]

VERIFIER_PROMPT = """You are a false-positive verification reviewer — the last line of defence \
before suggestions reach a human. You receive (1) an excerpt of the document and (2) a numbered \
list of proposed corrections and review findings produced by other reviewers.

For EACH item, decide whether it should be kept, and give a confidence between 0 and 1 that the \
item is a genuine, useful issue.

Reject (keep=false) any item that:
- contradicts the document's own consistent conventions (e.g. renames a consistently-used role or \
title, adds punctuation to one heading when the others have none);
- imposes an optional style preference the document applies consistently already;
- is speculative, duplicated by another item, or too trivial to be worth a human's attention;
- refers to text that does not actually appear in the document.

Keep genuine grammar/spelling errors, real inconsistencies, and structural/logic defects.

Return ONLY valid JSON, nothing else:
{"items": [{"id": "<id from the list>", "keep": true, "confidence": 0.0, "note": "<only when keep=false: one short reason>"}]}
Include every id you were given exactly once."""

SUMMARY_PROMPT = """You are an executive review summariser. You receive document statistics, \
computed scores, and the top verified findings. Write for a manager deciding whether the document \
is fit for release.

Return ONLY valid JSON, nothing else:
{
  "summary": "<3-4 sentence executive summary of document quality and what to fix first>",
  "risk_level": "<low|medium|high>",
  "top_issues": ["<up to 5 one-line issue statements, most important first>"],
  "readability": <0-100 integer, how easy the document is to read and follow>
}"""


async def run_doc_agent(client: httpx.AsyncClient, agent: dict, text: str) -> dict:
    """Run one document-level agent. Returns normalised {findings, verdict, ...}."""
    parsed = await call_json(client, agent["prompt"], text, max_tokens=4096)
    if parsed is None:
        return {"findings": [], "verdict": "", "failed": True}
    # Tolerate shape drift: valid JSON without a findings list means "no findings".
    if not isinstance(parsed.get("findings"), list):
        parsed["findings"] = []
    parsed.setdefault("verdict", "")
    return parsed


async def run_verifier(
    client: httpx.AsyncClient, text: str, items: list[dict]
) -> dict:
    """items: [{"id","kind","text"}] — returns {id: {"keep","confidence","note"}}."""
    if not items:
        return {}
    listing = "\n".join(f'{it["id"]}. [{it["kind"]}] {it["text"]}' for it in items)
    user = f"DOCUMENT EXCERPT:\n{text[:20000]}\n\nPROPOSED ITEMS:\n{listing}"
    parsed = await call_json(client, VERIFIER_PROMPT, user, max_tokens=6144)
    out: dict = {}
    for row in (parsed or {}).get("items") or []:
        if not isinstance(row, dict) or "id" not in row:
            continue
        try:
            conf = max(0.0, min(1.0, float(row.get("confidence", 0.5))))
        except (TypeError, ValueError):
            conf = 0.5
        out[str(row["id"])] = {
            "keep": bool(row.get("keep", True)),
            "confidence": conf,
            "note": str(row.get("note") or ""),
        }
    return out


async def run_summary(
    client: httpx.AsyncClient, stats: dict, scores: dict, top_findings: list[dict]
) -> dict:
    user = json.dumps(
        {
            "stats": {k: v for k, v in stats.items() if k not in ("review", "report")},
            "scores": scores,
            "top_findings": [
                {
                    "title": f.get("title"),
                    "severity": f.get("severity"),
                    "agent": f.get("agent"),
                }
                for f in top_findings[:12]
            ],
        }
    )
    parsed = await call_json(client, SUMMARY_PROMPT, user, max_tokens=1024)
    if parsed is None:
        return {"summary": "", "risk_level": "medium", "top_issues": [], "readability": None}
    risk = str(parsed.get("risk_level") or "medium").lower()
    if risk not in ("low", "medium", "high"):
        risk = "medium"
    try:
        readability = parsed.get("readability")
        readability = None if readability is None else max(0, min(100, int(readability)))
    except (TypeError, ValueError):
        readability = None
    return {
        "summary": str(parsed.get("summary") or "").strip(),
        "risk_level": risk,
        "top_issues": [str(t) for t in (parsed.get("top_issues") or [])[:5]],
        "readability": readability,
    }
