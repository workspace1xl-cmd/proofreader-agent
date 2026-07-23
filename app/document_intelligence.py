"""Format detection, protected-span discovery, and deterministic specialist reviews."""

from __future__ import annotations

import re
from collections import Counter
from itertools import pairwise
from typing import Any

DOCUMENT_TYPES = {"auto", "txt", "markdown", "docx", "html", "pdf"}
_MARKDOWN_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$", re.MULTILINE)
_NUMBERED_LINE = re.compile(r"^\s*(\d+(?:\.\d+)*)[.)]?[ \t]+(.+)$", re.MULTILINE)
_REFERENCE = re.compile(
    r"\b(section|step|table|figure|appendix)\s+([A-Z]|\d+(?:\.\d+)*)\b",
    re.IGNORECASE,
)
_CAPITALIZED_PHRASE = re.compile(
    r"\b(?:[A-Z][A-Za-z&/-]*)(?:\s+[A-Z][A-Za-z&/-]*){1,4}\b"
)


def detect_document_type(text: str, declared: str = "auto") -> str:
    normalized = declared.strip().lower()
    if normalized in DOCUMENT_TYPES - {"auto"}:
        return normalized
    sample = text[:10_000]
    if re.search(r"<(?:html|body|h[1-6]|p|table)\b", sample, re.IGNORECASE):
        return "html"
    markdown_signals = sum(
        (
            bool(_MARKDOWN_HEADING.search(sample)),
            bool(re.search(r"^\s*[-*+]\s+\S", sample, re.MULTILINE)),
            bool(re.search(r"\[[^\]]+]\([^)]+\)", sample)),
            bool(re.search(r"^\s*\|.+\|\s*$", sample, re.MULTILINE)),
            "```" in sample,
        )
    )
    return "markdown" if markdown_signals >= 2 else "txt"


def protected_ranges(text: str, document_type: str) -> list[tuple[int, int, str]]:
    """Return spans that proofreading must not alter."""

    patterns: list[tuple[str, str]] = [
        ("url", r"https?://[^\s<>)]+"),
        ("email", r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
        (
            "file",
            r"(?<!\w)(?:[A-Za-z]:\\|/)?(?:[\w.-]+[/\\])+[\w.-]+\.[A-Za-z0-9]{1,8}\b",
        ),
    ]
    if document_type == "markdown":
        patterns.extend(
            [
                ("fenced_code", r"```[\s\S]*?```|~~~[\s\S]*?~~~"),
                ("inline_code", r"`[^`\n]+`"),
                ("link_target", r"(?<=\]\()[^)]+(?=\))"),
            ]
        )
    elif document_type == "html":
        patterns.append(("html_tag", r"<[^>]+>"))
    output: list[tuple[int, int, str]] = []
    for kind, pattern in patterns:
        output.extend(
            (match.start(), match.end(), kind)
            for match in re.finditer(pattern, text, re.IGNORECASE)
        )
    return sorted(output)


def is_protected_change(
    text: str,
    start: int | None,
    end: int | None,
    original: str,
    corrected: str,
    document_type: str,
) -> str | None:
    if start is None or end is None:
        return "Correction could not be anchored to exact source text."
    for left, right, kind in protected_ranges(text, document_type):
        if start < right and end > left:
            return f"Suppressed change inside protected {kind} content."
    original_words = original.split()
    corrected_words = corrected.split()
    if (
        len(original_words) >= 2
        and all(word[:1].isupper() for word in original_words if word)
        and [word.casefold() for word in original_words]
        != [word.casefold() for word in corrected_words]
    ):
        return "Suppressed possible company, product, role, or defined terminology."
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line = text[line_start : len(text) if line_end < 0 else line_end]
    if re.match(r"^\s*(?:table|figure|form|record)\s+[\w.-]+\s*:", line, re.I):
        return "Suppressed change to a table, figure, form, or record identifier."
    return None


def _finding(
    name: str,
    evidence: str,
    rule: str,
    reason: str,
    suggested_fix: str,
    context: str,
    severity: str = "minor",
) -> dict[str, Any]:
    return {
        "finding": name,
        "title": name,
        "evidence": evidence,
        "rule": rule,
        "confidence": None,
        "severity": severity,
        "reason": reason,
        "detail": reason,
        "suggested_fix": suggested_fix,
        "supporting_context": context,
        "location": context[:240],
    }


def review_headings(
    text: str, document_type: str, metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    if document_type == "markdown":
        headings = [
            (len(match.group(1)), match.group(2).strip())
            for match in _MARKDOWN_HEADING.finditer(text)
        ]
    elif document_type == "docx":
        rows = metadata.get("headings")
        headings = (
            [
                (int(row.get("level", 1)), str(row.get("text") or ""))
                for row in rows
                if isinstance(row, dict)
            ]
            if isinstance(rows, list)
            else []
        )
    else:
        return []
    findings: list[dict[str, Any]] = []
    for previous, current in pairwise(headings):
        if current[0] > previous[0] + 1:
            findings.append(
                _finding(
                    "Skipped heading level",
                    f"{previous[1]!r} (level {previous[0]}) is followed by "
                    f"{current[1]!r} (level {current[0]}).",
                    "Heading levels must increase by no more than one level.",
                    "The hierarchy skips an intermediate heading level.",
                    f"Change {current[1]!r} to level {previous[0] + 1} or add "
                    "the missing parent section.",
                    current[1],
                )
            )
    duplicates = [
        heading
        for heading, count in Counter(
            title.casefold() for _, title in headings if title
        ).items()
        if count > 1
    ]
    for duplicate in duplicates:
        findings.append(
            _finding(
                "Duplicate heading",
                f"The heading {duplicate!r} appears more than once.",
                "Sibling sections should have unique, unambiguous headings.",
                "Repeated headings make navigation and cross-references ambiguous.",
                "Rename or consolidate the duplicated section.",
                duplicate,
            )
        )
    return findings


def review_numbering(text: str) -> list[dict[str, Any]]:
    top_level = [
        (int(match.group(1)), match.group(0).strip())
        for match in _NUMBERED_LINE.finditer(text)
        if "." not in match.group(1)
    ]
    if len(top_level) < 3:
        return []
    findings: list[dict[str, Any]] = []
    for (previous, _), (current, line) in pairwise(top_level):
        if current not in {1, previous + 1}:
            findings.append(
                _finding(
                    "Non-sequential numbering",
                    f"Number {previous} is followed by {current}.",
                    "Numbered sequences must be consecutive unless a new sequence "
                    "explicitly restarts at 1.",
                    "A number is skipped, repeated, or out of order.",
                    f"Renumber {line!r} and subsequent items sequentially.",
                    line,
                )
            )
    return findings


def review_markdown_tables(text: str, document_type: str) -> list[dict[str, Any]]:
    if document_type != "markdown":
        return []
    findings: list[dict[str, Any]] = []
    current: list[str] = []
    for line in [*text.splitlines(), ""]:
        if line.strip().startswith("|") and line.strip().endswith("|"):
            current.append(line)
            continue
        if len(current) >= 2:
            counts = [len(row.strip().strip("|").split("|")) for row in current]
            if len(set(counts)) > 1:
                findings.append(
                    _finding(
                        "Inconsistent table column count",
                        f"Rows contain {', '.join(map(str, counts))} cells.",
                        "Every row in a Markdown table must have the same number "
                        "of columns.",
                        "The table cannot be interpreted reliably.",
                        "Add or remove cells so every row matches the header.",
                        "\n".join(current[:4]),
                        "major",
                    )
                )
        current = []
    return findings


def review_cross_references(text: str) -> list[dict[str, Any]]:
    numbered = {match.group(1).casefold() for match in _NUMBERED_LINE.finditer(text)}
    captions = {
        match.group(2).casefold()
        for match in re.finditer(
            r"^(table|figure|appendix)\s+([A-Z]|\d+(?:\.\d+)*)\b",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
    }
    definitions = numbered | captions
    if len(definitions) < 2:
        return []
    findings: list[dict[str, Any]] = []
    for match in _REFERENCE.finditer(text):
        target = match.group(2).casefold()
        if target not in definitions:
            findings.append(
                _finding(
                    "Broken cross-reference",
                    f"{match.group(0)!r} does not match a detected target.",
                    "Every internal cross-reference must resolve to an existing "
                    "section, step, table, figure, or appendix.",
                    "The referenced target could not be located.",
                    "Correct the identifier or add the missing target.",
                    match.group(0),
                    "major",
                )
            )
    return findings


def review_terminology(text: str) -> list[dict[str, Any]]:
    phrases = Counter(match.group(0) for match in _CAPITALIZED_PHRASE.finditer(text))
    findings: list[dict[str, Any]] = []
    for phrase, count in phrases.items():
        words = phrase.split()
        if count < 3 or len(words) < 3:
            continue
        short = " ".join(words[-2:])
        total_short = len(re.findall(rf"\b{re.escape(short)}\b", text))
        standalone = total_short - count
        if 1 <= standalone <= max(2, count // 3):
            findings.append(
                _finding(
                    "Possible terminology inconsistency",
                    f"{phrase!r} appears {count} times; the shorter form "
                    f"{short!r} appears independently {standalone} time(s).",
                    "Defined roles and terms should use one approved form throughout.",
                    "A minority shortened form may refer to the same role or entity.",
                    "Confirm against the glossary or document owner; align only if "
                    "both forms refer to the same entity.",
                    f"{phrase} / {short}",
                )
            )
    return findings


def review_procedure_consistency(text: str) -> list[dict[str, Any]]:
    folded = text.casefold()
    if "procedure" not in folded or "records" not in folded:
        return []
    procedure_start = folded.find("procedure")
    records_start = folded.find("records", procedure_start + 1)
    if records_start < 0:
        return []
    procedure_text = text[procedure_start:records_start]
    records_text = text[records_start:]
    form_pattern = re.compile(r"\bform\s+([A-Z][A-Z0-9-]*\d)\b", re.IGNORECASE)
    procedure_forms = {
        match.group(1).upper() for match in form_pattern.finditer(procedure_text)
    }
    record_forms = {
        match.group(1).upper() for match in form_pattern.finditer(records_text)
    }
    if not procedure_forms or not record_forms or procedure_forms == record_forms:
        return []
    return [
        _finding(
            "Procedure and records form mismatch",
            f"Procedure cites {sorted(procedure_forms)}; Records cites "
            f"{sorted(record_forms)}.",
            "Forms generated by a procedure must match the forms identified as "
            "controlled records.",
            "The procedure and records sections identify different form numbers.",
            "Confirm the approved form identifier and align both sections.",
            f"Procedure forms: {sorted(procedure_forms)}; "
            f"Records forms: {sorted(record_forms)}",
            "major",
        )
    ]


def review_flowchart_logic(text: str) -> list[dict[str, Any]]:
    folded = text.casefold()
    if "flowchart" not in folded and "->" not in text and "→" not in text:
        return []
    written_start = folded.find("written procedure")
    flow_text = text if written_start < 0 else text[:written_start]
    flow_folded = flow_text.casefold()
    findings: list[dict[str, Any]] = []
    if "?" in flow_text:
        has_yes = bool(re.search(r"\b(?:yes|true)\b", flow_text, re.IGNORECASE))
        has_no = bool(re.search(r"\b(?:no|false)\b", flow_text, re.IGNORECASE))
        if has_yes != has_no:
            missing = "NO" if has_yes else "YES"
            findings.append(
                _finding(
                    f"Missing {missing} branch",
                    f"The flowchart contains a decision and a "
                    f"{'YES' if has_yes else 'NO'} path, but no {missing} path.",
                    "Every binary decision in a flowchart must define both YES and "
                    "NO outcomes.",
                    "One decision outcome has no defined route.",
                    f"Add and label the {missing} branch, matching the written "
                    "procedure.",
                    flow_text[:500],
                    "major",
                )
            )
    if ("->" in flow_text or "→" in flow_text) and "start" not in flow_folded:
        findings.append(
            _finding(
                "Missing flowchart start",
                "No Start node appears in the textual flowchart.",
                "A controlled process flow must have one identifiable start.",
                "The entry point is undefined.",
                "Add a single Start node and connect it to the first activity.",
                flow_text[:500],
            )
        )
    if ("->" in flow_text or "→" in flow_text) and not re.search(
        r"\b(?:end|stop|close)\b", flow_text, re.IGNORECASE
    ):
        findings.append(
            _finding(
                "Missing flowchart end",
                "No End or terminal node appears in the textual flowchart.",
                "Every terminal flow path must reach a defined end state.",
                "The process has no explicit terminal node.",
                "Add an End node and connect every terminal branch.",
                flow_text[:500],
            )
        )
    return findings


def review_iso_sections(text: str) -> list[dict[str, Any]]:
    folded = text.casefold()
    if (
        "standard operating procedure" not in folded
        and "work instruction" not in folded
    ):
        return []
    required = (
        "purpose",
        "scope",
        "responsibilities",
        "procedure",
        "records",
        "revision",
    )
    missing = [section for section in required if section not in folded]
    if not missing:
        return []
    return [
        _finding(
            "Missing controlled-document sections",
            f"Required section labels not found: {', '.join(missing)}.",
            "A controlled SOP/work instruction must identify its purpose, scope, "
            "responsibilities, procedure, records, and revision state.",
            "The document identifies itself as controlled procedural documentation "
            "but lacks required control information.",
            "Add the missing sections or explicitly document why they are not "
            "applicable.",
            ", ".join(missing),
            "major",
        )
    ]


RULE_REVIEWERS = {
    "terminology": lambda text, kind, metadata: review_terminology(text),
    "cross_reference": lambda text, kind, metadata: review_cross_references(text),
    "heading": review_headings,
    "numbering": lambda text, kind, metadata: review_numbering(text),
    "table": lambda text, kind, metadata: review_markdown_tables(text, kind),
    "procedure": lambda text, kind, metadata: review_procedure_consistency(text),
    "flowchart": lambda text, kind, metadata: review_flowchart_logic(text),
    "iso": lambda text, kind, metadata: review_iso_sections(text),
}


def is_reviewer_applicable(key: str, text: str) -> bool:
    """Avoid irrelevant LLM calls and the false positives they tend to create."""

    folded = text.casefold()
    numbered_steps = len(_NUMBERED_LINE.findall(text))
    if key == "procedure":
        return numbered_steps >= 3 or any(
            marker in folded
            for marker in ("procedure", "work instruction", "standard operating")
        )
    if key == "workflow":
        return any(
            marker in folded
            for marker in (
                "workflow",
                "flowchart",
                "if yes",
                "if no",
                "approved?",
                "complete?",
                "->",
            )
        )
    if key == "flowchart":
        return "flowchart" in folded or "->" in text or "→" in text
    if key == "iso":
        return any(
            marker in folded
            for marker in (
                "iso 9001",
                "qms",
                "standard operating procedure",
                "work instruction",
                "revision history",
                "document control",
            )
        )
    return True


def run_rule_reviewer(
    key: str,
    text: str,
    document_type: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    reviewer = RULE_REVIEWERS[key]
    findings = reviewer(text, document_type, metadata)
    return {
        "findings": findings,
        "verdict": (
            f"{len(findings)} evidence-based finding(s)." if findings else "No issues."
        ),
        "failed": False,
    }
