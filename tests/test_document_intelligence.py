from __future__ import annotations

import pytest

from app.document_intelligence import (
    detect_document_type,
    is_protected_change,
    is_reviewer_applicable,
    protected_ranges,
    review_cross_references,
    review_flowchart_logic,
    review_headings,
    review_iso_sections,
    review_markdown_tables,
    review_numbering,
    review_procedure_consistency,
    review_terminology,
    run_rule_reviewer,
    verify_safe_spelling,
)


@pytest.mark.parametrize(
    ("text", "declared", "expected"),
    [
        ("plain text", "docx", "docx"),
        ("<html><h1>Title</h1></html>", "auto", "html"),
        ("# Title\n\n- item\n", "auto", "markdown"),
        ("Ordinary business sentence.", "auto", "txt"),
        ("Anything", "unknown", "txt"),
    ],
)
def test_document_type_detection(text: str, declared: str, expected: str) -> None:
    assert detect_document_type(text, declared) == expected


def test_protected_ranges_cover_machine_content() -> None:
    text = (
        "Visit https://example.com/a, email qa@example.com, use path/to/file.txt.\n"
        "`inline()`\n```\ncode()\n```\n[link](https://target.example)\n"
    )
    kinds = {kind for _, _, kind in protected_ranges(text, "markdown")}
    assert {
        "url",
        "email",
        "file",
        "inline_code",
        "fenced_code",
        "link_target",
    } <= kinds
    assert protected_ranges("<strong>Text</strong>", "html")[0][2] == "html_tag"


def test_change_suppression_reasons() -> None:
    text = "Visit https://example.com.\nGraphic Design Manager\nTable QMS-1: Owner"
    url_start = text.index("https")
    assert "protected url" in (
        is_protected_change(
            text, url_start, url_start + 19, "https://example.com", "x", "txt"
        )
        or ""
    )
    role_start = text.index("Graphic")
    assert "terminology" in (
        is_protected_change(
            text,
            role_start,
            role_start + len("Graphic Design Manager"),
            "Graphic Design Manager",
            "Design Manager",
            "txt",
        )
        or ""
    )
    table_start = text.index("Owner")
    assert "identifier" in (
        is_protected_change(
            text, table_start, table_start + 5, "Owner", "Owners", "txt"
        )
        or ""
    )
    assert "anchored" in (
        is_protected_change(text, None, None, "missing", "x", "txt") or ""
    )
    assert is_protected_change(text, 0, 5, "Visit", "Review", "txt") is None


def test_safe_spelling_verifier_is_narrow_and_case_sensitive() -> None:
    assert verify_safe_spelling("teh", "the", "spelling")
    assert verify_safe_spelling("recieve", "receive", "spelling")
    assert not verify_safe_spelling("Teh", "The", "spelling")
    assert not verify_safe_spelling("product", "products", "spelling")
    assert not verify_safe_spelling("teh", "the", "style")


def test_markdown_heading_review_detects_skip_and_duplicate() -> None:
    text = "# Introduction\n\n### Detail\n\n## Repeated\n\n## Repeated\n"
    findings = review_headings(text, "markdown", {})
    assert {finding["finding"] for finding in findings} == {
        "Skipped heading level",
        "Duplicate heading",
    }
    assert review_headings(text, "txt", {}) == []


def test_docx_heading_review_uses_word_metadata_not_markdown() -> None:
    metadata = {
        "headings": [
            {"text": "Parent", "level": 1},
            {"text": "Child", "level": 3},
        ]
    }
    findings = review_headings("# Not Markdown", "docx", metadata)
    assert findings[0]["evidence"].startswith("'Parent'")


def test_numbering_review() -> None:
    broken = "1. First\n2. Second\n4. Fourth\n"
    assert review_numbering(broken)[0]["finding"] == "Non-sequential numbering"
    assert review_numbering("1. One\n2. Two\n3. Three") == []
    assert review_numbering("1. One\n3. Three") == []


def test_markdown_table_review() -> None:
    broken = "| A | B |\n|---|---|\n| one |\n"
    findings = review_markdown_tables(broken, "markdown")
    assert findings[0]["severity"] == "major"
    assert review_markdown_tables("| A |\n|---|\n", "markdown") == []
    assert review_markdown_tables(broken, "docx") == []


def test_cross_reference_review() -> None:
    text = "1. Start\n2. Continue\nSee Section 9 and Step 2."
    findings = review_cross_references(text)
    assert len(findings) == 1
    assert "Section 9" in findings[0]["evidence"]
    assert review_cross_references("See Section 9.") == []


def test_terminology_memory_reports_only_dominant_long_form() -> None:
    inconsistent = (
        "Graphic Design Manager approved. Graphic Design Manager reviewed. "
        "Graphic Design Manager signed. Design Manager archived."
    )
    findings = review_terminology(inconsistent)
    assert len(findings) == 1
    assert "appears 3 times" in findings[0]["evidence"]
    assert review_terminology("Design Manager approved once.") == []


def test_procedure_form_consistency() -> None:
    mismatch = "Procedure\nUse Form QMS-17.\n\nRecords\nRetain Form QMS-18."
    finding = review_procedure_consistency(mismatch)[0]
    assert finding["finding"] == "Procedure and records form mismatch"
    assert finding["severity"] == "major"
    assert (
        review_procedure_consistency(
            "Procedure\nUse Form QMS-17.\nRecords\nRetain Form QMS-17."
        )
        == []
    )
    assert review_procedure_consistency("Ordinary prose.") == []


def test_flowchart_logic_branches_and_terminals() -> None:
    findings = review_flowchart_logic("FLOWCHART\nStart -> Complete?\nYES -> Publish")
    names = {finding["finding"] for finding in findings}
    assert names == {"Missing NO branch", "Missing flowchart end"}
    complete = "FLOWCHART\nStart -> Complete?\nYES -> End\nNO -> End"
    assert review_flowchart_logic(complete) == []
    assert review_flowchart_logic("Ordinary prose.") == []
    assert (
        review_flowchart_logic("FLOWCHART\nReview -> End")[0]["finding"]
        == "Missing flowchart start"
    )


def test_iso_required_sections() -> None:
    incomplete = "STANDARD OPERATING PROCEDURE\nPurpose\nProcedure"
    finding = review_iso_sections(incomplete)[0]
    assert "scope" in finding["evidence"]
    complete = (
        "STANDARD OPERATING PROCEDURE Purpose Scope Responsibilities "
        "Procedure Records Revision"
    )
    assert review_iso_sections(complete) == []
    assert review_iso_sections("Business memo") == []


@pytest.mark.parametrize(
    "key",
    [
        "terminology",
        "cross_reference",
        "heading",
        "numbering",
        "table",
        "procedure",
        "flowchart",
        "iso",
    ],
)
def test_rule_reviewer_contract(key: str) -> None:
    result = run_rule_reviewer(key, "A clean sentence.", "txt", {})
    assert result["failed"] is False
    assert isinstance(result["findings"], list)
    assert result["verdict"]


@pytest.mark.parametrize(
    ("key", "text", "expected"),
    [
        ("procedure", "Ordinary prose.", False),
        ("procedure", "1. A\n2. B\n3. C", True),
        ("workflow", "Is it complete?", True),
        ("workflow", "Ordinary prose.", False),
        ("flowchart", "Start -> End", True),
        ("flowchart", "Ordinary prose.", False),
        ("iso", "QMS work instruction", True),
        ("iso", "Ordinary prose.", False),
        ("role", "Ordinary prose.", True),
    ],
)
def test_reviewer_applicability(key: str, text: str, expected: bool) -> None:
    assert is_reviewer_applicable(key, text) is expected
