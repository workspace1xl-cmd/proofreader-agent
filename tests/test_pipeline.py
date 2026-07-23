from __future__ import annotations

import asyncio
import itertools
from typing import Any

import pytest

import app.pipeline as pipeline


def test_split_spans_tiles_text_and_prefers_boundaries() -> None:
    text = ("Paragraph one. " * 500) + "\n\n" + ("Paragraph two! " * 500)
    spans = pipeline.split_spans(text)
    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)
    assert all(left[1] == right[0] for left, right in itertools.pairwise(spans))
    assert "".join(text[start:end] for start, end in spans) == text


@pytest.mark.parametrize("text", ["", "x", "a" * 20_000, "😀" * 20_000])
def test_split_spans_pathological_inputs(text: str) -> None:
    spans = pipeline.split_spans(text)
    assert "".join(text[start:end] for start, end in spans) == text


def test_anchor_exact_repeated_and_whitespace_drift() -> None:
    text = "He dont stop. She dont stop.\nTwo   spaces."
    raw = [
        {"original": "dont", "corrected": "doesn't", "category": "grammar"},
        {"original": "dont", "corrected": "doesn't", "category": "grammar"},
        {"original": "Two spaces", "corrected": "Two spaces", "category": "clarity"},
        {"original": "Two spaces", "corrected": "Two spaces.", "category": "clarity"},
    ]
    changes = pipeline.anchor_changes(text, raw, 100)
    assert [change["start"] for change in changes[:2]] == [103, 118]
    assert changes[2]["start"] == 129
    assert changes[2]["original"] == "Two   spaces"


def test_anchor_rejects_noops_invalid_rows_and_missing_text() -> None:
    changes = pipeline.anchor_changes(
        "hello",
        [
            None,
            {"original": "", "corrected": "x"},
            {"original": "same", "corrected": "same"},
            {"original": "absent", "corrected": "present", "category": "unknown"},
        ],
        0,
    )
    assert len(changes) == 1
    assert changes[0]["start"] is None
    assert changes[0]["category"] == "clarity"


def test_dedupe_and_overlap_resolution_prefers_verified_confidence() -> None:
    base = {
        "category": "grammar",
        "reason": "",
        "severity": "minor",
        "verified": True,
    }
    changes = [
        {
            **base,
            "original": "abc",
            "corrected": "x",
            "start": 0,
            "end": 3,
            "confidence": 0.7,
        },
        {
            **base,
            "original": "abc",
            "corrected": "x",
            "start": 0,
            "end": 3,
            "confidence": 0.7,
        },
        {
            **base,
            "original": "bc",
            "corrected": "y",
            "start": 1,
            "end": 3,
            "confidence": 0.9,
        },
        {
            **base,
            "original": "z",
            "corrected": "q",
            "start": 5,
            "end": 6,
            "confidence": 0.8,
        },
    ]
    deduped = pipeline.dedupe_changes(changes)
    resolved = pipeline.resolve_overlaps(deduped)
    assert len(deduped) == 3
    assert [(item["start"], item["corrected"]) for item in resolved] == [
        (1, "y"),
        (5, "q"),
    ]


def test_finding_dedupe_uses_title_and_location() -> None:
    findings = [
        {"title": "Missing branch", "location": "Step 2"},
        {"title": "missing BRANCH!", "location": "Step 2"},
        {"title": "Missing branch", "location": "Step 9"},
    ]
    assert len(pipeline.dedupe_findings(findings)) == 2


def test_apply_changes_uses_only_verified_exact_nonoverlapping() -> None:
    changes = [
        {
            "original": "bad",
            "corrected": "good",
            "start": 0,
            "end": 3,
            "verified": True,
        },
        {
            "original": "text",
            "corrected": "copy",
            "start": 4,
            "end": 8,
            "verified": False,
        },
        {
            "original": "missing",
            "corrected": "x",
            "start": None,
            "end": None,
            "verified": True,
        },
    ]
    assert pipeline.apply_changes("bad text", changes) == "good text"


def test_scores_are_deterministic_bounded_and_ignore_unverified() -> None:
    changes = [
        {"category": "grammar", "severity": "major", "verified": True},
        {"category": "spelling", "severity": "major", "verified": None},
        {"category": "punctuation", "severity": "minor", "verified": False},
    ]
    findings = [
        {"agent": "logic", "severity": "major", "verified": True},
        {"agent": "iso", "severity": "major", "verified": None},
    ]
    first = pipeline.compute_scores(changes, findings, 500)
    second = pipeline.compute_scores(changes, findings, 500)
    assert first == second
    assert first["grammar"] < 100 and first["logic"] < 100
    assert first["spelling"] == first["punctuation"] == first["iso"] == 100
    assert first["readability"] == first["style"] == 100
    assert all(0 <= score <= 100 for score in first.values())


async def _fake_chunk(
    client: Any,
    text: str,
    *,
    context_before: str = "",
    context_after: str = "",
    document_type: str = "txt",
) -> dict[str, Any]:
    await asyncio.sleep(0.001)
    changes = []
    if "teh" in text:
        changes.append(
            {
                "original": "teh",
                "corrected": "the",
                "category": "spelling",
                "reason": "Misspelling",
                "severity": "minor",
            }
        )
    return {"changes": changes, "failed": False}


async def _fake_agent(client: Any, agent: dict[str, str], text: str) -> dict[str, Any]:
    await asyncio.sleep(0.001)
    if agent["key"] == "workflow":
        return {
            "findings": [
                {
                    "title": "Missing No branch",
                    "detail": "Approved has no No branch.",
                    "location": "Approved?",
                    "severity": "major",
                }
            ],
            "verdict": "One issue",
            "failed": False,
        }
    return {"findings": [], "verdict": "Clean", "failed": False}


async def _fake_verifier(
    client: Any,
    items: list[dict[str, str]],
    *,
    batch_size: int = 40,
) -> dict[str, dict[str, Any]]:
    return {
        item["id"]: {"keep": True, "confidence": 0.99, "note": ""} for item in items
    }


async def _fake_summary(
    client: Any,
    stats: dict[str, Any],
    scores: dict[str, int],
    top_findings: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "summary": "Fix the identified issues.",
        "risk_level": "medium",
        "top_issues": ["Missing No branch"],
        "readability": 88,
        "failed": False,
    }


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "proofread_chunk", _fake_chunk)
    monkeypatch.setattr(pipeline, "run_doc_agent", _fake_agent)
    monkeypatch.setattr(pipeline, "run_verifier", _fake_verifier)
    monkeypatch.setattr(pipeline, "run_summary", _fake_summary)


def test_pipeline_integration_progress_merge_and_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_pipeline(monkeypatch)
    progress: list[tuple[int, int]] = []
    statuses: list[tuple[str, str]] = []

    async def on_progress(done: int, total: int) -> None:
        progress.append((done, total))

    async def on_status(key: str, label: str, state: str) -> None:
        statuses.append((key, state))

    text = "teh document. Approved?\n" + ("More words. " * 1_000)
    result = asyncio.run(pipeline.run_pipeline(text, on_progress, on_status))
    report = result["stats"]["report"]
    assert progress[0][0] == 0 and progress[-1][0] == progress[-1][1]
    assert result["corrected_text"].startswith("the document")
    assert text[result["changes"][0]["start"] : result["changes"][0]["end"]] == "teh"
    assert report["findings"][0]["verified"] is True
    assert report["summary"]["readability"] == 88
    assert len(report["agents"]) == 16
    assert report["document_type"] == "txt"
    assert ("verifier", "done") in statuses


def test_pipeline_isolates_agent_and_chunk_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def broken_chunk(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise ValueError("bad provider shape")

    async def broken_agent(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise ValueError("bad provider shape")

    _patch_pipeline(monkeypatch)
    monkeypatch.setattr(pipeline, "proofread_chunk", broken_chunk)
    monkeypatch.setattr(pipeline, "run_doc_agent", broken_agent)
    result = asyncio.run(pipeline.run_pipeline("A valid sentence."))
    assert result["corrected_text"] == "A valid sentence."
    statuses = {
        agent["key"]: agent["status"] for agent in result["stats"]["report"]["agents"]
    }
    assert statuses["grammar"] == "failed"
    assert statuses["role"] == "failed"
    assert statuses["heading"] == "done"


@pytest.mark.stress
def test_maximum_length_document_stays_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_pipeline(monkeypatch)
    text = ("Unicode café résumé 😀. " * 5_000)[:100_000]
    result = asyncio.run(pipeline.run_pipeline(text))
    assert result["stats"]["chars"] == len(text)
    assert result["stats"]["chunks"] >= 16
    assert len(result["corrected_text"]) == len(text)


def test_concurrency_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    active = 0
    maximum = 0

    async def monitored_chunk(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {"changes": [], "failed": False}

    async def monitored_agent(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return await monitored_chunk()

    _patch_pipeline(monkeypatch)
    monkeypatch.setattr(pipeline, "proofread_chunk", monitored_chunk)
    monkeypatch.setattr(pipeline, "run_doc_agent", monitored_agent)
    asyncio.run(pipeline.run_pipeline("sentence. " * 5_000))
    assert maximum <= pipeline.PIPELINE_CONCURRENCY


def test_concurrency_limit_is_shared_across_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    maximum = 0

    async def monitored_chunk(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {"changes": [], "failed": False}

    async def monitored_agent(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {"findings": [], "verdict": "Clean", "failed": False}

    _patch_pipeline(monkeypatch)
    monkeypatch.setattr(pipeline, "proofread_chunk", monitored_chunk)
    monkeypatch.setattr(pipeline, "run_doc_agent", monitored_agent)

    async def exercise() -> None:
        await asyncio.gather(
            pipeline.run_pipeline("sentence. " * 3_000),
            pipeline.run_pipeline("sentence. " * 3_000),
        )

    asyncio.run(exercise())
    assert maximum <= pipeline.PIPELINE_CONCURRENCY


def test_deterministic_findings_survive_external_verifier_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_pipeline(monkeypatch)

    async def unavailable_verifier(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    monkeypatch.setattr(pipeline, "run_verifier", unavailable_verifier)
    result = asyncio.run(
        pipeline.run_pipeline(
            "# Title\n\n### Skipped\n",
            document_type="markdown",
        )
    )
    finding = result["stats"]["report"]["findings"][0]
    assert finding["title"] == "Skipped heading level"
    assert finding["verified"] is True
    assert finding["confidence"] == 0.98
    assert finding["verification_basis"] == "deterministic_rule"


def test_pipeline_returns_only_high_confidence_unprotected_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def candidate_chunk(
        client: Any,
        text: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "failed": False,
            "changes": [
                {
                    "original": "https://example.com",
                    "corrected": "https://example.org",
                    "category": "style",
                },
                {
                    "original": "teh",
                    "corrected": "the",
                    "category": "spelling",
                    "rule": "Correct spelling",
                },
                {
                    "original": "very",
                    "corrected": "extremely",
                    "category": "style",
                    "rule": "Optional style",
                },
            ],
        }

    async def confidence_verifier(
        client: Any,
        items: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, dict[str, Any]]:
        return {
            item["id"]: {
                "keep": True,
                "confidence": 0.8 if "very" in item["text"] else 0.95,
                "note": "",
            }
            for item in items
        }

    _patch_pipeline(monkeypatch)
    monkeypatch.setattr(pipeline, "proofread_chunk", candidate_chunk)
    monkeypatch.setattr(pipeline, "run_verifier", confidence_verifier)
    result = asyncio.run(
        pipeline.run_pipeline(
            "Visit https://example.com. Fix teh word; it is very long.",
            document_type="txt",
        )
    )
    assert [change["original"] for change in result["changes"]] == ["teh"]
    verification = result["stats"]["report"]["verification"]
    assert verification["verified_corrections"] == 1
    assert verification["filtered_corrections"] == 2


def test_context_helpers_cover_missing_and_located_findings() -> None:
    text = "A" * 500 + "Target section" + "B" * 500
    assert pipeline._context_for_span(text, None, None) == ""
    located = pipeline._context_for_finding(text, {"location": "Target section"})
    assert "Target section" in located and len(located) < len(text)
    fallback = pipeline._context_for_finding(text, {"location": "missing"})
    assert "…" in fallback
    assert pipeline._context_for_finding("short", {"location": "missing"}) == "short"


def test_hybrid_reviewers_merge_rules_with_model_findings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_pipeline(monkeypatch)
    text = (
        "STANDARD OPERATING PROCEDURE\nPurpose\nScope\nResponsibilities\n"
        "Procedure\n1. Use Form QMS-17.\n2. Approve.\n3. Archive.\n"
        "Records\nRetain Form QMS-18.\nRevision"
    )
    result = asyncio.run(pipeline.run_pipeline(text, document_type="txt"))
    titles = {finding["title"] for finding in result["stats"]["report"]["findings"]}
    assert "Procedure and records form mismatch" in titles
