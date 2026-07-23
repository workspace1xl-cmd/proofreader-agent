from __future__ import annotations

from benchmarks.evaluate import score_case


def test_benchmark_scoring_counts_hits_misses_and_protected_changes() -> None:
    case = {
        "expected_text_contains": ["correct phrase", "missing phrase"],
        "expected_findings": ["broken branch"],
        "protected_literals": ["Product Name", "https://example.com"],
    }
    payload = {
        "corrected_text": "correct phrase and https://example.com",
        "changes": [{"original": "x"}],
        "stats": {
            "report": {
                "findings": [
                    {"title": "Broken branch", "detail": "No path"},
                    {"title": "Extra", "detail": "Unlabelled issue"},
                ]
            }
        },
    }
    metrics = score_case(case, payload)
    assert metrics["true_positives"] == 2
    assert metrics["false_negatives"] == 1
    assert metrics["protected_content_changed"] == ["Product Name"]
    assert metrics["false_positives"] == 2
