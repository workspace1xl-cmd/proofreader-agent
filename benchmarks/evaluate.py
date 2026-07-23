"""Run the curated human-labelled benchmark against a deployed instance."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent


def _contains_any(value: str, candidates: list[str]) -> bool:
    folded = value.casefold()
    return any(candidate.casefold() in folded for candidate in candidates)


def score_case(case: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    corrected = str(payload.get("corrected_text") or "")
    report = (payload.get("stats") or {}).get("report") or {}
    findings = report.get("findings") or []
    finding_text = " ".join(
        f"{finding.get('title', '')} {finding.get('detail', '')}"
        for finding in findings
        if isinstance(finding, dict)
    )
    expected_corrections = case.get("expected_text_contains") or []
    expected_findings = case.get("expected_findings") or []
    correction_hits = sum(
        expected.casefold() in corrected.casefold() for expected in expected_corrections
    )
    finding_hits = sum(
        expected.casefold() in finding_text.casefold() for expected in expected_findings
    )
    expected_total = len(expected_corrections) + len(expected_findings)
    true_positives = correction_hits + finding_hits
    predicted_total = len(payload.get("changes") or []) + len(findings)
    protected_changes = [
        literal
        for literal in case.get("protected_literals") or []
        if literal not in corrected
    ]
    false_positives = max(0, predicted_total - true_positives) + len(protected_changes)
    return {
        "expected": expected_total,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": expected_total - true_positives,
        "protected_content_changed": protected_changes,
        "correction_recall": (
            correction_hits / len(expected_corrections) if expected_corrections else 1.0
        ),
        "finding_recall": (
            finding_hits / len(expected_findings) if expected_findings else 1.0
        ),
    }


async def evaluate(base_url: str) -> dict[str, Any]:
    cases = json.loads((ROOT / "corpus.json").read_text())
    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=480) as client:
        for case in cases:
            started = time.perf_counter()
            response = await client.post(
                f"{base_url.rstrip('/')}/proofread",
                json={
                    "text": case["text"],
                    "document_type": case["document_type"],
                },
            )
            elapsed = time.perf_counter() - started
            response.raise_for_status()
            payload = response.json()
            results.append(
                {
                    "id": case["id"],
                    "review_standard": case["review_standard"],
                    "elapsed_seconds": round(elapsed, 3),
                    "metrics": score_case(case, payload),
                    "overall_score": (
                        ((payload.get("stats") or {}).get("report") or {})
                        .get("scores", {})
                        .get("overall")
                    ),
                    "corrections": len(payload.get("changes") or []),
                    "findings": len(
                        ((payload.get("stats") or {}).get("report") or {}).get(
                            "findings", []
                        )
                    ),
                }
            )
    true_positives = sum(item["metrics"]["true_positives"] for item in results)
    false_positives = sum(item["metrics"]["false_positives"] for item in results)
    false_negatives = sum(item["metrics"]["false_negatives"] for item in results)
    return {
        "endpoint": base_url,
        "generated_at": datetime.now(UTC).isoformat(),
        "cases": results,
        "summary": {
            "cases": len(results),
            "precision": (
                true_positives / (true_positives + false_positives)
                if true_positives + false_positives
                else 1.0
            ),
            "recall": (
                true_positives / (true_positives + false_negatives)
                if true_positives + false_negatives
                else 1.0
            ),
            "false_positive_rate": (
                false_positives / (true_positives + false_positives)
                if true_positives + false_positives
                else 0.0
            ),
            "mean_processing_seconds": round(
                sum(item["elapsed_seconds"] for item in results) / len(results), 3
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = asyncio.run(evaluate(args.url))
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
