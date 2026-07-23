from __future__ import annotations

import asyncio
from typing import Any

import pytest

import app.agents as agents


def test_doc_agent_normalizes_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses: list[dict[str, Any] | None] = [
        {"findings": "wrong shape", "verdict": 42},
        None,
    ]

    async def fake_call(*args: Any, **kwargs: Any) -> dict[str, Any] | None:
        return responses.pop(0)

    monkeypatch.setattr(agents, "call_json", fake_call)
    agent = {"prompt": "review"}
    success = asyncio.run(agents.run_doc_agent(object(), agent, "text"))
    failure = asyncio.run(agents.run_doc_agent(object(), agent, "text"))
    assert success == {"findings": [], "verdict": "42", "failed": False}
    assert failure["failed"] is True


def test_verifier_batches_every_item(monkeypatch: pytest.MonkeyPatch) -> None:
    batches: list[int] = []

    async def fake_call(*args: Any, **kwargs: Any) -> dict[str, Any]:
        listing = args[2]
        ids = [
            line.split(".", 1)[0]
            for line in listing.splitlines()
            if line.startswith("c")
        ]
        batches.append(len(ids))
        return {
            "items": [
                {"id": item_id, "keep": True, "confidence": 0.8} for item_id in ids
            ]
        }

    monkeypatch.setattr(agents, "call_json", fake_call)
    items = [
        {"id": f"c{i}", "kind": "grammar", "text": "x", "context": "x"}
        for i in range(95)
    ]
    result = asyncio.run(agents.run_verifier(object(), items, batch_size=40))
    assert batches == [40, 40, 15]
    assert len(result) == 95


def test_verifier_does_not_coerce_string_boolean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_call(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"items": [{"id": "c0", "keep": "false", "confidence": "0.7"}]}

    monkeypatch.setattr(agents, "call_json", fake_call)
    items = [{"id": "c0", "kind": "grammar", "text": "x", "context": "x"}]
    result = asyncio.run(agents.run_verifier(object(), items))
    assert result["c0"]["keep"] is False
    assert result["c0"]["confidence"] == 0.7


def test_verifier_validates_evidence_rule_confidence_and_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_call(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "items": [
                {
                    "id": "c0",
                    "keep": True,
                    "confidence": "invalid",
                    "rule_valid": True,
                    "evidence_valid": True,
                },
                {"id": "unknown", "keep": True, "confidence": 1},
                "invalid",
            ]
        }

    monkeypatch.setattr(agents, "call_json", fake_call)
    items = [{"id": "c0", "kind": "grammar", "text": "x", "context": "x"}]
    result = asyncio.run(agents.run_verifier(object(), items))
    assert result["c0"]["keep"] is True
    assert result["c0"]["confidence"] == 0.5
    assert "unknown" not in result


def test_verifier_handles_missing_items_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_call(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"items": "invalid"}

    monkeypatch.setattr(agents, "call_json", fake_call)
    items = [{"id": "c0", "kind": "grammar", "text": "x", "context": "x"}]
    assert asyncio.run(agents.run_verifier(object(), items)) == {}


def test_summary_success_normalizes_model_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_call(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "summary": "Release after fixes.",
            "risk_level": "high",
            "top_issues": ["One", 2, "Three", "Four", "Five", "Six"],
            "readability": 150,
        }

    monkeypatch.setattr(agents, "call_json", fake_call)
    result = asyncio.run(
        agents.run_summary(
            object(),
            {"words": 10, "report": "excluded"},
            {"overall": 95},
            [{"title": "x", "severity": "minor", "agent": "role"}],
        )
    )
    assert result["risk_level"] == "low"
    assert result["readability"] == 100
    assert result["top_issues"] == ["One", "2", "Three", "Four", "Five"]


def test_summary_failure_and_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses: list[dict[str, Any] | None] = [
        None,
        {"summary": "", "top_issues": "wrong", "readability": "wrong"},
    ]

    async def fake_call(*args: Any, **kwargs: Any) -> dict[str, Any] | None:
        return responses.pop(0)

    monkeypatch.setattr(agents, "call_json", fake_call)
    failed = asyncio.run(agents.run_summary(object(), {}, {"overall": 50}, []))
    invalid = asyncio.run(agents.run_summary(object(), {}, {"overall": 75}, []))
    assert failed["failed"] is True and failed["risk_level"] == "high"
    assert invalid["readability"] is None
    assert invalid["top_issues"] == []


@pytest.mark.parametrize(
    ("score", "expected"),
    [(95, "low"), (75, "medium"), (20, "high")],
)
def test_fallback_risk(score: int, expected: str) -> None:
    assert agents._fallback_risk({"overall": score}) == expected
