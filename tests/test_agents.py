from __future__ import annotations

import asyncio
from typing import Any

import pytest

import app.agents as agents


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
    assert result["c0"]["keep"] is None
    assert result["c0"]["confidence"] == 0.7


@pytest.mark.parametrize(
    ("score", "expected"),
    [(95, "low"), (75, "medium"), (20, "high")],
)
def test_fallback_risk(score: int, expected: str) -> None:
    assert agents._fallback_risk({"overall": score}) == expected
