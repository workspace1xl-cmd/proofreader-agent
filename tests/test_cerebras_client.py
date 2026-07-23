from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

import app.cerebras_client as cerebras


def _response(request: httpx.Request, status: int, body: Any) -> httpx.Response:
    return httpx.Response(status, request=request, json=body)


def test_parse_response_accepts_fence_and_rejects_nonobjects() -> None:
    assert cerebras._parse_response('```json\n{"ok": true}\n```') == {"ok": True}
    assert cerebras._parse_response('prefix {"ok": true} suffix') == {"ok": True}
    assert cerebras._parse_response("[]") is None
    assert cerebras._parse_response("not json") is None


def test_retry_after_supports_seconds_and_http_dates() -> None:
    assert cerebras._retry_after_seconds("3") == 3
    assert cerebras._retry_after_seconds("invalid") is None
    assert cerebras._retry_after_seconds("Wed, 21 Oct 2037 07:28:00 GMT") > 0


def test_call_json_retries_transient_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cerebras, "CEREBRAS_API_KEY", "test")
    calls = 0
    sleeps: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _response(request, 503, {"error": "busy"})
        return _response(
            request,
            200,
            {"choices": [{"message": {"content": json.dumps({"ok": True})}}]},
        )

    async def no_sleep(delay: float) -> None:
        sleeps.append(delay)

    async def run() -> dict[str, Any] | None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await cerebras.call_json(client, "system", "user", sleep=no_sleep)

    assert asyncio.run(run()) == {"ok": True}
    assert calls == 2 and len(sleeps) == 1


def test_call_json_retries_timeout_then_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cerebras, "CEREBRAS_API_KEY", "test")
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("slow", request=request)

    async def no_sleep(delay: float) -> None:
        return None

    async def run() -> dict[str, Any] | None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await cerebras.call_json(
                client, "system", "user", attempts=3, sleep=no_sleep
            )

    assert asyncio.run(run()) is None
    assert calls == 3


def test_call_json_does_not_retry_nontransient_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cerebras, "CEREBRAS_API_KEY", "test")
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(request, 400, {"error": "bad"})

    async def run() -> dict[str, Any] | None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await cerebras.call_json(client, "system", "user")

    assert asyncio.run(run()) is None
    assert calls == 1
