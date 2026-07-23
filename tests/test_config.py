from __future__ import annotations

import app.config as config


def test_positive_int_falls_back_for_invalid_environment(monkeypatch) -> None:
    monkeypatch.setenv("INVALID_LIMIT", "not-an-integer")
    assert config._positive_int("INVALID_LIMIT", 42) == 42
    monkeypatch.setenv("INVALID_LIMIT", "-3")
    assert config._positive_int("INVALID_LIMIT", 42, minimum=5) == 5
