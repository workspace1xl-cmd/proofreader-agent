from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest

import app.supabase_client as storage
from app.supabase_client import get_job, list_jobs, save_job


def test_concurrent_memory_history_reads_and_writes() -> None:
    result = {
        "corrected_text": "text",
        "changes": [],
        "summary": "clean",
        "stats": {},
    }

    def write(index: int) -> str:
        return str(save_job(f"text {index}", result)["id"])

    with ThreadPoolExecutor(max_workers=8) as executor:
        ids = list(executor.map(write, range(100)))
        snapshots = list(executor.map(lambda _: list_jobs(50), range(20)))

    assert len(ids) == len(set(ids))
    assert all(len(snapshot) <= 50 for snapshot in snapshots)
    assert get_job(ids[-1]) is not None


class Query:
    def __init__(self, data: list[dict[str, Any]]) -> None:
        self.data = data

    def insert(self, row: dict[str, Any]) -> Query:
        self.data = [{"id": "database-id", **row}]
        return self

    def select(self, fields: str) -> Query:
        return self

    def eq(self, field: str, value: str) -> Query:
        return self

    def order(self, field: str, desc: bool = False) -> Query:
        return self

    def limit(self, value: int) -> Query:
        return self

    def execute(self) -> SimpleNamespace:
        return SimpleNamespace(data=self.data)


class Client:
    def __init__(self, data: list[dict[str, Any]]) -> None:
        self.data = data

    def table(self, name: str) -> Query:
        return Query(list(self.data))


def test_database_storage_success_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(storage, "_client", Client([]))
    result = {
        "corrected_text": "correct",
        "changes": [],
        "summary": "done",
        "stats": {},
    }
    assert storage.storage_mode() == "supabase"
    assert save_job("original", result)["id"] == "database-id"

    row = {
        "id": "job-id",
        "original_text": "original",
        "summary": "done",
        "stats": {},
        "created_at": "now",
    }
    monkeypatch.setattr(storage, "_client", Client([row]))
    assert get_job("job-id") == row
    assert list_jobs(1) == [row]


class BrokenClient:
    def table(self, name: str) -> Query:
        raise RuntimeError("database unavailable")


def test_database_storage_failure_falls_back_to_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(storage, "_client", BrokenClient())
    result = {
        "corrected_text": "correct",
        "changes": [],
        "summary": "done",
        "stats": {},
    }
    saved = save_job("original", result)
    assert saved["id"]
    assert get_job(saved["id"]) == saved
    assert list_jobs(1)[0]["id"] == saved["id"]
