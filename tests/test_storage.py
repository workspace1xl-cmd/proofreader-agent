from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

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
