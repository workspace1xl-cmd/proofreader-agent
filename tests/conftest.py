from __future__ import annotations

import pytest

from app.supabase_client import clear_memory_for_tests


@pytest.fixture(autouse=True)
def clean_memory_history() -> None:
    clear_memory_for_tests()
