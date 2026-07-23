"""Compatibility entry point: run the discoverable pytest suite."""

from __future__ import annotations

import sys

import pytest

if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", *sys.argv[1:]]))
