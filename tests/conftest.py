"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from pa_triage.logging_utils import redactor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"


@pytest.fixture(autouse=True)
def _clear_redactor():
    """Ensure PII redactor state does not leak between tests."""
    redactor.clear()
    yield
    redactor.clear()


@pytest.fixture
def samples_dir() -> Path:
    return SAMPLES_DIR


@pytest.fixture
def load_sample() -> Callable[[str], dict[str, Any]]:
    """Return a loader that reads a sample bundle by filename stem or name."""

    def _load(name: str) -> dict[str, Any]:
        path = SAMPLES_DIR / name
        if not path.exists() and not name.endswith(".json"):
            path = SAMPLES_DIR / f"{name}.json"
        if not path.exists():
            # allow lookup by stem prefix, e.g. "approve_clean"
            matches = sorted(SAMPLES_DIR.glob(f"*{name}*.json"))
            if matches:
                path = matches[0]
        return json.loads(path.read_text())

    return _load
