"""Test fixture loading utilities for ingestion pipeline testing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FIXTURE_DIR = Path(__file__).parent
MANIFEST_PATH = FIXTURE_DIR / "fixture_manifest.json"


def load_fixture(filename: str) -> Path:
    """Get the path to a test fixture file."""
    path = FIXTURE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    return path


def load_manifest() -> dict[str, Any]:
    """Load the fixture manifest JSON."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Fixture manifest not found: {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def get_fixture_info(filename: str) -> dict[str, Any]:
    """Get metadata for a specific fixture from the manifest."""
    manifest = load_manifest()
    for fixture in manifest["fixtures"]:
        if fixture["filename"] == filename:
            return fixture
    raise ValueError(f"Fixture not in manifest: {filename}")


def get_fixtures_by_type(file_type: str) -> list[dict[str, Any]]:
    """Get all fixtures of a given type (happy_path, error_path, edge_case)."""
    manifest = load_manifest()
    return [f for f in manifest["fixtures"] if f["file_type"] == file_type]
