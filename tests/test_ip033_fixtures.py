"""Tests for IP-033: Ingestion test fixture library.

Validates that all fixtures exist, are properly formatted,
and match the fixture manifest expectations.
"""


import pytest

from tests.fixtures.ingestion import (
    FIXTURE_DIR,
    MANIFEST_PATH,
    get_fixture_info,
    get_fixtures_by_type,
    load_fixture,
    load_manifest,
)

# ─── Manifest Tests ───────────────────────────────────────────────────────────


class TestFixtureManifest:
    """Tests for the fixture manifest itself."""

    def test_manifest_exists(self):
        """Manifest file should exist."""
        assert MANIFEST_PATH.exists()

    def test_manifest_valid_json(self):
        """Manifest should be valid JSON."""
        manifest = load_manifest()
        assert "fixtures" in manifest
        assert "version" in manifest

    def test_all_manifested_fixtures_exist(self):
        """Every fixture listed in manifest should exist on disk."""
        manifest = load_manifest()
        for fixture in manifest["fixtures"]:
            fixture_path = FIXTURE_DIR / fixture["filename"]
            assert fixture_path.exists(), f"Missing fixture: {fixture['filename']}"

    def test_manifest_has_required_fields(self):
        """Each manifest entry should have required metadata fields."""
        manifest = load_manifest()
        required_fields = {
            "filename",
            "description",
            "tests",
            "expected_chunks",
            "expected_headings",
            "file_type",
        }
        for fixture in manifest["fixtures"]:
            for field in required_fields:
                assert field in fixture, (
                    f"Missing field '{field}' in {fixture['filename']}"
                )


# ─── Fixture Loading Tests ────────────────────────────────────────────────────


class TestFixtureLoading:
    """Tests for fixture loading utilities."""

    def test_load_fixture_returns_path(self):
        """load_fixture should return a valid Path."""
        path = load_fixture("minimal_valid.pdf")
        assert path.exists()
        assert path.suffix == ".pdf"

    def test_load_fixture_raises_for_missing(self):
        """load_fixture should raise FileNotFoundError for missing fixtures."""
        with pytest.raises(FileNotFoundError, match="Fixture not found"):
            load_fixture("nonexistent.pdf")

    def test_get_fixture_info_returns_metadata(self):
        """get_fixture_info should return metadata for a fixture."""
        info = get_fixture_info("minimal_valid.pdf")
        assert info["filename"] == "minimal_valid.pdf"
        assert "description" in info
        assert "file_type" in info

    def test_get_fixtures_by_type_returns_filtered(self):
        """get_fixtures_by_type should return only matching fixtures."""
        happy = get_fixtures_by_type("happy_path")
        assert len(happy) > 0
        assert all(f["file_type"] == "happy_path" for f in happy)

    def test_get_fixtures_by_type_empty_for_unknown(self):
        """get_fixtures_by_type should return empty list for unknown type."""
        assert get_fixtures_by_type("nonexistent_type") == []


# ─── Fixture Content Validation ──────────────────────────────────────────────


class TestFixtureContent:
    """Tests that validate fixture file contents match manifest expectations."""

    def test_minimal_valid_pdf_is_valid_pdf(self):
        """minimal_valid.pdf should start with PDF header."""
        path = load_fixture("minimal_valid.pdf")
        content = path.read_bytes()
        assert content.startswith(b"%PDF-")

    def test_oversized_pdf_exceeds_limit(self):
        """oversized.pdf should exceed 50MB."""
        path = load_fixture("oversized.pdf")
        size = path.stat().st_size
        assert size > 50 * 1024 * 1024  # 50MB

    def test_corrupted_docx_is_zip(self):
        """corrupted.docx should be a valid ZIP (but with garbage content)."""
        path = load_fixture("corrupted.docx")
        content = path.read_bytes()
        # ZIP magic number
        assert content[:4] == b"PK\x03\x04"

    def test_valid_docx_is_zip(self):
        """valid.docx should be a valid ZIP file."""
        path = load_fixture("valid.docx")
        content = path.read_bytes()
        assert content[:4] == b"PK\x03\x04"

    def test_valid_pptx_is_zip(self):
        """valid.pptx should be a valid ZIP file."""
        path = load_fixture("valid.pptx")
        content = path.read_bytes()
        assert content[:4] == b"PK\x03\x04"

    def test_encrypted_pdf_has_encrypt_marker(self):
        """encrypted.pdf should contain /Encrypt marker."""
        path = load_fixture("encrypted.pdf")
        content = path.read_text(errors="ignore")
        assert "/Encrypt" in content


# ─── Error Path Fixture Tests ────────────────────────────────────────────────


class TestErrorPathFixtures:
    """Tests for error path fixtures."""

    def test_error_fixtures_exist(self):
        """Error path fixtures should exist."""
        error_fixtures = get_fixtures_by_type("error_path")
        assert len(error_fixtures) >= 2  # At least corrupted and encrypted

    def test_corrupted_fixture_fails_preflight(self):
        """corrupted.docx should fail content validation."""
        path = load_fixture("corrupted.docx")
        path.read_bytes()
        # Corrupted docx has no word/document.xml
        import zipfile

        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
        assert "word/document.xml" not in names

    def test_oversized_fixture_triggers_size_check(self):
        """oversized.pdf should trigger size limit check."""
        path = load_fixture("oversized.pdf")
        size = path.stat().st_size
        limit = 50 * 1024 * 1024  # 50MB
        assert size > limit


# ─── Happy Path Fixture Tests ────────────────────────────────────────────────


class TestHappyPathFixtures:
    """Tests for happy path fixtures."""

    def test_happy_fixtures_exist(self):
        """Happy path fixtures should exist."""
        happy_fixtures = get_fixtures_by_type("happy_path")
        assert len(happy_fixtures) >= 3

    def test_all_fixtures_are_non_empty(self):
        """All fixture files should have content."""
        manifest = load_manifest()
        for fixture in manifest["fixtures"]:
            path = FIXTURE_DIR / fixture["filename"]
            assert path.stat().st_size > 0, f"Empty fixture: {fixture['filename']}"
