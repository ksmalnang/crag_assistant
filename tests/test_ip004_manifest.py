"""Unit tests for IP-004: Build Intake Manifest Generator."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from ingestion.manifest import (
    IntakeManifest,
    ManifestEntry,
    generate_intake_manifest,
)


class TestManifestEntry:
    """Test manifest entry creation."""

    def test_basic_entry(self):
        """Test basic entry creation."""
        entry = ManifestEntry(
            file_path="/path/to/file.pdf",
            document_id="doc-123",
            file_size_bytes=1024,
            detected_mime="application/pdf",
            status="queued",
        )

        assert entry.file_path == "/path/to/file.pdf"
        assert entry.document_id == "doc-123"
        assert entry.file_size_bytes == 1024
        assert entry.detected_mime == "application/pdf"
        assert entry.status == "queued"
        assert entry.queued_at is not None
        assert entry.error_reason is None

    def test_entry_with_error(self):
        """Test entry with error information."""
        entry = ManifestEntry(
            file_path="/path/to/file.exe",
            document_id="doc-123",
            file_size_bytes=2048,
            detected_mime="application/x-msdownload",
            status="error",
            error_reason="unsupported_format",
        )

        assert entry.status == "error"
        assert entry.error_reason == "unsupported_format"

    def test_entry_to_dict(self):
        """Test entry serialization to dictionary."""
        entry = ManifestEntry(
            file_path="/path/to/file.pdf",
            document_id="doc-123",
            file_size_bytes=1024,
            detected_mime="application/pdf",
            status="queued",
        )

        data = entry.to_dict()
        assert data["file_path"] == "/path/to/file.pdf"
        assert data["document_id"] == "doc-123"
        assert data["file_size_bytes"] == 1024
        assert data["detected_mime"] == "application/pdf"
        assert data["status"] == "queued"
        assert "queued_at" in data
        assert "error_reason" not in data

    def test_entry_to_dict_with_error(self):
        """Test entry serialization with error reason."""
        entry = ManifestEntry(
            file_path="/path/to/file.exe",
            document_id="doc-123",
            file_size_bytes=2048,
            detected_mime="application/x-msdownload",
            status="error",
            error_reason="unsupported_format",
        )

        data = entry.to_dict()
        assert data["error_reason"] == "unsupported_format"


class TestIntakeManifest:
    """Test intake manifest generation."""

    @pytest.fixture
    def manifest_dir(self, tmp_path):
        """Create temporary manifest directory."""
        return tmp_path / "manifests"

    @pytest.fixture
    def manifest(self, manifest_dir):
        """Create manifest instance."""
        return IntakeManifest(
            run_id="test-run-123",
            manifest_dir=manifest_dir,
        )

    def test_manifest_initialization(self, manifest):
        """Test manifest initialization."""
        assert manifest.run_id == "test-run-123"
        assert len(manifest.entries) == 0
        assert manifest.created_at is not None

    def test_auto_generated_run_id(self):
        """Test auto-generated UUID run_id."""
        manifest = IntakeManifest()
        assert manifest.run_id is not None
        assert len(manifest.run_id) > 0
        # Should be a valid UUID format

    def test_add_queued_entry(self, manifest):
        """Test adding queued entry."""
        entry = manifest.add_queued_entry(
            file_path="/path/file.pdf",
            document_id="doc-1",
            file_size_bytes=1024,
            detected_mime="application/pdf",
        )

        assert len(manifest.entries) == 1
        assert entry.status == "queued"
        assert manifest.entries[0] == entry

    def test_add_skipped_entry(self, manifest):
        """Test adding skipped entry."""
        entry = manifest.add_skipped_entry(
            file_path="/path/file.pdf",
            document_id="doc-1",
            file_size_bytes=1024,
            detected_mime="application/pdf",
        )

        assert entry.status == "skipped"

    def test_add_error_entry(self, manifest):
        """Test adding error entry."""
        entry = manifest.add_error_entry(
            file_path="/path/file.exe",
            document_id="doc-1",
            file_size_bytes=2048,
            detected_mime="application/x-msdownload",
            error_reason="unsupported_format",
        )

        assert entry.status == "error"
        assert entry.error_reason == "unsupported_format"

    def test_manifest_to_dict(self, manifest):
        """Test manifest serialization."""
        manifest.add_queued_entry(
            file_path="/path/file1.pdf",
            document_id="doc-1",
            file_size_bytes=1024,
            detected_mime="application/pdf",
        )
        manifest.add_skipped_entry(
            file_path="/path/file2.pdf",
            document_id="doc-2",
            file_size_bytes=2048,
            detected_mime="application/pdf",
        )

        data = manifest.to_dict()
        assert data["run_id"] == "test-run-123"
        assert data["total_entries"] == 2
        assert len(data["entries"]) == 2
        assert "created_at" in data

    def test_manifest_save(self, manifest, manifest_dir):
        """Test saving manifest to file."""
        manifest.add_queued_entry(
            file_path="/path/file.pdf",
            document_id="doc-1",
            file_size_bytes=1024,
            detected_mime="application/pdf",
        )

        saved_path = manifest.save()

        assert saved_path.exists()
        assert saved_path.name == "test-run-123.json"

        # Verify content
        with open(saved_path) as f:
            data = json.load(f)
        assert data["run_id"] == "test-run-123"
        assert data["total_entries"] == 1

    def test_manifest_save_atomic(self, manifest):
        """Test manifest is written atomically."""
        manifest.add_queued_entry(
            file_path="/path/file.pdf",
            document_id="doc-1",
            file_size_bytes=1024,
            detected_mime="application/pdf",
        )

        manifest.save()

        # Temp file should not exist after successful save
        temp_path = manifest.manifest_dir / f"{manifest.run_id}.tmp"
        assert not temp_path.exists()

    def test_get_summary(self, manifest):
        """Test manifest summary generation."""
        manifest.add_queued_entry(
            file_path="/path/file1.pdf",
            document_id="doc-1",
            file_size_bytes=1024,
            detected_mime="application/pdf",
        )
        manifest.add_queued_entry(
            file_path="/path/file2.pdf",
            document_id="doc-2",
            file_size_bytes=2048,
            detected_mime="application/pdf",
        )
        manifest.add_skipped_entry(
            file_path="/path/file3.pdf",
            document_id="doc-3",
            file_size_bytes=3072,
            detected_mime="application/pdf",
        )
        manifest.add_error_entry(
            file_path="/path/file4.exe",
            document_id="doc-4",
            file_size_bytes=4096,
            detected_mime="application/x-msdownload",
            error_reason="unsupported_format",
        )

        summary = manifest.get_summary()
        assert summary["queued"] == 2
        assert summary["skipped"] == 1
        assert summary["error"] == 1


class TestGenerateIntakeManifest:
    """Test convenience function."""

    def test_generate_manifest(self, tmp_path):
        """Test convenience function generates manifest."""
        manifest_dir = tmp_path / "manifests"
        entries = [
            {
                "file_path": "/path/file1.pdf",
                "document_id": "doc-1",
                "file_size_bytes": 1024,
                "detected_mime": "application/pdf",
                "status": "queued",
            },
            {
                "file_path": "/path/file2.pdf",
                "document_id": "doc-2",
                "file_size_bytes": 2048,
                "detected_mime": "application/pdf",
                "status": "skipped",
            },
            {
                "file_path": "/path/file3.exe",
                "document_id": "doc-3",
                "file_size_bytes": 3072,
                "detected_mime": "application/x-msdownload",
                "status": "error",
                "error_reason": "unsupported_format",
            },
        ]

        saved_path = generate_intake_manifest(
            entries=entries,
            manifest_dir=manifest_dir,
        )

        assert saved_path.exists()
        assert saved_path.suffix == ".json"

        # Verify content
        with open(saved_path) as f:
            data = json.load(f)
        assert data["total_entries"] == 3
        assert len(data["entries"]) == 3

        # Check mixed statuses
        statuses = [e["status"] for e in data["entries"]]
        assert "queued" in statuses
        assert "skipped" in statuses
        assert "error" in statuses

    def test_generate_manifest_with_custom_run_id(self, tmp_path):
        """Test convenience function with custom run_id."""
        manifest_dir = tmp_path / "manifests"
        entries = []

        saved_path = generate_intake_manifest(
            entries=entries,
            run_id="custom-run-456",
            manifest_dir=manifest_dir,
        )

        assert saved_path.name == "custom-run-456.json"


class TestManifestIntegration:
    """Integration tests for manifest workflow."""

    def test_mixed_bag_manifest(self, tmp_path):
        """Test manifest with mixed bag of new, skipped, and rejected files."""
        manifest_dir = tmp_path / "manifests"
        manifest = IntakeManifest(manifest_dir=manifest_dir)

        # Simulate intake decisions
        manifest.add_queued_entry(
            file_path="/watch/new_file.pdf",
            document_id="doc-new",
            file_size_bytes=1024,
            detected_mime="application/pdf",
        )

        manifest.add_skipped_entry(
            file_path="/watch/existing.pdf",
            document_id="doc-existing",
            file_size_bytes=2048,
            detected_mime="application/pdf",
        )

        manifest.add_error_entry(
            file_path="/watch/bad_file.exe",
            document_id="doc-bad",
            file_size_bytes=3072,
            detected_mime="application/x-msdownload",
            error_reason="unsupported_format",
        )

        # Save and verify
        saved_path = manifest.save()

        with open(saved_path) as f:
            data = json.load(f)

        assert data["total_entries"] == 3

        # Verify each entry type
        entries_by_status = {}
        for entry in data["entries"]:
            status = entry["status"]
            entries_by_status[status] = entry

        assert "queued" in entries_by_status
        assert entries_by_status["queued"]["file_path"] == "/watch/new_file.pdf"

        assert "skipped" in entries_by_status
        assert entries_by_status["skipped"]["file_path"] == "/watch/existing.pdf"

        assert "error" in entries_by_status
        assert entries_by_status["error"]["error_reason"] == "unsupported_format"

    def test_manifest_reflects_intake_decisions(self, tmp_path):
        """Test manifest reflects intake decisions, not parse results."""
        manifest_dir = tmp_path / "manifests"
        manifest = IntakeManifest(manifest_dir=manifest_dir)

        # All entries should reflect intake-level decisions only
        # (format validation, size checks, etc.) - not parsing results

        manifest.add_queued_entry(
            file_path="/watch/valid.pdf",
            document_id="doc-1",
            file_size_bytes=1024,
            detected_mime="application/pdf",
        )

        manifest.add_error_entry(
            file_path="/watch/empty.pdf",
            document_id="doc-2",
            file_size_bytes=0,
            detected_mime="application/pdf",
            error_reason="empty_file",
        )

        saved_path = manifest.save()

        with open(saved_path) as f:
            data = json.load(f)

        # Manifest should show what was queued vs rejected at intake time
        assert data["entries"][0]["status"] == "queued"
        assert data["entries"][1]["status"] == "error"
        assert data["entries"][1]["error_reason"] == "empty_file"
