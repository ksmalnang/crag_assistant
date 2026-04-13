"""Unit tests for IP-002: Build Local Folder Watcher Intake Node."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion.ledger import IngestionLedger
from ingestion.watcher import (
    FileChangeEvent,
    FolderWatcher,
    IntakeFileHandler,
    compute_file_hash,
)


class TestComputeFileHash:
    """Test SHA256 content hashing."""

    def test_hash_consistency(self, tmp_path):
        """Test that same content produces same hash."""
        file1 = tmp_path / "test1.txt"
        file1.write_text("Hello, world!")

        hash1 = compute_file_hash(file1)
        hash2 = compute_file_hash(file1)

        assert hash1 == hash2

    def test_hash_uniqueness(self, tmp_path):
        """Test that different content produces different hashes."""
        file1 = tmp_path / "test1.txt"
        file2 = tmp_path / "test2.txt"
        file1.write_text("Hello, world!")
        file2.write_text("Goodbye, world!")

        hash1 = compute_file_hash(file1)
        hash2 = compute_file_hash(file2)

        assert hash1 != hash2

    def test_hash_large_file(self, tmp_path):
        """Test hashing large files."""
        large_file = tmp_path / "large.bin"
        large_file.write_bytes(os.urandom(1024 * 1024))  # 1MB

        hash_value = compute_file_hash(large_file)
        assert len(hash_value) == 64  # SHA256 hex length


class TestIngestionLedger:
    """Test SQLite ledger functionality."""

    @pytest.fixture
    def ledger(self, tmp_path):
        """Create a temporary ledger for testing."""
        db_path = tmp_path / "test_ledger.db"
        return IngestionLedger(db_path=db_path)

    def test_upsert_and_get_entry(self, ledger):
        """Test inserting and retrieving ledger entry."""
        ledger.upsert_entry(
            file_path="/path/to/file.pdf",
            document_id="doc-123",
            content_hash="abc123",
            status="pending",
        )

        entry = ledger.get_entry("/path/to/file.pdf")
        assert entry is not None
        assert entry["document_id"] == "doc-123"
        assert entry["content_hash"] == "abc123"
        assert entry["status"] == "pending"

    def test_upsert_update(self, ledger):
        """Test upsert updates existing entry."""
        ledger.upsert_entry(
            file_path="/path/to/file.pdf",
            document_id="doc-123",
            content_hash="hash1",
            status="pending",
        )

        ledger.upsert_entry(
            file_path="/path/to/file.pdf",
            document_id="doc-123",
            content_hash="hash2",
            status="ingested",
        )

        entry = ledger.get_entry("/path/to/file.pdf")
        assert entry["content_hash"] == "hash2"
        assert entry["status"] == "ingested"

    def test_mark_ingested(self, ledger):
        """Test marking file as ingested."""
        ledger.mark_ingested(
            file_path="/path/to/file.pdf",
            document_id="doc-123",
            content_hash="abc123",
        )

        entry = ledger.get_entry("/path/to/file.pdf")
        assert entry["status"] == "ingested"
        assert entry["ingested_at"] is not None

    def test_flag_for_reingestion(self, ledger):
        """Test flagging file for re-ingestion."""
        ledger.upsert_entry(
            file_path="/path/to/file.pdf",
            document_id="doc-123",
            content_hash="abc123",
            status="ingested",
        )

        ledger.flag_for_reingestion("/path/to/file.pdf")

        entry = ledger.get_entry("/path/to/file.pdf")
        assert entry["status"] == "reingest_pending"

    def test_delete_entry(self, ledger):
        """Test deleting ledger entry."""
        ledger.upsert_entry(
            file_path="/path/to/file.pdf",
            document_id="doc-123",
            content_hash="abc123",
        )

        result = ledger.delete_entry("/path/to/file.pdf")
        assert result is True

        entry = ledger.get_entry("/path/to/file.pdf")
        assert entry is None

    def test_delete_nonexistent_entry(self, ledger):
        """Test deleting nonexistent entry."""
        result = ledger.delete_entry("/nonexistent/file.pdf")
        assert result is False

    def test_get_all_entries(self, ledger):
        """Test getting all entries."""
        ledger.upsert_entry("/path/file1.pdf", "doc-1", "hash1", "pending")
        ledger.upsert_entry("/path/file2.pdf", "doc-2", "hash2", "ingested")
        ledger.upsert_entry("/path/file3.pdf", "doc-3", "hash3", "pending")

        all_entries = ledger.get_all_entries()
        assert len(all_entries) == 3

        pending_entries = ledger.get_all_entries(status="pending")
        assert len(pending_entries) == 2

    def test_get_entries_for_hash_check(self, ledger):
        """Test getting entries for hash comparison."""
        ledger.upsert_entry("/path/file1.pdf", "doc-1", "hash1")
        ledger.upsert_entry("/path/file2.pdf", "doc-2", "hash2")

        hash_dict = ledger.get_entries_for_hash_check()
        assert hash_dict["/path/file1.pdf"] == "hash1"
        assert hash_dict["/path/file2.pdf"] == "hash2"


class TestFileChangeEvent:
    """Test FileChangeEvent class."""

    def test_new_file_event(self):
        """Test new file event creation."""
        event = FileChangeEvent(
            file_path="/path/to/file.pdf",
            event_type="new",
            content_hash="abc123",
            document_id="doc-123",
        )

        assert event.file_path == "/path/to/file.pdf"
        assert event.event_type == "new"
        assert event.document_id == "doc-123"

    def test_auto_generated_document_id(self):
        """Test auto-generated document ID."""
        event = FileChangeEvent(
            file_path="/path/to/file.pdf",
            event_type="new",
        )

        assert event.document_id is not None
        assert len(event.document_id) > 0

    def test_event_repr(self):
        """Test event string representation."""
        event = FileChangeEvent("/path/to/file.pdf", "new")
        assert "new" in repr(event)
        assert "/path/to/file.pdf" in repr(event)


class TestIntakeFileHandler:
    """Test watchdog file handler."""

    @pytest.fixture
    def ledger(self, tmp_path):
        """Create a temporary ledger."""
        db_path = tmp_path / "test_ledger.db"
        return IngestionLedger(db_path=db_path)

    @pytest.fixture
    def handler(self, ledger):
        """Create file handler instance."""
        return IntakeFileHandler(
            ledger=ledger,
            supported_extensions={".pdf", ".txt", ".docx"},
        )

    def test_supported_file_check(self, handler):
        """Test file extension checking."""
        assert handler._is_supported_file("/path/to/file.pdf") is True
        assert handler._is_supported_file("/path/to/file.txt") is True
        assert handler._is_supported_file("/path/to/file.exe") is False

    def test_process_new_file(self, handler, ledger, tmp_path):
        """Test processing a new file."""
        new_file = tmp_path / "new_file.pdf"
        new_file.write_text("Test content")

        handler._process_file(str(new_file), "new")

        # Check event was created
        assert len(handler.events) == 1
        event = handler.events[0]
        assert event.event_type == "new"

        # Check ledger entry
        entry = ledger.get_entry(str(new_file))
        assert entry is not None
        assert entry["status"] == "pending"

    def test_process_unchanged_file(self, handler, ledger, tmp_path):
        """Test processing unchanged file (should skip)."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("Test content")

        # First processing - new file
        handler._process_file(str(test_file), "new")
        handler.events.clear()

        # Second processing - unchanged
        handler._process_file(str(test_file), "modified")

        # Should be marked as unchanged
        assert len(handler.events) == 1
        assert handler.events[0].event_type == "unchanged"

    def test_process_modified_file(self, handler, ledger, tmp_path):
        """Test processing modified file (content change)."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("Original content")

        # First processing
        handler._process_file(str(test_file), "new")
        handler.events.clear()

        # Modify file
        test_file.write_text("Modified content")

        # Second processing - should detect change
        handler._process_file(str(test_file), "modified")

        assert len(handler.events) == 1
        assert handler.events[0].event_type == "modified"


class TestFolderWatcher:
    """Test folder watcher functionality."""

    @pytest.fixture
    def watch_dir(self, tmp_path):
        """Create temporary watch directory."""
        return tmp_path / "watch"

    @pytest.fixture
    def ledger(self, tmp_path):
        """Create temporary ledger."""
        db_path = tmp_path / "test_ledger.db"
        return IngestionLedger(db_path=db_path)

    @pytest.fixture
    def watcher(self, watch_dir, ledger):
        """Create folder watcher instance."""
        return FolderWatcher(
            watch_dir=watch_dir,
            ledger=ledger,
            supported_extensions={".pdf", ".txt"},
        )

    def test_scan_new_file(self, watcher, watch_dir):
        """Test scanning new file."""
        # Create file in watch directory
        test_file = watch_dir / "new_file.pdf"
        test_file.write_text("Test content")

        events = watcher.scan()

        assert len(events) == 1
        assert events[0].event_type == "new"

    def test_scan_unchanged_file(self, watcher, watch_dir, ledger):
        """Test scanning unchanged file."""
        test_file = watch_dir / "test.pdf"
        test_file.write_text("Test content")

        # First scan
        watcher.scan()

        # Second scan
        events = watcher.scan()

        # Should report as unchanged
        assert len(events) == 1
        assert events[0].event_type == "unchanged"

    def test_scan_modified_file(self, watcher, watch_dir):
        """Test scanning modified file."""
        test_file = watch_dir / "test.pdf"
        test_file.write_text("Original content")

        # First scan
        watcher.scan()

        # Modify file
        test_file.write_text("Modified content")

        # Second scan
        events = watcher.scan()

        assert len(events) == 1
        assert events[0].event_type == "modified"

    def test_scan_deleted_file(self, watcher, watch_dir, ledger):
        """Test scanning detects deleted files."""
        test_file = watch_dir / "test.pdf"
        test_file.write_text("Test content")

        # First scan - add to ledger
        watcher.scan()

        # Delete file
        test_file.unlink()

        # Second scan - should detect deletion
        events = watcher.scan()

        assert len(events) == 1
        assert events[0].event_type == "deleted"

    def test_scan_recursive(self, watcher, watch_dir):
        """Test recursive scanning of subdirectories."""
        # Create nested file
        subdir = watch_dir / "subdir"
        subdir.mkdir()
        nested_file = subdir / "nested.pdf"
        nested_file.write_text("Nested content")

        events = watcher.scan()

        assert len(events) == 1
        assert events[0].event_type == "new"

    def test_context_manager(self, watcher):
        """Test context manager usage."""
        with watcher:
            # Watcher should start
            assert watcher._observer is not None

        # Watcher should stop
        assert watcher._observer is None
