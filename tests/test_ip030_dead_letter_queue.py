"""Tests for IP-030: Dead letter queue."""

import json
from pathlib import Path

import pytest

from ingestion.dead_letter import DeadLetterQueue
from ingestion.errors_base import IngestionError, UpsertNodeError

# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def temp_dlq_dir(tmp_path: Path) -> Path:
    """Create a temporary dead letter directory."""
    return tmp_path / "dead_letter"


@pytest.fixture
def dlq(temp_dlq_dir: Path) -> DeadLetterQueue:
    """Create a DeadLetterQueue with temporary directory."""
    return DeadLetterQueue(base_dir=temp_dlq_dir)


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """Create a sample file for dead letter testing."""
    f = tmp_path / "engineering_2025-S1_lecture_test.pdf"
    f.write_bytes(b"%PDF-1.4\nSample content\n")
    return f


# ─── DeadLetterQueue Tests ────────────────────────────────────────────────────


class TestDeadLetterQueue:
    """Tests for the DeadLetterQueue class."""

    def test_store_creates_files(self, dlq: DeadLetterQueue, sample_file: Path):
        """store should create both the file copy and error JSON."""
        error = IngestionError(
            document_id="doc-123",
            file_path=str(sample_file),
            node="upsert_node",
            reason="connection_failed",
            message="Failed to connect to Qdrant",
        )

        error_path = dlq.store(
            run_id="run-001",
            file_path=str(sample_file),
            error=error,
        )

        assert error_path.exists()
        assert error_path.suffix == ".json"

        # File should also exist (remove .error.json suffix)
        file_path = error_path.parent / error_path.name.replace(".error.json", "")
        assert file_path.exists()

    def test_store_error_json_is_valid(self, dlq: DeadLetterQueue, sample_file: Path):
        """Error JSON should be valid and contain all fields."""
        error = UpsertNodeError(
            file_path=str(sample_file),
            reason="batch_failed",
            batch_index=2,
            retry_count=3,
        )

        error_path = dlq.store(
            run_id="run-001",
            file_path=str(sample_file),
            error=error,
        )

        data = json.loads(error_path.read_text())
        assert data["node"] == "upsert_node"
        assert data["reason"] == "batch_failed"
        assert data["failed_at"] is not None
        assert data["original_filename"] == sample_file.name

    def test_list_entries(
        self, dlq: DeadLetterQueue, sample_file: Path, tmp_path: Path
    ):
        """list_entries should return all dead letter entries."""
        # Create two different files for two entries
        f1 = tmp_path / "engineering_2025-S1_lecture_a.pdf"
        f2 = tmp_path / "engineering_2025-S1_lecture_b.pdf"
        f1.write_bytes(b"%PDF-1.4\nContent A\n")
        f2.write_bytes(b"%PDF-1.4\nContent B\n")

        error1 = IngestionError(
            file_path=str(f1),
            node="intake_node",
            reason="empty_file",
        )
        error2 = IngestionError(
            file_path=str(f2),
            node="parser_node",
            reason="corrupted",
        )
        dlq.store(run_id="run-001", file_path=str(f1), error=error1)
        dlq.store(run_id="run-001", file_path=str(f2), error=error2)

        entries = dlq.list_entries(run_id="run-001")
        assert len(entries) == 2

    def test_list_entries_all_runs(self, dlq: DeadLetterQueue, sample_file: Path):
        """list_entries without run_id should list all runs."""
        error = IngestionError(
            file_path=str(sample_file),
            node="intake_node",
            reason="empty_file",
        )
        dlq.store(run_id="run-001", file_path=str(sample_file), error=error)
        dlq.store(run_id="run-002", file_path=str(sample_file), error=error)

        entries = dlq.list_entries()
        assert len(entries) == 2

    def test_get_entry(self, dlq: DeadLetterQueue, sample_file: Path):
        """get_entry should return specific entry by error file path."""
        error = IngestionError(
            file_path=str(sample_file),
            node="parser_node",
            reason="corrupted",
        )
        error_path = dlq.store(
            run_id="run-001", file_path=str(sample_file), error=error
        )

        entry = dlq.get_entry(str(error_path))
        assert entry is not None
        assert entry["node"] == "parser_node"

    def test_get_entry_not_found(self, dlq: DeadLetterQueue):
        """get_entry should return None for non-existent entry."""
        assert dlq.get_entry("/nonexistent/path.error.json") is None

    def test_remove_entry(self, dlq: DeadLetterQueue, sample_file: Path):
        """remove_entry should delete both file and error JSON."""
        error = IngestionError(
            file_path=str(sample_file),
            node="intake_node",
            reason="empty_file",
        )
        error_path = dlq.store(
            run_id="run-001", file_path=str(sample_file), error=error
        )
        result = dlq.remove_entry(str(error_path))
        assert result is True

        # Both files should be gone
        assert not error_path.exists()
        assert not error_path.with_suffix("").exists()

    def test_remove_run(self, dlq: DeadLetterQueue, sample_file: Path):
        """remove_run should delete entire run directory."""
        error = IngestionError(
            file_path=str(sample_file),
            node="intake_node",
            reason="empty_file",
        )
        dlq.store(run_id="run-001", file_path=str(sample_file), error=error)
        dlq.store(run_id="run-001", file_path=str(sample_file), error=error)

        result = dlq.remove_run("run-001")
        assert result is True

        # Run directory should be gone
        run_dir = dlq._run_dir("run-001")
        assert not run_dir.exists()

    def test_remove_nonexistent_run(self, dlq: DeadLetterQueue):
        """remove_run should return False for non-existent run."""
        assert dlq.remove_run("nonexistent") is False

    def test_empty_list(self, dlq: DeadLetterQueue):
        """list_entries should return empty list for no entries."""
        assert dlq.list_entries(run_id="empty-run") == []
