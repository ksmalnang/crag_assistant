"""Tests for IP-027: Post-run ingestion report generator."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ingestion.orchestrator import BatchRunSummary
from ingestion.report import IngestionReport

# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_summary() -> BatchRunSummary:
    """Create a sample BatchRunSummary for testing."""
    return BatchRunSummary(
        run_id="test-run-001",
        started_at="2025-01-01T00:00:00+00:00",
        completed_at="2025-01-01T00:05:00+00:00",
        total_files=10,
        ingested=7,
        skipped=2,
        skipped_unchanged=1,
        skipped_error=1,
        failed=1,
        total_chunks_created=35,
        total_vectors_upserted=35,
        errors=[
            {"file": "bad.pdf", "reason": "parse_error", "node": "parser_node"},
        ],
        duration_seconds=300.0,
    )


@pytest.fixture
def temp_report_dir(tmp_path: Path) -> Path:
    """Create a temporary report directory."""
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    return report_dir


# ─── IngestionReport Tests ─────────────────────────────────────────────────────


class TestIngestionReport:
    """Tests for the IngestionReport class."""

    def test_to_dict_includes_all_fields(self, sample_summary: BatchRunSummary):
        """to_dict should include all required report fields."""
        report = IngestionReport(
            run_id="test-run-001",
            summary=sample_summary,
        )

        d = report.to_dict()

        assert "run_id" in d
        assert "started_at" in d
        assert "completed_at" in d
        assert "duration_seconds" in d
        assert "total_files" in d
        assert "ingested" in d
        assert "skipped" in d
        assert "skipped_unchanged" in d
        assert "skipped_error" in d
        assert "failed" in d
        assert "total_chunks_created" in d
        assert "total_vectors_upserted" in d
        assert "errors" in d
        assert "generated_at" in d

    def test_to_dict_without_summary(self):
        """to_dict should handle missing summary gracefully."""
        report = IngestionReport(run_id="test-run-001")
        d = report.to_dict()

        assert d["run_id"] == "test-run-001"
        assert d["status"] == "no_summary"

    def test_saves_report_to_json(
        self, sample_summary: BatchRunSummary, temp_report_dir: Path
    ):
        """Report should save to JSON file."""
        report = IngestionReport(
            run_id="test-run-001",
            summary=sample_summary,
            report_dir=temp_report_dir,
        )

        report_path = report.save()

        assert report_path.exists()
        assert report_path.suffix == ".json"

        # Verify content
        with open(report_path) as f:
            data = json.load(f)
        assert data["run_id"] == "test-run-001"
        assert data["total_files"] == 10

    def test_load_last_report(
        self, sample_summary: BatchRunSummary, temp_report_dir: Path
    ):
        """Should load the most recent report."""
        # Save two reports
        report1 = IngestionReport(
            run_id="run-001",
            summary=sample_summary,
            report_dir=temp_report_dir,
        )
        report1.save()

        report2_summary = BatchRunSummary(
            run_id="run-002",
            started_at="2025-01-02T00:00:00",
            total_files=5,
            ingested=5,
            skipped=0,
            skipped_unchanged=0,
            skipped_error=0,
            failed=0,
            total_chunks_created=25,
            total_vectors_upserted=25,
            duration_seconds=60.0,
        )
        report2 = IngestionReport(
            run_id="run-002",
            summary=report2_summary,
            report_dir=temp_report_dir,
        )
        report2.save()

        # Load last report
        with patch.object(IngestionReport, "REPORT_DIR", temp_report_dir):
            last = IngestionReport.load_last_report()

        assert last is not None
        assert last["run_id"] == "run-002"

    def test_load_last_report_no_reports(self, temp_report_dir: Path):
        """Should return None when no reports exist."""
        with patch.object(IngestionReport, "REPORT_DIR", temp_report_dir):
            last = IngestionReport.load_last_report()

        assert last is None

    def test_load_specific_report(
        self, sample_summary: BatchRunSummary, temp_report_dir: Path
    ):
        """Should load a specific report by run_id."""
        report = IngestionReport(
            run_id="test-run-001",
            summary=sample_summary,
            report_dir=temp_report_dir,
        )
        report.save()

        with patch.object(IngestionReport, "REPORT_DIR", temp_report_dir):
            loaded = IngestionReport.load_report("test-run-001")

        assert loaded is not None
        assert loaded["run_id"] == "test-run-001"

    def test_load_nonexistent_report(self, temp_report_dir: Path):
        """Should return None for non-existent report."""
        with patch.object(IngestionReport, "REPORT_DIR", temp_report_dir):
            loaded = IngestionReport.load_report("nonexistent-run")

        assert loaded is None

    def test_print_last_report_no_reports(self, capsys, temp_report_dir: Path):
        """Should print message when no reports exist."""
        with patch.object(IngestionReport, "REPORT_DIR", temp_report_dir):
            IngestionReport.print_last_report()

        captured = capsys.readouterr()
        assert "No ingestion reports found" in captured.out

    def test_print_report_not_found(self, capsys, temp_report_dir: Path):
        """Should print message when report not found."""
        with patch.object(IngestionReport, "REPORT_DIR", temp_report_dir):
            IngestionReport.print_report("nonexistent")

        captured = capsys.readouterr()
        assert "Report not found" in captured.out

    def test_print_report_with_errors(
        self, sample_summary: BatchRunSummary, capsys, temp_report_dir: Path
    ):
        """Should print errors section when present."""
        report = IngestionReport(
            run_id="test-run-001",
            summary=sample_summary,
            report_dir=temp_report_dir,
        )
        report.save()

        with patch.object(IngestionReport, "REPORT_DIR", temp_report_dir):
            IngestionReport.print_report("test-run-001")

        captured = capsys.readouterr()
        assert "Errors (1)" in captured.out
        assert "bad.pdf" in captured.out
        assert "parse_error" in captured.out

    def test_report_fields_match_acceptance_criteria(
        self, sample_summary: BatchRunSummary
    ):
        """
        Verify report has all fields from IP-027 acceptance criteria:
        - run_id
        - started_at
        - completed_at
        - duration_seconds
        - total_files
        - ingested
        - skipped
        - failed
        - total_chunks_created
        - total_vectors_upserted
        - errors: list[{file, reason, node}]
        """
        report = IngestionReport(
            run_id="test-run-001",
            summary=sample_summary,
        )
        d = report.to_dict()

        # All required fields
        assert "run_id" in d
        assert "started_at" in d
        assert "completed_at" in d
        assert "duration_seconds" in d
        assert "total_files" in d
        assert "ingested" in d
        assert "skipped" in d
        assert "failed" in d
        assert "total_chunks_created" in d
        assert "total_vectors_upserted" in d
        assert "errors" in d

        # Verify error structure
        assert len(d["errors"]) > 0
        error = d["errors"][0]
        assert "file" in error
        assert "reason" in error
        assert "node" in error
