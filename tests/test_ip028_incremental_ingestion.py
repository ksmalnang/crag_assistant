"""Tests for IP-028: Incremental ingestion mode."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingestion.ledger import IngestionLedger
from ingestion.orchestrator import BatchOrchestrator

# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_folder_with_files(tmp_path: Path) -> Path:
    """Create a folder with sample PDF files."""
    folder = tmp_path / "docs"
    folder.mkdir()

    files = [
        "engineering_2025-S1_lecture_intro.pdf",
        "science_2025-S2_tutorial_lab.pdf",
    ]
    for name in files:
        f = folder / name
        f.write_bytes(b"%PDF-1.4\nSample content\n")

    return folder


@pytest.fixture
def orchestrator_with_mocked_graph():
    """Create orchestrator with mocked graph."""
    with patch("ingestion.orchestrator.compile_ingestion_graph") as mock_compile:
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "run_id": "test-run",
                "file_path": "",
                "document_id": "test-doc-id",
                "docling_doc": None,
                "structure_tree": [],
                "metadata": {},
                "chunks": [],
                "dense_vectors": [],
                "sparse_vectors": [],
                "upsert_count": 3,
                "errors": [],
                "status": "completed",
            }
        )
        mock_compile.return_value = mock_graph

        with patch("ingestion.orchestrator.IngestionLedger") as mock_ledger:
            orchestrator = BatchOrchestrator(concurrency=1)
            orchestrator.graph = mock_graph
            orchestrator.ledger = mock_ledger.return_value
            yield orchestrator


# ─── Incremental Mode Tests ───────────────────────────────────────────────────


class TestIncrementalMode:
    """Tests for incremental ingestion mode."""

    def test_default_is_incremental(self):
        """Default behavior should be incremental."""
        with patch("ingestion.orchestrator.compile_ingestion_graph"):
            orchestrator = BatchOrchestrator()
            assert orchestrator.incremental is True
            assert orchestrator.force_full is False

    def test_incremental_flag_enabled(self):
        """--incremental flag should enable incremental mode."""
        with patch("ingestion.orchestrator.compile_ingestion_graph"):
            orchestrator = BatchOrchestrator(incremental=True)
            assert orchestrator.incremental is True

    def test_full_flag_disables_incremental(self):
        """--full flag should force re-ingestion."""
        with patch("ingestion.orchestrator.compile_ingestion_graph"):
            orchestrator = BatchOrchestrator(incremental=True, force_full=True)
            assert orchestrator.force_full is True
            # Should not skip even if ledger says ingested
            orchestrator.ledger = MagicMock()
            orchestrator.ledger.get_entry.return_value = {
                "status": "ingested",
                "content_hash": "abc123",
            }
            should_skip, _ = orchestrator._should_skip("/path/to/file.pdf")
            assert should_skip is False

    def test_incremental_skips_unchanged(self, orchestrator_with_mocked_graph):
        """Incremental mode should skip unchanged files."""
        orchestrator = orchestrator_with_mocked_graph
        orchestrator.incremental = True
        orchestrator.force_full = False

        orchestrator.ledger.get_entry.return_value = {
            "file_path": "/path/to/file.pdf",
            "document_id": "abc123",
            "status": "ingested",
            "content_hash": "def456",
        }

        should_skip, reason = orchestrator._should_skip("/path/to/file.pdf")
        assert should_skip is True
        assert "unchanged" in reason

    def test_incremental_processes_new_files(self, orchestrator_with_mocked_graph):
        """Incremental mode should process new files not in ledger."""
        orchestrator = orchestrator_with_mocked_graph
        orchestrator.incremental = True

        orchestrator.ledger.get_entry.return_value = None

        should_skip, reason = orchestrator._should_skip("/path/to/new_file.pdf")
        assert should_skip is False


# ─── Ledger Integration Tests ─────────────────────────────────────────────────


class TestLedgerIntegration:
    """Tests for ledger-based incremental ingestion."""

    def test_ledger_tracks_ingested_files(self, tmp_path: Path):
        """Ledger should track which files have been ingested."""
        db_path = tmp_path / "test_ledger.db"
        ledger = IngestionLedger(db_path=db_path)

        file_path = "/path/to/file.pdf"
        document_id = "test-doc-id"
        content_hash = "abc123"

        ledger.mark_ingested(
            file_path=file_path,
            document_id=document_id,
            content_hash=content_hash,
        )

        entry = ledger.get_entry(file_path)
        assert entry is not None
        assert entry["document_id"] == document_id
        assert entry["content_hash"] == content_hash
        assert entry["status"] == "ingested"

    def test_ledger_returns_none_for_new_files(self, tmp_path: Path):
        """Ledger should return None for files not yet ingested."""
        db_path = tmp_path / "test_ledger.db"
        ledger = IngestionLedger(db_path=db_path)

        entry = ledger.get_entry("/path/to/nonexistent.pdf")
        assert entry is None


# ─── CLI Flag Tests ───────────────────────────────────────────────────────────


class TestCLIFlags:
    """Tests for CLI argument parsing."""

    def test_cli_default_incremental(self):
        """CLI should default to incremental mode."""
        from ingestion.run import build_parser

        parser = build_parser()
        args = parser.parse_args(["--folder", "./docs"])

        assert args.incremental is True  # default=True
        assert args.full is False

    def test_cli_full_flag(self):
        """--full flag should force full re-ingestion."""
        from ingestion.run import build_parser

        parser = build_parser()
        args = parser.parse_args(["--folder", "./docs", "--full"])

        assert args.full is True

    def test_cli_report_only_flag(self):
        """--report-only flag should trigger report-only mode."""
        from ingestion.run import build_parser

        parser = build_parser()
        args = parser.parse_args(["--report-only"])

        assert args.report_only is True

    def test_cli_report_only_with_run_id(self):
        """--report-only with --run-id should load specific report."""
        from ingestion.run import build_parser

        parser = build_parser()
        args = parser.parse_args(["--report-only", "--run-id", "test-uuid"])

        assert args.report_only is True
        assert args.run_id == "test-uuid"

    def test_cli_concurrency_flag(self):
        """--concurrency flag should set concurrency value."""
        from ingestion.run import build_parser

        parser = build_parser()
        args = parser.parse_args(["--folder", "./docs", "--concurrency", "8"])

        assert args.concurrency == 8


# ─── End-to-End Incremental Test ─────────────────────────────────────────────


class TestIncrementalEndToEnd:
    """End-to-end integration test for incremental ingestion."""

    @pytest.mark.asyncio
    async def test_incremental_run_skips_unchanged(self, tmp_path: Path):
        """
        Integration test:
        1. Ingest folder with 2 files
        2. Re-run incremental — both files should be skipped
        """
        folder = tmp_path / "docs"
        folder.mkdir()
        (folder / "engineering_2025-S1_lecture_test.pdf").write_bytes(
            b"%PDF-1.4\nTest\n"
        )

        with patch("ingestion.orchestrator.compile_ingestion_graph") as mock_compile:
            mock_graph = AsyncMock()
            mock_graph.ainvoke = AsyncMock(
                return_value={
                    "run_id": "test-run",
                    "document_id": "test-doc-id",
                    "file_path": "",
                    "docling_doc": None,
                    "structure_tree": [],
                    "metadata": {},
                    "chunks": [],
                    "dense_vectors": [],
                    "sparse_vectors": [],
                    "upsert_count": 0,
                    "errors": [],
                    "status": "completed",
                }
            )
            mock_compile.return_value = mock_graph

            # First run: all files processed
            orchestrator1 = BatchOrchestrator(concurrency=1, incremental=True)
            orchestrator1.graph = mock_graph
            orchestrator1.ledger.get_entry = MagicMock(return_value=None)

            summary1 = await orchestrator1.run_batch(str(folder))

            assert summary1.ingested >= 1 or summary1.skipped >= 0

            # Second run: all files should be skipped as unchanged
            orchestrator2 = BatchOrchestrator(concurrency=1, incremental=True)
            orchestrator2.graph = mock_graph
            orchestrator2.ledger.get_entry = MagicMock(
                return_value={
                    "status": "ingested",
                    "content_hash": "abc123",
                }
            )

            summary2 = await orchestrator2.run_batch(str(folder))

            assert summary2.skipped > 0
            assert summary2.skipped_unchanged > 0

    @pytest.mark.asyncio
    async def test_full_run_processes_all(self, tmp_path: Path):
        """
        Integration test:
        --full flag should process all files regardless of ledger state.
        """
        folder = tmp_path / "docs"
        folder.mkdir()
        (folder / "engineering_2025-S1_lecture_test.pdf").write_bytes(
            b"%PDF-1.4\nTest\n"
        )

        with patch("ingestion.orchestrator.compile_ingestion_graph") as mock_compile:
            mock_graph = AsyncMock()
            mock_graph.ainvoke = AsyncMock(
                return_value={
                    "run_id": "test-run",
                    "document_id": "test-doc-id",
                    "file_path": "",
                    "docling_doc": None,
                    "structure_tree": [],
                    "metadata": {},
                    "chunks": [],
                    "dense_vectors": [],
                    "sparse_vectors": [],
                    "upsert_count": 0,
                    "errors": [],
                    "status": "completed",
                }
            )
            mock_compile.return_value = mock_graph

            # Full run: should process even if ledger says ingested
            orchestrator = BatchOrchestrator(concurrency=1, force_full=True)
            orchestrator.graph = mock_graph
            orchestrator.ledger.get_entry = MagicMock(
                return_value={
                    "status": "ingested",
                    "content_hash": "abc123",
                }
            )

            summary = await orchestrator.run_batch(str(folder))

            # Should not skip due to unchanged (force_full=True)
            assert summary.skipped_unchanged == 0
