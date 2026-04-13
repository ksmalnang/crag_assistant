"""Tests for IP-026: Batch orchestrator."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingestion.orchestrator import (
    BatchOrchestrator,
    BatchRunSummary,
    DocumentResult,
)

# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_folder(tmp_path: Path) -> Path:
    """Create a folder with sample PDF files for testing."""
    folder = tmp_path / "docs"
    folder.mkdir()

    # Create sample files with conventional naming
    files = [
        "engineering_2025-S1_lecture_intro.pdf",
        "science_2025-S2_tutorial_lab.pdf",
        "arts_2024-S1_assignment_essay.pdf",
    ]
    for name in files:
        f = folder / name
        f.write_bytes(b"%PDF-1.4\nSample content\n")

    return folder


@pytest.fixture
def empty_folder(tmp_path: Path) -> Path:
    """Create an empty folder."""
    folder = tmp_path / "empty"
    folder.mkdir()
    return folder


@pytest.fixture
def orchestrator():
    """Create a batch orchestrator for testing."""
    with patch("ingestion.orchestrator.compile_ingestion_graph") as mock_compile:
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "run_id": "test-run",
                "file_path": "",
                "document_id": "test-doc-id",
                "docling_doc": None,
                "structure_tree": [],
                "metadata": {
                    "faculty": "engineering",
                    "doc_type": "lecture",
                    "semester": "2025-S1",
                    "source_path": "",
                    "source_name": "",
                    "unclassified": False,
                },
                "chunks": [],
                "dense_vectors": [],
                "sparse_vectors": [],
                "upsert_count": 0,
                "errors": [],
                "status": "completed",
            }
        )
        mock_compile.return_value = mock_graph

        with patch("ingestion.orchestrator.IngestionLedger") as mock_ledger:
            mock_ledger_instance = MagicMock()
            mock_ledger_instance.get_entry.return_value = None
            mock_ledger.return_value = mock_ledger_instance

            orchestrator = BatchOrchestrator(concurrency=2)
            # Replace the graph and ledger
            orchestrator.graph = mock_graph
            orchestrator.ledger = mock_ledger_instance
            yield orchestrator


# ─── DocumentResult Tests ─────────────────────────────────────────────────────


class TestDocumentResult:
    """Tests for DocumentResult dataclass."""

    def test_default_values(self):
        """DocumentResult should have sensible defaults."""
        result = DocumentResult(file_path="/path/to/file.pdf")
        assert result.file_path == "/path/to/file.pdf"
        assert result.document_id is None
        assert result.status == "pending"
        assert result.chunks_created == 0
        assert result.vectors_upserted == 0
        assert result.error_reason is None
        assert result.error_node is None


# ─── BatchRunSummary Tests ────────────────────────────────────────────────────


class TestBatchRunSummary:
    """Tests for BatchRunSummary dataclass."""

    def test_to_dict_includes_all_fields(self):
        """to_dict should include all summary fields."""
        summary = BatchRunSummary(
            run_id="test-run-001",
            started_at="2025-01-01T00:00:00",
            completed_at="2025-01-01T00:01:00",
            total_files=10,
            ingested=8,
            skipped=1,
            skipped_unchanged=1,
            skipped_error=0,
            failed=1,
            total_chunks_created=50,
            total_vectors_upserted=50,
            errors=[
                {"file": "bad.pdf", "reason": "parse_error", "node": "parser_node"}
            ],
            duration_seconds=60.0,
        )

        d = summary.to_dict()
        assert d["run_id"] == "test-run-001"
        assert d["total_files"] == 10
        assert d["ingested"] == 8
        assert d["skipped"] == 1
        assert d["skipped_unchanged"] == 1
        assert d["failed"] == 1
        assert d["total_chunks_created"] == 50
        assert d["total_vectors_upserted"] == 50
        assert len(d["errors"]) == 1


# ─── BatchOrchestrator Tests ──────────────────────────────────────────────────


class TestBatchOrchestrator:
    """Tests for the BatchOrchestrator."""

    def test_collects_files_from_folder(self, sample_folder: Path):
        """Orchestrator should collect all supported files."""
        with patch("ingestion.orchestrator.compile_ingestion_graph"):
            orchestrator = BatchOrchestrator(concurrency=2)
            files = orchestrator._collect_files(str(sample_folder))
            assert len(files) == 3
            assert all(f.endswith(".pdf") for f in files)

    def test_collects_no_files_from_empty_folder(self, empty_folder: Path):
        """Orchestrator should return empty list for empty folder."""
        with patch("ingestion.orchestrator.compile_ingestion_graph"):
            orchestrator = BatchOrchestrator(concurrency=2)
            files = orchestrator._collect_files(str(empty_folder))
            assert len(files) == 0

    def test_skip_logic_incremental_mode(self, orchestrator: BatchOrchestrator):
        """Should skip files already ingested in incremental mode."""
        orchestrator.incremental = True
        orchestrator.force_full = False

        # Mock ledger entry showing file is already ingested
        orchestrator.ledger.get_entry.return_value = {
            "file_path": "/path/to/file.pdf",
            "document_id": "abc123",
            "status": "ingested",
            "content_hash": "def456",
        }

        should_skip, reason = orchestrator._should_skip("/path/to/file.pdf")
        assert should_skip is True
        assert "unchanged" in reason

    def test_skip_logic_force_full_mode(self, orchestrator: BatchOrchestrator):
        """Should not skip files when force_full is True."""
        orchestrator.incremental = True
        orchestrator.force_full = True

        orchestrator.ledger.get_entry.return_value = {
            "file_path": "/path/to/file.pdf",
            "document_id": "abc123",
            "status": "ingested",
            "content_hash": "def456",
        }

        should_skip, reason = orchestrator._should_skip("/path/to/file.pdf")
        assert should_skip is False

    def test_skip_logic_non_incremental(self, orchestrator: BatchOrchestrator):
        """Should not skip files when incremental is False."""
        orchestrator.incremental = False

        should_skip, reason = orchestrator._should_skip("/path/to/file.pdf")
        assert should_skip is False

    @pytest.mark.asyncio
    async def test_ingest_document_success(
        self, orchestrator: BatchOrchestrator, tmp_path: Path
    ):
        """Should successfully ingest a document."""
        pdf_file = tmp_path / "engineering_2025-S1_lecture_test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\nTest content\n")

        orchestrator.graph.ainvoke.return_value = {
            "run_id": "test-run",
            "file_path": str(pdf_file),
            "document_id": "test-doc-id",
            "docling_doc": None,
            "structure_tree": [],
            "metadata": {},
            "chunks": [],
            "dense_vectors": [],
            "sparse_vectors": [],
            "upsert_count": 5,
            "errors": [],
            "status": "completed",
        }

        result = await orchestrator.ingest_document(str(pdf_file), "test-run")

        assert result.status == "success"
        assert result.document_id == "test-doc-id"
        assert result.vectors_upserted == 5

    @pytest.mark.asyncio
    async def test_ingest_document_skipped(
        self, orchestrator: BatchOrchestrator, tmp_path: Path
    ):
        """Should skip already ingested document in incremental mode."""
        pdf_file = tmp_path / "engineering_2025-S1_lecture_test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\nTest content\n")

        orchestrator.incremental = True
        orchestrator.ledger.get_entry.return_value = {
            "file_path": str(pdf_file),
            "document_id": "abc123",
            "status": "ingested",
            "content_hash": "def456",
        }

        result = await orchestrator.ingest_document(str(pdf_file), "test-run")

        assert result.status == "skipped"

    @pytest.mark.asyncio
    async def test_run_batch_empty_folder(
        self, orchestrator: BatchOrchestrator, empty_folder: Path
    ):
        """Should handle empty folder gracefully."""
        summary = await orchestrator.run_batch(str(empty_folder))

        assert summary.total_files == 0
        assert summary.run_id is not None

    def test_concurrency_uses_semaphore(self):
        """Orchestrator should use asyncio.Semaphore for bounded concurrency."""
        with patch("ingestion.orchestrator.compile_ingestion_graph"):
            orchestrator = BatchOrchestrator(concurrency=4)
            assert isinstance(orchestrator.semaphore, asyncio.Semaphore)


# ─── Incremental Ingestion Integration Test ───────────────────────────────────


class TestIncrementalIngestion:
    """Integration test for incremental ingestion mode."""

    @pytest.mark.asyncio
    async def test_incremental_skips_unchanged(self, tmp_path: Path):
        """
        Integration test: ingest folder, then re-run incremental —
        all files should be skipped as unchanged.
        """
        # Create sample folder
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
                    "file_path": "",
                    "document_id": "test-doc-id",
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

            # First run: ingest all
            orchestrator1 = BatchOrchestrator(concurrency=1, incremental=True)
            orchestrator1.graph = mock_graph

            # Patch ledger to show nothing ingested initially
            with patch.object(orchestrator1.ledger, "get_entry", return_value=None):
                await orchestrator1.run_batch(str(folder))

            # Second run: should skip all as unchanged
            orchestrator2 = BatchOrchestrator(concurrency=1, incremental=True)
            orchestrator2.graph = mock_graph

            # Patch ledger to show files already ingested
            with patch.object(
                orchestrator2.ledger,
                "get_entry",
                return_value={
                    "status": "ingested",
                    "content_hash": "abc123",
                },
            ):
                summary2 = await orchestrator2.run_batch(str(folder))

            assert summary2.skipped > 0
