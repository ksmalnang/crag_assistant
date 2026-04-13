"""Tests for IP-034: Unit tests for every ingestion pipeline node.

Tests all IP7/8 nodes with mocked external dependencies.
"""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion.nodes import (
    _make_error,
    chunker_node,
    embedding_node,
    health_check_node,
    intake_node,
    metadata_resolver_node,
    parser_node,
    upsert_node,
)
from ingestion.state import IngestionState

# ─── Shared Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a sample PDF file."""
    f = tmp_path / "engineering_2025-S1_lecture_intro.pdf"
    f.write_bytes(b"%PDF-1.4\nSample content for testing.\n")
    return f


@pytest.fixture
def base_state(sample_pdf: Path) -> IngestionState:
    """Create a base IngestionState for testing."""
    content = sample_pdf.read_bytes()
    doc_id = hashlib.sha256(content).hexdigest()
    return {
        "run_id": "test-run-001",
        "file_path": str(sample_pdf),
        "document_id": doc_id,
        "docling_doc": None,
        "structure_tree": [],
        "metadata": {},
        "chunks": [],
        "dense_vectors": [],
        "sparse_vectors": [],
        "upsert_count": 0,
        "errors": [],
        "status": "pending",
    }


# ─── _make_error Tests ───────────────────────────────────────────────────────


class TestMakeError:
    """Tests for the _make_error helper function."""

    def test_creates_error_entry(self):
        """_make_error should create an ErrorEntry dict."""
        entry = _make_error(
            node="test_node",
            reason="test_reason",
            message="test message",
            file_path="/path/to/file.pdf",
        )
        assert isinstance(entry, dict)
        assert entry["node"] == "test_node"
        assert entry["reason"] == "test_reason"
        assert entry["message"] == "test message"
        assert entry["file_path"] == "/path/to/file.pdf"


# ─── Intake Node Tests ───────────────────────────────────────────────────────


class TestIntakeNode:
    """Tests for the intake_node function."""

    @patch("ingestion.nodes.PreflightValidator")
    def test_passes_valid_pdf(
        self, mock_validator, sample_pdf: Path, base_state: IngestionState
    ):
        """Should pass valid PDF files."""
        mock_validator.return_value.validate.return_value = True

        result = intake_node(base_state)

        assert result["document_id"] != ""
        assert len(result["document_id"]) == 64  # SHA256 hex
        assert result["status"] == "processing"
        assert result["errors"] == []

    @patch("ingestion.nodes.PreflightValidator")
    def test_rejects_unsupported_extension(self, mock_validator, tmp_path: Path):
        """Should reject unsupported file extensions."""
        f = tmp_path / "document.xyz"
        f.write_bytes(b"some content")

        state: IngestionState = {
            "run_id": "test-run",
            "file_path": str(f),
            "document_id": "",
            "docling_doc": None,
            "structure_tree": [],
            "metadata": {},
            "chunks": [],
            "dense_vectors": [],
            "sparse_vectors": [],
            "upsert_count": 0,
            "errors": [],
            "status": "pending",
        }

        result = intake_node(state)

        assert result["status"] == "failed"
        assert len(result["errors"]) > 0
        assert result["errors"][0]["node"] == "intake_node"

    def test_rejects_empty_file(self, tmp_path: Path):
        """Should reject empty files."""
        f = tmp_path / "empty.pdf"
        f.write_bytes(b"")

        state: IngestionState = {
            "run_id": "test-run",
            "file_path": str(f),
            "document_id": "",
            "docling_doc": None,
            "structure_tree": [],
            "metadata": {},
            "chunks": [],
            "dense_vectors": [],
            "sparse_vectors": [],
            "upsert_count": 0,
            "errors": [],
            "status": "pending",
        }

        result = intake_node(state)

        assert result["status"] == "failed"
        assert len(result["errors"]) > 0

    def test_rejects_nonexistent_file(self):
        """Should reject files that don't exist."""
        state: IngestionState = {
            "run_id": "test-run",
            "file_path": "/nonexistent/file.pdf",
            "document_id": "",
            "docling_doc": None,
            "structure_tree": [],
            "metadata": {},
            "chunks": [],
            "dense_vectors": [],
            "sparse_vectors": [],
            "upsert_count": 0,
            "errors": [],
            "status": "pending",
        }

        result = intake_node(state)

        assert result["status"] == "failed"
        assert len(result["errors"]) > 0


# ─── Parser Node Tests ───────────────────────────────────────────────────────


class TestParserNode:
    """Tests for the parser_node function."""

    @pytest.mark.asyncio
    async def test_parser_node_success(self, base_state: IngestionState):
        """Parser node should parse the document successfully (mocked)."""
        mock_parsed = MagicMock()
        mock_parsed.page_count = 3
        mock_parsed.structure_tree = []
        mock_parsed.tables = []
        mock_parsed.has_headings = False
        mock_parsed.is_scanned_pdf = False
        mock_parsed.text_content = "Sample content"

        async def mock_parse_file(*args, **kwargs):
            return mock_parsed

        with patch("ingestion.nodes.DoclingParser") as mock_parser:
            mock_parser.return_value.parse_file = mock_parse_file

            result = await parser_node(base_state)

        assert "docling_doc" in result
        assert result["docling_doc"] is mock_parsed
        assert "structure_tree" in result
        assert result["errors"] == []

    @pytest.mark.asyncio
    async def test_parser_node_handles_parse_error(self, base_state: IngestionState):
        """Parser node should catch parse errors."""
        with patch("ingestion.nodes.DoclingParser") as mock_parser:
            mock_parser.return_value.parse_file.side_effect = Exception("Parse failed")

            result = await parser_node(base_state)

        assert result["status"] == "failed"
        assert len(result["errors"]) > 0
        assert result["errors"][0]["node"] == "parser_node"


# ─── Metadata Resolver Node Tests ────────────────────────────────────────────


class TestMetadataResolverNode:
    """Tests for the metadata_resolver_node function."""

    def test_parses_conventional_filename(
        self, sample_pdf: Path, base_state: IngestionState
    ):
        """Should parse conventional filename conventions."""
        result = metadata_resolver_node(base_state)

        assert "metadata" in result
        meta = result["metadata"]
        assert meta["faculty"] == "engineering"
        assert meta["semester"] == "2025-S1"
        assert meta["doc_type"] == "lecture"
        assert meta["unclassified"] is False

    def test_handles_unclassified_filename(self, tmp_path: Path):
        """Should handle unclassified filenames with warnings."""
        f = tmp_path / "weird_name.pdf"
        f.write_bytes(b"%PDF-1.4\ncontent\n")

        state: IngestionState = {
            "run_id": "test-run",
            "file_path": str(f),
            "document_id": "test-doc-id",
            "docling_doc": None,
            "structure_tree": [],
            "metadata": {},
            "chunks": [],
            "dense_vectors": [],
            "sparse_vectors": [],
            "upsert_count": 0,
            "errors": [],
            "status": "pending",
        }

        result = metadata_resolver_node(state)

        assert "metadata" in result
        assert result["metadata"]["unclassified"] is True
        # Should have warning error but continue
        assert len(result["errors"]) > 0
        assert result["errors"][0]["reason"] == "unclassified_filename"


# ─── Chunker Node Tests ──────────────────────────────────────────────────────


class TestChunkerNode:
    """Tests for the chunker_node function."""

    def test_chunks_document_with_headings(self, base_state: IngestionState):
        """Should chunk documents with headings."""
        mock_parsed = MagicMock()
        mock_parsed.has_headings = True
        mock_parsed.structure_tree = []
        mock_parsed.tables = []
        mock_parsed.text_content = "Heading\n\nContent under heading."
        mock_parsed.document_id = "test-doc"
        mock_parsed.is_scanned_pdf = False

        state = {**base_state, "docling_doc": mock_parsed}

        result = chunker_node(state)

        assert "chunks" in result
        assert result["errors"] == []

    def test_handles_no_parsed_document(self, base_state: IngestionState):
        """Should fail when no parsed document is available."""
        state = {**base_state, "docling_doc": None}

        result = chunker_node(state)

        assert result["status"] == "failed"
        assert len(result["errors"]) > 0
        assert result["errors"][0]["node"] == "chunker_node"


# ─── Embedding Node Tests ────────────────────────────────────────────────────


class TestEmbeddingNode:
    """Tests for the embedding_node function."""

    def test_fails_with_no_chunks(self, base_state: IngestionState):
        """Should fail when no chunks are available."""
        result = embedding_node(base_state)

        assert result["status"] == "failed"
        assert len(result["errors"]) > 0
        assert result["errors"][0]["node"] == "embedding_node"

    def test_fails_with_no_parsed_document(self, base_state: IngestionState):
        """Should fail when no parsed document for text extraction."""
        mock_chunk = MagicMock()
        mock_chunk.document_id = "test-doc"
        mock_chunk.chunk_index = 0
        mock_chunk.char_start = 0
        mock_chunk.char_end = 100
        mock_chunk.heading_path = []

        state = {
            **base_state,
            "chunks": [mock_chunk],
            "docling_doc": None,
        }

        result = embedding_node(state)

        assert result["status"] == "failed"
        assert len(result["errors"]) > 0


# ─── Upsert Node Tests ───────────────────────────────────────────────────────


class TestUpsertNode:
    """Tests for the upsert_node function."""

    def test_fails_with_no_chunks(self, base_state: IngestionState):
        """Should fail when no chunks to upsert."""
        result = upsert_node(base_state)

        assert result["status"] == "failed"
        assert len(result["errors"]) > 0

    def test_fails_with_no_vectors(self, base_state: IngestionState):
        """Should fail when no vectors available."""
        mock_chunk = MagicMock()
        mock_chunk.document_id = "test-doc"
        mock_chunk.chunk_index = 0
        mock_chunk.char_start = 0
        mock_chunk.char_end = 100
        mock_chunk.heading_path = []

        state = {
            **base_state,
            "chunks": [mock_chunk],
            "docling_doc": MagicMock(text_content="test"),
            "metadata": {
                "source_path": str(base_state["file_path"]),
                "source_name": "Test",
                "faculty": "engineering",
                "doc_type": "lecture",
                "semester": "2025-S1",
            },
            "dense_vectors": [],  # No vectors
            "sparse_vectors": [],
        }

        result = upsert_node(state)

        assert result["status"] == "failed"
        assert len(result["errors"]) > 0


# ─── Health Check Node Tests ─────────────────────────────────────────────────


class TestHealthCheckNode:
    """Tests for the health_check_node function."""

    @patch("ingestion.nodes.QdrantClient")
    def test_runs_health_check(self, mock_client, base_state: IngestionState):
        """Should run health check and return results."""
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        mock_instance.get_collection.return_value = MagicMock(
            status="green",
            vectors_count=5,
            indexed_vectors_count=5,
            points_count=5,
        )
        mock_instance.query_points.return_value = MagicMock(points=[MagicMock()])

        state = {
            **base_state,
            "upsert_count": 5,
        }

        result = health_check_node(state)

        # Should complete even if checks fail
        assert "errors" in result
        assert result["status"] == "completed"

    @patch("ingestion.nodes.QdrantClient")
    def test_handles_health_check_failure(
        self, mock_client, base_state: IngestionState
    ):
        """Should handle health check failures gracefully."""
        mock_client.side_effect = Exception("Connection failed")

        state = {
            **base_state,
            "upsert_count": 5,
        }

        result = health_check_node(state)

        # Should still complete (status=completed) since data was written
        assert result["status"] == "completed"
        assert len(result["errors"]) > 0
        assert result["errors"][0]["node"] == "health_check_node"
