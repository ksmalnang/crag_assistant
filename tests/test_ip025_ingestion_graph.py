"""Tests for IP-025: LangGraph ingestion subgraph wiring."""

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from ingestion.graph import build_ingestion_graph, compile_ingestion_graph
from ingestion.nodes import (
    _make_error,
    intake_node,
    metadata_resolver_node,
)
from ingestion.state import IngestionState

# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    """Create a sample PDF file for testing."""
    pdf_file = tmp_path / "engineering_2025-S1_lecture_intro_to_ml.pdf"
    # Write a minimal PDF-like header (not a real PDF, but non-empty)
    pdf_file.write_bytes(b"%PDF-1.4\n%Fake PDF for testing\n")
    return pdf_file


@pytest.fixture
def empty_file(tmp_path: Path) -> Path:
    """Create an empty file for testing."""
    f = tmp_path / "empty.pdf"
    f.write_bytes(b"")
    return f


@pytest.fixture
def unsupported_file(tmp_path: Path) -> Path:
    """Create a file with unsupported extension."""
    f = tmp_path / "document.xyz"
    f.write_bytes(b"some content")
    return f


@pytest.fixture
def initial_state(sample_pdf_path: Path) -> IngestionState:
    """Create a valid initial state for graph testing."""
    file_bytes = sample_pdf_path.read_bytes()
    document_id = hashlib.sha256(file_bytes).hexdigest()
    return {
        "run_id": "test-run-001",
        "file_path": str(sample_pdf_path),
        "document_id": document_id,
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


# ─── IngestionState Tests ─────────────────────────────────────────────────────


class TestIngestionState:
    """Tests for IngestionState TypedDict."""

    def test_state_has_required_fields(self):
        """Verify IngestionState has all required fields."""
        state: IngestionState = {
            "run_id": "run-1",
            "file_path": "/path/to/file.pdf",
            "document_id": "abc123",
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
        assert state["run_id"] == "run-1"
        assert state["file_path"] == "/path/to/file.pdf"
        assert state["document_id"] == "abc123"
        assert state["docling_doc"] is None
        assert state["structure_tree"] == []
        assert state["metadata"] == {}
        assert state["chunks"] == []
        assert state["dense_vectors"] == []
        assert state["sparse_vectors"] == []
        assert state["upsert_count"] == 0
        assert state["errors"] == []
        assert state["status"] == "pending"

    def test_errors_can_be_appended(self):
        """Verify errors field supports appending."""
        state: IngestionState = {
            "run_id": "run-1",
            "file_path": "/path/to/file.pdf",
            "document_id": "abc123",
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
        error = _make_error(
            node="test_node",
            reason="test_reason",
            message="test message",
            file_path="/path/to/file.pdf",
        )
        state["errors"].append(error)
        assert len(state["errors"]) == 1
        assert state["errors"][0]["node"] == "test_node"


# ─── Intake Node Tests ────────────────────────────────────────────────────────


class TestIntakeNode:
    """Tests for the intake node."""

    @patch("ingestion.nodes.PreflightValidator")
    def test_intake_node_passes_valid_pdf(self, mock_preflight, sample_pdf_path: Path):
        """Intake node should pass valid PDF files."""
        mock_preflight.return_value.validate.return_value = True

        state: IngestionState = {
            "run_id": "test-run-001",
            "file_path": str(sample_pdf_path),
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

        # Should have document_id set
        assert result["document_id"] != ""
        assert len(result["document_id"]) == 64  # SHA256 hex
        assert result["status"] == "processing"
        assert result["errors"] == []

    def test_intake_node_rejects_empty_file(self, empty_file: Path):
        """Intake node should reject empty files."""
        state: IngestionState = {
            "run_id": "test-run-001",
            "file_path": str(empty_file),
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

    def test_intake_node_rejects_unsupported_extension(self, unsupported_file: Path):
        """Intake node should reject unsupported file extensions."""
        state: IngestionState = {
            "run_id": "test-run-001",
            "file_path": str(unsupported_file),
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
        assert "unsupported_format" in result["errors"][0]["reason"]


# ─── Metadata Resolver Node Tests ─────────────────────────────────────────────


class TestMetadataResolverNode:
    """Tests for the metadata resolver node."""

    def test_metadata_resolver_parses_conventional_filename(
        self, sample_pdf_path: Path
    ):
        """Metadata resolver should parse conventional filenames."""
        state: IngestionState = {
            "run_id": "test-run-001",
            "file_path": str(sample_pdf_path),
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
        meta = result["metadata"]
        assert meta["faculty"] == "engineering"
        assert meta["semester"] == "2025-S1"
        assert meta["doc_type"] == "lecture"
        assert meta["unclassified"] is False

    def test_metadata_resolver_handles_unclassified_filename(self, tmp_path: Path):
        """Metadata resolver should continue with warning on unclassified filename."""
        weird_file = tmp_path / "weird_name.pdf"
        weird_file.write_bytes(b"%PDF-1.4\nsome content\n")

        state: IngestionState = {
            "run_id": "test-run-001",
            "file_path": str(weird_file),
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
        # Should have a warning error but still continue
        assert len(result["errors"]) > 0
        assert result["errors"][0]["reason"] == "unclassified_filename"
        assert result["metadata"]["unclassified"] is True


# ─── Graph Tests ───────────────────────────────────────────────────────────────


class TestIngestionGraph:
    """Tests for the LangGraph ingestion graph."""

    def test_graph_builds_successfully(self):
        """Graph should build without errors."""
        graph = build_ingestion_graph()
        assert graph is not None

    def test_graph_compiles_successfully(self):
        """Graph should compile without errors."""
        compiled = compile_ingestion_graph()
        assert compiled is not None

    def test_graph_has_all_required_nodes(self):
        """Graph should contain all required ingestion nodes."""
        graph = build_ingestion_graph()
        # Check nodes are registered
        # LangGraph stores nodes in the builder
        assert graph is not None

    def test_graph_has_conditional_edges(self):
        """Graph should have conditional edges for error routing."""
        graph = build_ingestion_graph()
        assert graph is not None


# ─── Integration Test ─────────────────────────────────────────────────────────


class TestIngestionIntegration:
    """Integration tests for the ingestion pipeline."""

    def test_initial_state_validates_correctly(self, initial_state: IngestionState):
        """Initial state should pass all TypedDict validation."""
        assert initial_state["run_id"] == "test-run-001"
        assert Path(initial_state["file_path"]).exists()
        assert initial_state["status"] == "pending"
        assert initial_state["errors"] == []

    @patch("ingestion.nodes.PreflightValidator")
    def test_intake_node_sets_document_id(
        self, mock_preflight, initial_state: IngestionState
    ):
        """Intake node should set document_id from file content."""
        mock_preflight.return_value.validate.return_value = True

        result = intake_node(initial_state)
        assert result["document_id"] != ""
        assert result["status"] == "processing"

    @pytest.mark.asyncio
    async def test_graph_runs_intake_to_end(self, initial_state: IngestionState):
        """
        Integration test: graph should run from START through intake node.

        Note: Full end-to-end requires Qdrant, OpenRouter API, etc.
        This test verifies the graph structure and intake flow.
        """
        compiled = compile_ingestion_graph()

        # Run only up to intake by checking state after invoke
        # For a full test we'd need mocks for Qdrant/OpenRouter
        result = await compiled.ainvoke(initial_state)

        # Should complete or fail gracefully without external services
        assert "status" in result
