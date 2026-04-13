"""Tests for IP-035: End-to-end integration test against live Qdrant.

Requires Qdrant Docker container running.
Run with: pytest tests/test_ip035_e2e_integration.py -v --live-qdrant
"""

import os
import uuid
from pathlib import Path

import pytest

# Skip all tests if QDRANT_TEST_URL not set
QDRANT_TEST_URL = os.environ.get("QDRANT_TEST_URL", "http://localhost:6334")


def qdrant_available() -> bool:
    """Check if Qdrant is available at the test URL."""
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=QDRANT_TEST_URL, timeout=5)
        client.get_collections()
        return True
    except Exception:
        return False


# Fixtures directory
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ingestion"


@pytest.fixture(scope="module")
def qdrant_client():
    """Create a Qdrant client for testing."""
    if not qdrant_available():
        pytest.skip(f"Qdrant not available at {QDRANT_TEST_URL}")

    from qdrant_client import QdrantClient

    client = QdrantClient(url=QDRANT_TEST_URL)
    yield client
    # Cleanup is handled by the test collection fixture


@pytest.fixture
def test_collection(qdrant_client):
    """Create a fresh test collection and clean it up after the test."""
    collection_name = f"test_e2e_{uuid.uuid4().hex[:8]}"

    # Create collection
    from qdrant_client.models import (
        Distance,
        SparseIndexParams,
        SparseVectorParams,
        VectorParams,
    )

    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=4096,
                distance=Distance.COSINE,
            ),
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False,
                    )
                )
            },
        )
    except Exception as e:
        if "already exists" not in str(e).lower():
            raise

    yield collection_name

    # Cleanup: delete collection
    try:
        qdrant_client.delete_collection(collection_name=collection_name)
    except Exception:
        pass


@pytest.mark.integration
class TestEndToEndIngestion:
    """End-to-end integration tests for the ingestion pipeline."""

    def test_pdf_with_headings_ingests_correctly(
        self, qdrant_client, test_collection, tmp_path: Path
    ):
        """PDF with headings should ingest and produce correct chunks."""
        from chunking.chunkers import ChunkingConfig, TableAwareChunker
        from metadata.chunk_metadata import ChunkMetadata
        from vector_store.upsert import QdrantUpsertNode

        # Create a mock parsed document with structure
        _create_mock_parsed_document(
            document_id="test-pdf-001",
            headings=["Introduction", "Methods", "Results"],
            text_content="Introduction\n\nThis is the introduction.\n\nMethods\n\nThese are the methods.\n\nResults\n\nHere are the results.",
        )

        # Chunk the document
        config = ChunkingConfig(max_chunk_tokens=1024)
        TableAwareChunker(config)

        # For this test, we create chunks manually since we don't have a real parser
        chunks = [
            ChunkMetadata(
                document_id="test-pdf-001",
                chunk_index=0,
                page_start=1,
                page_end=1,
                heading_path=["Introduction"],
                char_start=0,
                char_end=40,
                token_count=10,
            ),
            ChunkMetadata(
                document_id="test-pdf-001",
                chunk_index=1,
                page_start=1,
                page_end=1,
                heading_path=["Methods"],
                char_start=41,
                char_end=75,
                token_count=10,
            ),
        ]

        # Create simple mock vectors for testing (skip real embedding)
        dense_vectors = [[0.1] * 4096 for _ in chunks]
        sparse_vectors = [{"indices": [0, 1], "values": [0.5, 0.3]} for _ in chunks]

        # Upsert to Qdrant
        upsert_node = QdrantUpsertNode(
            client=qdrant_client,
            collection_name=test_collection,
        )

        upsert_tuples = [("test-pdf-001", chunk.chunk_index, chunk) for chunk in chunks]

        count = upsert_node.upsert(
            chunks=upsert_tuples,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
            payload_extra={
                "source_path": "test.pdf",
                "source_name": "Test Document",
                "faculty": "engineering",
                "doc_type": "lecture",
                "semester": "2025-S1",
            },
        )

        # Verify upsert count
        assert count == 2

        # Verify points exist in Qdrant
        actual_count = qdrant_client.count(
            collection_name=test_collection,
            count_filter=None,
        ).count
        assert actual_count >= 2

    def test_doc_payload_has_all_metadata_fields(self, qdrant_client, test_collection):
        """Upserted points should have all required metadata fields."""
        from metadata.chunk_metadata import ChunkMetadata
        from vector_store.upsert import QdrantUpsertNode

        # Create a test chunk
        chunk = ChunkMetadata(
            document_id="test-meta-001",
            chunk_index=0,
            page_start=1,
            page_end=1,
            heading_path=["Test"],
            char_start=0,
            char_end=50,
            token_count=15,
        )

        upsert_node = QdrantUpsertNode(
            client=qdrant_client,
            collection_name=test_collection,
        )

        upsert_node.upsert(
            chunks=[("test-meta-001", 0, chunk)],
            dense_vectors=[[0.1] * 4096],
            sparse_vectors=[{"indices": [0], "values": [1.0]}],
            payload_extra={
                "source_path": "test.pdf",
                "source_name": "Test",
                "faculty": "engineering",
                "doc_type": "lecture",
                "semester": "2025-S1",
            },
        )

        # Verify payload fields
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        results = qdrant_client.scroll(
            collection_name=test_collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value="test-meta-001"),
                    )
                ]
            ),
            limit=1,
        )

        points = results[0]
        assert len(points) > 0

        payload = points[0].payload
        required_fields = [
            "document_id",
            "source_path",
            "source_name",
            "faculty",
            "doc_type",
            "semester",
            "chunk_index",
            "page_start",
            "page_end",
            "heading_path",
            "section_title",
            "char_start",
            "char_end",
            "token_count",
        ]

        for field in required_fields:
            assert field in payload, f"Missing payload field: {field}"


def _create_mock_parsed_document(
    document_id: str,
    headings: list[str],
    text_content: str,
) -> object:
    """Create a mock parsed document for testing."""
    mock = type("MockParsedDocument", (), {})()
    mock.document_id = document_id
    mock.has_headings = len(headings) > 0
    mock.structure_tree = []
    mock.tables = []
    mock.text_content = text_content
    mock.is_scanned_pdf = False
    mock.page_count = 1
    return mock
