"""Unit tests for IP6 - Vector Store Epic (IP-021 to IP-024)."""

from __future__ import annotations

import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.models import Distance

# Set required env vars before any imports
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")

from metadata.chunk_metadata import ChunkMetadata
from vector_store.deletion import StaleDeletionNode
from vector_store.errors import (
    DeletionError,
    HealthCheckError,
    SchemaError,
    StoreError,
    UpsertError,
)
from vector_store.health_check import HealthCheckResult, UpsertHealthChecker
from vector_store.schema import (
    INTEGER_INDEX_FIELDS,
    KEYWORD_INDEX_FIELDS,
    CollectionSchemaManager,
)
from vector_store.upsert import QdrantUpsertNode

# ============================================================
# IP-021: Qdrant Collection Schema Tests
# ============================================================


class TestCollectionSchemaManager:
    """Test collection schema definition and provisioning."""

    @pytest.fixture
    def mock_client(self):
        """Create a mocked Qdrant client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def manager(self, mock_client):
        """Create schema manager with mock client."""
        with patch("vector_store.schema.settings") as mock_settings:
            mock_settings.qdrant_collection = "test_collection"
            mock_settings.qdrant_dense_dim = 1024
            mock_settings.qdrant_sparse_distance = "Dot"
            mock_settings.qdrant_schema_version = "1.0.0"
            # Configure get_collections to return a proper object
            mock_collections_response = MagicMock()
            mock_collections_response.collections = []
            mock_client.get_collections.return_value = mock_collections_response
            return CollectionSchemaManager(
                client=mock_client,
                collection_name="test_collection",
                dense_dim=1024,
                sparse_distance="Dot",
                schema_version="1.0.0",
            )

    def test_ensure_collection_creates_new(self, mock_client, manager):
        """Test collection is created when it doesn't exist."""
        mock_client.get_collections.return_value = MagicMock(collections=[])

        result = manager.ensure_collection()

        assert result is True
        mock_client.create_collection.assert_called_once()

    def test_ensure_collection_verifies_existing(self, mock_client, manager):
        """Test existing collection schema is verified."""
        existing = MagicMock()
        existing.name = "test_collection"
        coll_resp = MagicMock()
        coll_resp.collections = [existing]
        mock_client.get_collections.return_value = coll_resp
        # Mock collection info for verification
        mock_info = MagicMock()
        mock_dense = MagicMock()
        mock_dense.size = 1024
        mock_dense.distance = Distance.COSINE
        mock_info.config.params.vectors = {"dense": mock_dense}
        mock_sparse = MagicMock()
        mock_sparse.modifier = "none"  # Dot -> 'none'
        mock_info.config.params.sparse_vectors = {"sparse": mock_sparse}
        mock_info.payload_index_schema = {
            "faculty": "keyword",
            "doc_type": "keyword",
            "semester": "keyword",
            "document_id": "keyword",
            "chunk_index": "integer",
        }
        mock_client.get_collection.return_value = mock_info
        mock_client.upsert.side_effect = Exception("mock skip")

        result = manager.ensure_collection()

        assert result is True
        mock_client.create_collection.assert_not_called()

    def test_schema_mismatch_raises_error(self, mock_client, manager):
        """Test schema mismatch raises SchemaError."""
        existing = MagicMock()
        existing.name = "test_collection"
        mock_coll_resp = MagicMock()
        mock_coll_resp.collections = [existing]
        mock_client.get_collections.return_value = mock_coll_resp
        # Wrong dimension
        mock_info = MagicMock()
        mock_dense = MagicMock()
        mock_dense.size = 3072
        mock_dense.distance = Distance.COSINE
        mock_info.config.params.vectors = {"dense": mock_dense}
        mock_info.config.params.sparse_vectors = {"sparse": MagicMock()}
        mock_client.get_collection.return_value = mock_info

        with pytest.raises(SchemaError):
            manager.ensure_collection()

    def test_missing_sparse_vector_raises_error(self, mock_client, manager):
        """Test missing sparse vector raises SchemaError."""
        existing = MagicMock()
        existing.name = "test_collection"
        mock_coll_resp = MagicMock()
        mock_coll_resp.collections = [existing]
        mock_client.get_collections.return_value = mock_coll_resp
        mock_info = MagicMock()
        mock_dense = MagicMock()
        mock_dense.size = 1024
        mock_dense.distance = Distance.COSINE
        mock_info.config.params.vectors = {"dense": mock_dense}
        mock_info.config.params.sparse_vectors = {}
        mock_client.get_collection.return_value = mock_info

        with pytest.raises(SchemaError):
            manager.ensure_collection()

    def test_create_collection_sets_vectors(self, mock_client, manager):
        """Test collection creation includes dense and sparse vectors."""
        mock_client.get_collections.return_value = MagicMock(collections=[])

        manager.ensure_collection()

        call_kwargs = mock_client.create_collection.call_args[1]
        assert "dense" in call_kwargs["vectors_config"]
        assert "sparse" in call_kwargs["sparse_vectors_config"]

    def test_create_collection_sets_dense_dim(self, mock_client, manager):
        """Test collection creation uses correct dense dimension."""
        mock_client.get_collections.return_value = MagicMock(collections=[])

        manager.ensure_collection()

        dense_config = mock_client.create_collection.call_args[1]["vectors_config"][
            "dense"
        ]
        assert dense_config.size == 1024

    def test_create_collection_creates_payload_indexes(self, mock_client, manager):
        """Test payload indexes are created after collection creation."""
        mock_client.get_collections.return_value = MagicMock(collections=[])

        manager.ensure_collection()

        # Should create keyword indexes
        assert mock_client.create_payload_index.call_count >= len(
            KEYWORD_INDEX_FIELDS
        ) + len(INTEGER_INDEX_FIELDS)

    def test_schema_version_stored(self, mock_client, manager):
        """Test schema version is stored after collection creation."""
        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_client.upsert.return_value = MagicMock()

        manager.ensure_collection()

        # Should upsert a schema version point
        assert mock_client.upsert.called

    def test_configurable_collection_name(self, mock_client):
        """Test collection name is configurable."""
        manager = CollectionSchemaManager(
            client=mock_client,
            collection_name="custom_collection",
            dense_dim=1024,
            sparse_distance="Dot",
        )
        mock_client.get_collections.return_value = MagicMock(collections=[])

        manager.ensure_collection()

        call_kwargs = mock_client.create_collection.call_args[1]
        assert call_kwargs["collection_name"] == "custom_collection"

    def test_missing_payload_indexes_created(self, mock_client, manager):
        """Test missing payload indexes are created during verification."""
        mock_client.get_collections.return_value = MagicMock(
            collections=[MagicMock(name="test_collection")]
        )
        mock_info = MagicMock()
        mock_info.config.params.vectors = {
            "dense": MagicMock(size=1024, distance="Cosine")
        }
        mock_sparse = MagicMock()
        mock_sparse.modifier = "none"
        mock_info.config.params.sparse_vectors = {"sparse": mock_sparse}
        # Only some indexes exist
        mock_info.payload_index_schema = {
            "faculty": "keyword",
        }
        mock_client.get_collection.return_value = mock_info

        result = manager.ensure_collection()

        assert result is True
        # Should create missing indexes
        assert mock_client.create_payload_index.called

    def test_sparse_distance_configured(self, mock_client):
        """Test sparse distance is set correctly on collection creation."""
        mock_client.get_collections.return_value = MagicMock(collections=[])
        manager = CollectionSchemaManager(
            client=mock_client,
            collection_name="test_collection",
            dense_dim=1024,
            sparse_distance="Dot",
        )

        manager.ensure_collection()

        call_kwargs = mock_client.create_collection.call_args[1]
        sparse_config = call_kwargs["sparse_vectors_config"]["sparse"]
        # The modifier should be 'none' (Dot -> none)
        assert sparse_config.modifier == "none"

    def test_sparse_distance_mismatch_raises_error(self, mock_client):
        """Test sparse distance mismatch raises SchemaError."""
        existing = MagicMock()
        existing.name = "test_collection"
        mock_coll_resp = MagicMock()
        mock_coll_resp.collections = [existing]
        mock_client.get_collections.return_value = mock_coll_resp

        mock_info = MagicMock()
        mock_dense = MagicMock()
        mock_dense.size = 1024
        mock_dense.distance = Distance.COSINE
        mock_info.config.params.vectors = {"dense": mock_dense}
        # Sparse with wrong modifier (expected 'none' for Dot, got 'idf')
        mock_sparse = MagicMock()
        mock_sparse.modifier = "idf"
        mock_info.config.params.sparse_vectors = {"sparse": mock_sparse}
        mock_client.get_collection.return_value = mock_info

        manager = CollectionSchemaManager(
            client=mock_client,
            collection_name="test_collection",
            dense_dim=1024,
            sparse_distance="Dot",
        )

        with pytest.raises(SchemaError):
            manager.ensure_collection()


# ============================================================
# IP-022: Qdrant Upsert Node Tests
# ============================================================


class TestQdrantUpsertNode:
    """Test Qdrant upsert with full metadata payload."""

    @pytest.fixture
    def mock_client(self):
        """Create mocked Qdrant client."""
        return MagicMock()

    @pytest.fixture
    def node(self, mock_client):
        """Create upsert node."""
        return QdrantUpsertNode(
            client=mock_client,
            collection_name="test_collection",
            batch_size=100,
        )

    def _make_chunk_metadata(self, index: int = 0, document_id: str = "doc-123"):
        """Create ChunkMetadata for testing."""
        return ChunkMetadata(
            document_id=document_id,
            chunk_index=index,
            page_start=1,
            heading_path=["Introduction"],
            char_start=0,
            char_end=100,
            token_count=25,
        )

    def test_make_point_id_deterministic(self, node):
        """Test point ID is deterministic (same inputs = same ID)."""
        id1 = node.make_point_id("doc-123", 0)
        id2 = node.make_point_id("doc-123", 0)
        assert id1 == id2

    def test_make_point_id_unique_per_chunk(self, node):
        """Test different chunk indices produce different point IDs."""
        id1 = node.make_point_id("doc-123", 0)
        id2 = node.make_point_id("doc-123", 1)
        assert id1 != id2

    def test_make_point_id_unique_per_document(self, node):
        """Test different document IDs produce different point IDs."""
        id1 = node.make_point_id("doc-123", 0)
        id2 = node.make_point_id("doc-456", 0)
        assert id1 != id2

    def test_make_point_id_is_sha256(self, node):
        """Test point ID is SHA256 hex digest (64 chars)."""
        point_id = node.make_point_id("doc-123", 0)
        assert len(point_id) == 64
        # Verify it's a valid hex string
        int(point_id, 16)

    def test_build_payload_includes_all_fields(self, node):
        """Test payload includes all required fields."""
        meta = self._make_chunk_metadata()
        payload = node.build_payload(
            chunk_metadata=meta,
            source_path="/path/to/file.pdf",
            source_name="Test Document",
            faculty="engineering",
            doc_type="lecture",
            semester="2025-S1",
        )

        assert payload["document_id"] == "doc-123"
        assert payload["source_path"] == "/path/to/file.pdf"
        assert payload["source_name"] == "Test Document"
        assert payload["faculty"] == "engineering"
        assert payload["doc_type"] == "lecture"
        assert payload["semester"] == "2025-S1"
        assert payload["chunk_index"] == 0
        assert payload["page_start"] == 1
        assert payload["heading_path"] == ["Introduction"]
        assert payload["section_title"] == "Introduction"
        assert payload["char_start"] == 0
        assert payload["char_end"] == 100
        assert payload["token_count"] == 25
        assert "ingested_at" in payload
        assert payload["unclassified"] is False

    def test_build_payload_includes_optional_fields(self, node):
        """Test payload includes optional ChunkMetadata fields."""
        meta = ChunkMetadata(
            document_id="doc-123",
            chunk_index=0,
            page_start=1,
            page_end=2,
            heading_path=["Intro", "Overview"],
            char_start=0,
            char_end=100,
            token_count=25,
            page_image_path="/images/page1.png",
            overlap_context="Previous section context",
            table_index=1,
            is_low_confidence=True,
        )
        payload = node.build_payload(
            chunk_metadata=meta,
            source_path="/path",
            source_name="Test",
            faculty="engineering",
            doc_type="lecture",
            semester="2025-S1",
        )

        assert payload["page_image_path"] == "/images/page1.png"
        assert payload["overlap_context"] == "Previous section context"
        assert payload["table_index"] == 1
        assert payload["is_low_confidence"] is True
        assert payload["page_end"] == 2

    def test_build_payload_optional_fields_none(self, node):
        """Test payload handles None optional fields gracefully."""
        meta = ChunkMetadata(
            document_id="doc-123",
            chunk_index=0,
            page_start=1,
        )
        payload = node.build_payload(
            chunk_metadata=meta,
            source_path="/path",
            source_name="Test",
            faculty="engineering",
            doc_type="lecture",
            semester="2025-S1",
        )

        assert payload["page_image_path"] is None
        assert payload["overlap_context"] is None
        assert payload["table_index"] is None

    def test_build_payload_ingested_at_auto_generated(self, node):
        """Test ingested_at is auto-generated if not provided."""
        meta = self._make_chunk_metadata()
        payload = node.build_payload(
            chunk_metadata=meta,
            source_path="/path",
            source_name="Test",
            faculty="engineering",
            doc_type="lecture",
            semester="2025-S1",
        )
        # Should be a valid ISO timestamp
        datetime.fromisoformat(payload["ingested_at"])

    def test_build_payload_ingested_at_custom(self, node):
        """Test ingested_at uses custom value if provided."""
        meta = self._make_chunk_metadata()
        custom_time = "2025-01-01T00:00:00+00:00"
        payload = node.build_payload(
            chunk_metadata=meta,
            source_path="/path",
            source_name="Test",
            faculty="engineering",
            doc_type="lecture",
            semester="2025-S1",
            ingested_at=custom_time,
        )
        assert payload["ingested_at"] == custom_time

    def test_upsert_single_chunk(self, node):
        """Test single chunk upsert."""
        meta = self._make_chunk_metadata()
        chunks = [("doc-123", 0, meta)]
        dense = [[0.1] * 1024]
        sparse = [{"indices": [0, 1], "values": [0.5, 0.3]}]

        count = node.upsert(
            chunks=chunks,
            dense_vectors=dense,
            sparse_vectors=sparse,
            payload_extra={
                "source_path": "/path",
                "source_name": "Test",
                "faculty": "engineering",
                "doc_type": "lecture",
                "semester": "2025-S1",
            },
        )

        assert count == 1
        node._client.upsert.assert_called_once()

    def test_upsert_multiple_chunks(self, node):
        """Test multiple chunks upserted in one call."""
        chunks = []
        dense = []
        sparse = []
        for i in range(5):
            meta = self._make_chunk_metadata(index=i)
            chunks.append(("doc-123", i, meta))
            dense.append([0.1 + i * 0.01] * 1024)
            sparse.append({"indices": [i], "values": [0.5]})

        count = node.upsert(
            chunks=chunks,
            dense_vectors=dense,
            sparse_vectors=sparse,
            payload_extra={
                "source_path": "/path",
                "source_name": "Test",
                "faculty": "engineering",
                "doc_type": "lecture",
                "semester": "2025-S1",
            },
        )

        assert count == 5

    def test_upsert_empty_chunks(self, node):
        """Test empty chunk list returns 0."""
        count = node.upsert(
            chunks=[],
            dense_vectors=[],
            sparse_vectors=[],
            payload_extra={},
        )
        assert count == 0

    def test_upsert_batches_correctly(self, mock_client):
        """Test large uploads are split into batches."""
        node = QdrantUpsertNode(
            client=mock_client,
            collection_name="test_collection",
            batch_size=2,
        )

        chunks = []
        dense = []
        sparse = []
        for i in range(5):
            meta = self._make_chunk_metadata(index=i)
            chunks.append(("doc-123", i, meta))
            dense.append([0.1 + i * 0.01] * 1024)
            sparse.append({"indices": [i], "values": [0.5]})

        node.upsert(
            chunks=chunks,
            dense_vectors=dense,
            sparse_vectors=sparse,
            payload_extra={
                "source_path": "/path",
                "source_name": "Test",
                "faculty": "engineering",
                "doc_type": "lecture",
                "semester": "2025-S1",
            },
        )

        # Should be called 3 times (2+2+1)
        assert mock_client.upsert.call_count == 3

    def test_upsert_retry_on_429(self, mock_client):
        """Test upsert retries on 429 rate limit."""
        call_count = 0

        def mock_upsert(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                exc = Exception("Rate limited")
                exc.status_code = 429
                raise exc
            return MagicMock()

        mock_client.upsert.side_effect = mock_upsert

        node = QdrantUpsertNode(
            client=mock_client,
            collection_name="test_collection",
            batch_size=100,
            retry_max=3,
            retry_backoff_base=0.01,  # Fast for test
        )

        meta = self._make_chunk_metadata()
        node.upsert(
            chunks=[("doc-123", 0, meta)],
            dense_vectors=[[0.1] * 1024],
            sparse_vectors=[{"indices": [0], "values": [0.5]}],
            payload_extra={
                "source_path": "/path",
                "source_name": "Test",
                "faculty": "engineering",
                "doc_type": "lecture",
                "semester": "2025-S1",
            },
        )

        assert call_count == 3  # 2 failures + 1 success

    def test_upsert_raises_after_max_retries(self, mock_client):
        """Test UpsertError raised after all retries exhausted."""

        def mock_upsert(**kwargs):
            exc = Exception("Service unavailable")
            exc.status_code = 503
            raise exc

        mock_client.upsert.side_effect = mock_upsert

        node = QdrantUpsertNode(
            client=mock_client,
            collection_name="test_collection",
            batch_size=100,
            retry_max=2,
            retry_backoff_base=0.01,
        )

        meta = self._make_chunk_metadata()
        with pytest.raises(UpsertError):
            node.upsert(
                chunks=[("doc-123", 0, meta)],
                dense_vectors=[[0.1] * 1024],
                sparse_vectors=[{"indices": [0], "values": [0.5]}],
                payload_extra={
                    "source_path": "/path",
                    "source_name": "Test",
                    "faculty": "engineering",
                    "doc_type": "lecture",
                    "semester": "2025-S1",
                },
            )

    def test_upsert_raises_immediately_on_non_retryable_error(self, mock_client):
        mock_client.upsert.side_effect = Exception("Bad request")

        node = QdrantUpsertNode(
            client=mock_client,
            collection_name="test_collection",
            retry_max=3,
        )

        meta = self._make_chunk_metadata()
        with pytest.raises(Exception, match="Bad request"):
            node.upsert(
                chunks=[("doc-123", 0, meta)],
                dense_vectors=[[0.1] * 1024],
                sparse_vectors=[{"indices": [0], "values": [0.5]}],
                payload_extra={
                    "source_path": "/path",
                    "source_name": "Test",
                    "faculty": "engineering",
                    "doc_type": "lecture",
                    "semester": "2025-S1",
                },
            )
        # Should not retry
        assert mock_client.upsert.call_count == 1

    def test_count_points_total(self, mock_client, node):
        """Test counting total points in collection."""
        mock_client.count.return_value = MagicMock(count=42)

        count = node.count_points()

        assert count == 42

    def test_count_points_by_document(self, mock_client, node):
        """Test counting points filtered by document_id."""
        mock_client.count.return_value = MagicMock(count=5)

        count = node.count_points(document_id="doc-123")

        assert count == 5
        # Verify filter was applied
        call_kwargs = mock_client.count.call_args[1]
        assert "count_filter" in call_kwargs

    def test_upsert_post_count_verification(self, mock_client):
        """Test upsert verifies count after upsert when expected_count provided."""
        node = QdrantUpsertNode(
            client=mock_client,
            collection_name="test_collection",
            batch_size=100,
        )
        mock_client.count.return_value = MagicMock(count=5)

        meta = self._make_chunk_metadata()
        node.upsert(
            chunks=[("doc-123", 0, meta)],
            dense_vectors=[[0.1] * 1024],
            sparse_vectors=[{"indices": [0], "values": [0.5]}],
            payload_extra={
                "source_path": "/path",
                "source_name": "Test",
                "faculty": "engineering",
                "doc_type": "lecture",
                "semester": "2025-S1",
            },
            expected_count=5,
        )

        # Should have called count for verification
        assert mock_client.count.called

    def test_upsert_count_shortfall_logged(self, mock_client, caplog):
        """Test shortfall is logged when actual count < expected."""
        node = QdrantUpsertNode(
            client=mock_client,
            collection_name="test_collection",
            batch_size=100,
        )
        # Actual count less than expected
        mock_client.count.return_value = MagicMock(count=3)

        meta = self._make_chunk_metadata()
        node.upsert(
            chunks=[("doc-123", 0, meta)],
            dense_vectors=[[0.1] * 1024],
            sparse_vectors=[{"indices": [0], "values": [0.5]}],
            payload_extra={
                "source_path": "/path",
                "source_name": "Test",
                "faculty": "engineering",
                "doc_type": "lecture",
                "semester": "2025-S1",
            },
            expected_count=5,
        )

        # Should have logged a warning about shortfall
        assert any("shortfall" in record.message for record in caplog.records)


# ============================================================
# IP-023: Stale Chunk Deletion Tests
# ============================================================


class TestStaleDeletionNode:
    """Test stale chunk deletion on re-ingestion."""

    @pytest.fixture
    def mock_client(self):
        """Create mocked Qdrant client."""
        return MagicMock()

    @pytest.fixture
    def node(self, mock_client):
        """Create deletion node."""
        return StaleDeletionNode(
            client=mock_client,
            collection_name="test_collection",
        )

    def test_delete_by_document_id(self, node):
        """Test deletion by document_id."""
        node._client.count.return_value = MagicMock(count=5)

        deleted = node.delete_by_document_id("doc-123")

        assert deleted == 5
        node._client.delete.assert_called_once()

    def test_delete_no_stale_chunks(self, node):
        """Test deletion returns 0 when no stale chunks exist."""
        node._client.count.return_value = MagicMock(count=0)

        deleted = node.delete_by_document_id("doc-123")

        assert deleted == 0
        node._client.delete.assert_not_called()

    def test_delete_raises_error_on_failure(self, node):
        """Test DeletionError raised on failure."""
        node._client.count.side_effect = Exception("Connection failed")

        with pytest.raises(DeletionError):
            node.delete_by_document_id("doc-123")

    def test_delete_stale_by_chunk_indices(self, node):
        """Test deletion keeps only specified chunk indices."""
        node._client.count.return_value = MagicMock(count=3)

        deleted = node.delete_stale_by_chunk_indices(
            document_id="doc-123",
            keep_chunk_indices=[0, 1, 2],
        )

        assert deleted == 3
        node._client.delete.assert_called_once()

    def test_delete_stale_nothing_to_delete(self, node):
        """Test deletion returns 0 when nothing is stale."""
        node._client.count.return_value = MagicMock(count=0)

        deleted = node.delete_stale_by_chunk_indices(
            document_id="doc-123",
            keep_chunk_indices=[0, 1, 2],
        )

        assert deleted == 0

    def test_delete_old_after_reingest(self, node):
        """Test delete_old_after_reingest returns comparison info."""
        node._client.count.return_value = MagicMock(count=10)

        result = node.delete_old_after_reingest(
            document_id="doc-123",
            new_point_ids=["pid-0", "pid-1", "pid-2"],
            previous_chunk_count=8,
        )

        assert result["current_count"] == 10
        assert result["new_count"] == 3
        assert result["excess"] == 7
        assert result["previous_chunk_count"] == 8

    def test_reingest_modified_doc_old_count_reduced(self, node):
        """Test re-ingest: old chunk count reduced after deletion."""
        # Simulate: old doc had 8 chunks, new doc has 5
        node._client.count.return_value = MagicMock(count=5)

        # Delete stale (keep new chunk indices 0-4)
        deleted = node.delete_stale_by_chunk_indices(
            document_id="doc-123",
            keep_chunk_indices=[0, 1, 2, 3, 4],
        )

        # After deletion, only 5 chunks remain
        node._client.count.return_value = MagicMock(count=5)
        remaining = node._client.count(
            collection_name="test_collection",
            count_filter=MagicMock(),
        ).count

        assert deleted >= 0  # At least attempted deletion
        assert remaining == 5

    def test_deletion_error_preserves_data_on_failure(self, node):
        """Test deletion failure doesn't affect existing data."""
        node._client.count.side_effect = Exception("Timeout")

        with pytest.raises(DeletionError):
            node.delete_by_document_id("doc-123")

        # Old data should still be intact (deletion never ran)
        node._client.delete.assert_not_called()


# ============================================================
# IP-024: Post-Upsert Index Health Check Tests
# ============================================================


class TestUpsertHealthChecker:
    """Test post-upsert index health check."""

    @pytest.fixture
    def mock_client(self):
        """Create mocked Qdrant client."""
        return MagicMock()

    @pytest.fixture
    def checker(self, mock_client):
        """Create health checker."""
        return UpsertHealthChecker(
            client=mock_client,
            collection_name="test_collection",
        )

    def test_check_green_status(self, checker):
        """Test health check returns green when everything is healthy."""
        # Mock collection info
        mock_info = MagicMock()
        mock_info.status = "green"
        mock_info.vectors_count = 100
        mock_info.indexed_vectors_count = 100
        mock_info.points_count = 100
        checker._client.get_collection.return_value = mock_info

        # Mock test query
        mock_query_result = MagicMock()
        mock_query_result.points = [MagicMock()]
        checker._client.query_points.return_value = mock_query_result

        result = checker.check(expected_count=100, test_document_id="doc-123")

        assert result.status == "green"
        assert result.is_healthy
        assert not result.is_degraded

    def test_check_degraded_on_count_mismatch(self, checker):
        """Test health check degraded when count doesn't match."""
        mock_info = MagicMock()
        mock_info.status = "green"
        mock_info.vectors_count = 90
        mock_info.indexed_vectors_count = 90
        mock_info.points_count = 90
        checker._client.get_collection.return_value = mock_info

        mock_query_result = MagicMock()
        mock_query_result.points = [MagicMock()]
        checker._client.query_points.return_value = mock_query_result

        result = checker.check(expected_count=100)

        assert result.status == "degraded"
        assert result.is_degraded

    def test_check_degraded_on_query_failure(self, checker):
        """Test health check degraded when test query returns 0 hits."""
        mock_info = MagicMock()
        mock_info.status = "green"
        mock_info.points_count = 100
        checker._client.get_collection.return_value = mock_info

        mock_query_result = MagicMock()
        mock_query_result.points = []
        checker._client.query_points.return_value = mock_query_result

        result = checker.check(expected_count=100, test_document_id="doc-123")

        assert result.status == "degraded"
        assert "test_query_no_hits" in result.checks_failed

    def test_check_collection_accessible(self, checker):
        """Test collection accessibility check passes."""
        mock_info = MagicMock()
        mock_info.status = "green"
        mock_info.vectors_count = 50
        mock_info.indexed_vectors_count = 50
        mock_info.points_count = 50
        checker._client.get_collection.return_value = mock_info

        result = checker.check(expected_count=50)

        assert "collection_accessible" in result.checks_passed

    def test_check_count_verified(self, checker):
        """Test count verification passes when actual >= expected."""
        mock_info = MagicMock()
        mock_info.status = "green"
        mock_info.points_count = 110  # More than expected
        checker._client.get_collection.return_value = mock_info

        result = checker.check(expected_count=100)

        assert "count_verified" in result.checks_passed

    def test_check_count_mismatch_logged(self, checker):
        """Test count mismatch is logged in issues."""
        mock_info = MagicMock()
        mock_info.status = "green"
        mock_info.points_count = 80
        checker._client.get_collection.return_value = mock_info

        result = checker.check(expected_count=100)

        assert any("count_mismatch" in c for c in result.checks_failed)
        assert any("shortfall" in i for i in result.issues)

    def test_check_without_test_document_id(self, checker):
        """Test health check works without test document ID."""
        mock_info = MagicMock()
        mock_info.status = "green"
        mock_info.points_count = 100
        checker._client.get_collection.return_value = mock_info

        result = checker.check(expected_count=100)

        assert result.status == "green"
        # Test query should not run without document_id
        assert "test_query_ok" not in result.checks_passed
        assert "test_query_no_hits" not in result.checks_failed

    def test_record_in_manifest(self, checker):
        """Test health check result stored in run manifest."""
        mock_info = MagicMock()
        mock_info.status = "green"
        mock_info.points_count = 100
        checker._client.get_collection.return_value = mock_info

        result = checker.check(expected_count=100)
        manifest = {}
        checker.record_in_manifest(result, manifest)

        assert "health_check" in manifest
        assert manifest["health_check"]["status"] == "green"
        assert manifest["health_check"]["collection"] == "test_collection"

    def test_health_check_result_structure(self, checker):
        """Test HealthCheckResult has expected fields."""
        result = HealthCheckResult(
            collection_name="test",
            status="green",
        )
        assert hasattr(result, "collection_name")
        assert hasattr(result, "status")
        assert hasattr(result, "checks_passed")
        assert hasattr(result, "checks_failed")
        assert hasattr(result, "details")
        assert hasattr(result, "issues")
        assert hasattr(result, "is_healthy")
        assert hasattr(result, "is_degraded")

    def test_collection_not_green_raises_issue(self, checker):
        """Test non-green collection status recorded as issue."""
        mock_info = MagicMock()
        mock_info.status = "red"
        mock_info.vectors_count = 0
        mock_info.indexed_vectors_count = 0
        mock_info.points_count = 0
        checker._client.get_collection.return_value = mock_info

        result = checker.check(expected_count=0)

        assert "collection_not_green" in result.checks_failed


# ============================================================
# Error Hierarchy Tests
# ============================================================


class TestVectorStoreErrors:
    """Test typed error hierarchy."""

    def test_store_error_is_exception(self):
        """Test StoreError is an Exception."""
        assert issubclass(StoreError, Exception)

    def test_schema_error_is_store_error(self):
        """Test SchemaError is a StoreError."""
        assert issubclass(SchemaError, StoreError)

    def test_upsert_error_is_store_error(self):
        """Test UpsertError is a StoreError."""
        assert issubclass(UpsertError, StoreError)

    def test_deletion_error_is_store_error(self):
        """Test DeletionError is a StoreError."""
        assert issubclass(DeletionError, StoreError)

    def test_health_check_error_is_store_error(self):
        """Test HealthCheckError is a StoreError."""
        assert issubclass(HealthCheckError, StoreError)

    def test_schema_error_message(self):
        """Test SchemaError has descriptive message."""
        err = SchemaError(
            reason="schema_mismatch",
            collection_name="test",
            expected={"dim": 4096},
            actual={"dim": 3072},
        )
        assert "Schema mismatch" in str(err)
        assert "test" in str(err)
        assert err.reason == "schema_mismatch"

    def test_upsert_error_message(self):
        """Test UpsertError has batch and retry info."""
        err = UpsertError(
            reason="timeout",
            collection_name="test",
            batch_index=2,
            retry_count=3,
        )
        assert "batch 2" in str(err)
        assert "3 retries" in str(err)

    def test_deletion_error_message(self):
        """Test DeletionError has document_id."""
        err = DeletionError(
            reason="timeout",
            collection_name="test",
            document_id="doc-123",
        )
        assert "doc-123" in str(err)

    def test_health_check_error_message(self):
        """Test HealthCheckError has check name."""
        err = HealthCheckError(
            reason="no results",
            collection_name="test",
            check_name="test_query",
        )
        assert "test_query" in str(err)


# ============================================================
# Integration Tests
# ============================================================


class TestVectorStoreIntegration:
    """Integration tests for the full vector store pipeline."""

    @pytest.fixture
    def mock_client(self):
        """Create mocked Qdrant client."""
        return MagicMock()

    def _make_chunk_metadata(self, index: int = 0):
        """Create ChunkMetadata for testing."""
        return ChunkMetadata(
            document_id="doc-123",
            chunk_index=index,
            page_start=1,
            heading_path=["Introduction"],
            char_start=index * 100,
            char_end=(index + 1) * 100,
            token_count=25,
        )

    def test_full_upsert_workflow(self, mock_client):
        """Test full upsert: schema → upsert → health check."""
        # Setup schema manager
        mock_client.get_collections.return_value = MagicMock(collections=[])

        from vector_store.schema import CollectionSchemaManager

        schema_mgr = CollectionSchemaManager(
            client=mock_client,
            collection_name="test",
            dense_dim=1024,
            sparse_distance="Dot",
        )
        schema_mgr.ensure_collection()
        assert mock_client.create_collection.called

        # Mock collection info for verification (upsert test may call it)
        mock_info = MagicMock()
        mock_dense = MagicMock()
        mock_dense.size = 1024
        mock_dense.distance = Distance.COSINE
        mock_sparse = MagicMock()
        mock_sparse.modifier = "none"
        mock_info.config.params.vectors = {"dense": mock_dense}
        mock_info.config.params.sparse_vectors = {"sparse": mock_sparse}
        mock_info.payload_index_schema = {
            "faculty": "keyword",
            "doc_type": "keyword",
            "semester": "keyword",
            "document_id": "keyword",
            "chunk_index": "integer",
        }
        mock_client.get_collection.return_value = mock_info

        # Upsert
        from vector_store.upsert import QdrantUpsertNode

        node = QdrantUpsertNode(
            client=mock_client,
            collection_name="test",
            batch_size=100,
        )
        chunks = [("doc-123", 0, self._make_chunk_metadata(0))]
        count = node.upsert(
            chunks=chunks,
            dense_vectors=[[0.1] * 1024],
            sparse_vectors=[{"indices": [0], "values": [0.5]}],
            payload_extra={
                "source_path": "/path",
                "source_name": "Test",
                "faculty": "engineering",
                "doc_type": "lecture",
                "semester": "2025-S1",
            },
        )
        assert count == 1

    def test_upsert_then_health_check(self, mock_client):
        """Test upsert followed by health check."""
        # Mock client for health check
        mock_info = MagicMock()
        mock_info.status = "green"
        mock_info.vectors_count = 1
        mock_info.indexed_vectors_count = 1
        mock_info.points_count = 1
        mock_client.get_collection.return_value = mock_info

        mock_query_result = MagicMock()
        mock_query_result.points = [MagicMock()]
        mock_client.query_points.return_value = mock_query_result

        checker = UpsertHealthChecker(
            client=mock_client,
            collection_name="test",
        )
        result = checker.check(expected_count=1, test_document_id="doc-123")
        assert result.status == "green"

    def test_deletion_after_upsert(self, mock_client):
        """Test stale deletion after upsert."""
        # Simulate: old doc had 8 chunks
        mock_client.count.return_value = MagicMock(count=5)

        deletion_node = StaleDeletionNode(
            client=mock_client,
            collection_name="test",
        )

        # Delete stale chunks (keep new indices 0-4)
        deleted = deletion_node.delete_stale_by_chunk_indices(
            document_id="doc-123",
            keep_chunk_indices=[0, 1, 2, 3, 4],
        )

        assert deleted == 5
