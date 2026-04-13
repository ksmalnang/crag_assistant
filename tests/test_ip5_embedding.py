"""Unit tests for IP5 - Embedding Epic (IP-017 to IP-020)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from embedding.cache import EmbeddingCache
from embedding.dense_node import DenseEmbeddingNode
from embedding.errors import (
    DegenerateVectorError,
    DenseVectorNaNError,
    DenseVectorNormError,
    SparseVectorEmptyError,
)
from embedding.quality import EmbeddingQualityChecker
from embedding.sparse_node import SparseEmbeddingNode

# ============================================================
# IP-017: Dense Embedding Node Tests (OpenRouter qwen3-embedding-8b)
# ============================================================


class TestDenseEmbeddingNode:
    """Test dense embedding generation via OpenRouter."""

    def _make_mock_embedding_response(self, texts, dim=4096):
        """Create a mock OpenAI embeddings API response."""
        mock_response = MagicMock()
        mock_data_list = []
        for i, _ in enumerate(texts):
            mock_data = MagicMock()
            mock_data.embedding = [0.1 + i * 0.01] * dim
            mock_data_list.append(mock_data)
        mock_response.data = mock_data_list
        return mock_response

    @pytest.fixture
    def mock_client(self):
        """Create a mocked OpenAI client."""
        client = MagicMock()
        client.embeddings.create = MagicMock(
            return_value=self._make_mock_embedding_response(["test"])
        )
        return client

    def test_model_is_qwen3_embedding_8b(self):
        """Test default model is qwen/qwen3-embedding-8b."""
        node = DenseEmbeddingNode()
        assert node.model_name == "qwen/qwen3-embedding-8b"

    def test_no_fallback_configured(self):
        """Test that no fallback mechanism exists."""
        node = DenseEmbeddingNode()
        # Should not have openai_fallback or _openai_client attributes
        assert not hasattr(node, "openai_fallback")

    def test_batch_size_configurable(self):
        """Test batch size is configurable."""
        node = DenseEmbeddingNode(batch_size=16)
        assert node.batch_size == 16

    def test_embed_single_returns_4096_dim(self, mock_client):
        """Test single embedding produces 4096-dimensional vector."""
        with patch("embedding.dense_node.OpenAI", return_value=mock_client):
            node = DenseEmbeddingNode()
            node._get_client()  # Trigger client init
            mock_client.embeddings.create.return_value = (
                self._make_mock_embedding_response(["test"], dim=4096)
            )
            vector = node.embed_single("CS101 Introduction to Computer Science")
            assert len(vector) == 4096

    def test_embed_batch_returns_valid_vectors(self, mock_client):
        """Test batch embedding produces valid vectors for all inputs."""
        with patch("embedding.dense_node.OpenAI", return_value=mock_client):
            node = DenseEmbeddingNode()
            node._get_client()
            texts = [
                "CS101 Introduction to Computer Science",
                "MATH201 Linear Algebra",
                "PHYS101 Classical Mechanics",
            ]
            mock_client.embeddings.create.return_value = (
                self._make_mock_embedding_response(texts, dim=4096)
            )
            vectors = node.embed(texts)
            assert len(vectors) == 3
            for vec in vectors:
                assert len(vec) == 4096

    def test_batching_splits_correctly(self, mock_client):
        """Test large batches are split according to batch_size."""
        with patch("embedding.dense_node.OpenAI", return_value=mock_client):
            node = DenseEmbeddingNode(batch_size=2)
            node._get_client()
            texts = [f"Text {i}" for i in range(5)]

            # Mock returns different vectors per call to track batching
            def mock_create(input, model):
                mock_response = MagicMock()
                mock_response.data = [
                    MagicMock(embedding=[0.1 + i * 0.01] * 4096)
                    for i in range(len(input))
                ]
                return mock_response

            mock_client.embeddings.create.side_effect = mock_create
            vectors = node.embed(texts)
            # Should have been called 3 times (2+2+1)
            assert mock_client.embeddings.create.call_count == 3
            assert len(vectors) == 5

    def test_empty_input_returns_empty_list(self):
        """Test empty input produces no embeddings."""
        node = DenseEmbeddingNode()
        result = node.embed([])
        assert result == []

    def test_chunk_ids_must_match_text_count(self):
        """Test texts and chunk_ids must have same length."""
        node = DenseEmbeddingNode()
        with pytest.raises(ValueError, match="same length"):
            node.embed(["text1", "text2"], ["id1"])

    def test_auto_generated_chunk_ids(self, mock_client):
        """Test chunk IDs are auto-generated if not provided."""
        with patch("embedding.dense_node.OpenAI", return_value=mock_client):
            node = DenseEmbeddingNode()
            node._get_client()
            mock_client.embeddings.create.return_value = (
                self._make_mock_embedding_response(["a", "b"], dim=4096)
            )
            vectors = node.embed(["text1", "text2"])
            assert len(vectors) == 2

    def test_metadata_not_included_in_input(self, mock_client):
        """Test that only chunk text is used for embedding, not metadata."""
        with patch("embedding.dense_node.OpenAI", return_value=mock_client):
            node = DenseEmbeddingNode()
            node._get_client()
            mock_client.embeddings.create.return_value = (
                self._make_mock_embedding_response(["text"], dim=4096)
            )
            vector = node.embed_single("Pure chunk text only")
            assert len(vector) == 4096
            # Verify the API was called with only the text
            call_args = mock_client.embeddings.create.call_args
            assert call_args.kwargs["input"] == ["Pure chunk text only"]

    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert DenseEmbeddingNode.cosine_similarity(a, b) == pytest.approx(1.0)

        c = [0.0, 1.0, 0.0]
        assert DenseEmbeddingNode.cosine_similarity(a, c) == pytest.approx(0.0)

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors is 0."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        similarity = DenseEmbeddingNode.cosine_similarity(a, b)
        assert similarity == 0.0

    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector returns 0."""
        a = [0.0, 0.0]
        b = [1.0, 1.0]
        similarity = DenseEmbeddingNode.cosine_similarity(a, b)
        assert similarity == 0.0


# ============================================================
# IP-018: Sparse Embedding Node Tests
# ============================================================


class TestSparseEmbeddingNode:
    """Test sparse embedding generation using BM25."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for vocab storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def corpus_texts(self):
        """Create corpus texts for fitting."""
        return [
            "CS101 Introduction to Computer Science",
            "CS102 Data Structures and Algorithms",
            "MATH201 Linear Algebra and Calculus",
            "PHYS101 Classical Mechanics",
            "CS101 Programming Lab Session",
        ]

    @pytest.fixture
    def node(self, temp_dir, corpus_texts):
        """Create sparse embedding node fitted on corpus."""
        node = SparseEmbeddingNode(
            model_name="Qdrant/bm25",
            vocab_dir=temp_dir,
            collection_name="test_collection",
        )
        node.fit(corpus_texts, force=True)
        return node

    def test_sparse_vector_format(self, node):
        """Test sparse vectors are {indices: list[int], values: list[float]}."""
        result = node.embed_single("CS101 algorithms")
        assert "indices" in result
        assert "values" in result
        assert isinstance(result["indices"], list)
        assert isinstance(result["values"], list)
        # All indices should be ints, values should be floats
        for idx in result["indices"]:
            assert isinstance(idx, int)
        for val in result["values"]:
            assert isinstance(val, float)

    def test_cs101_higher_sparse_score(self, node):
        """Test chunk containing 'CS101' retrieves higher sparse score."""
        results = node.embed(
            [
                "CS101 Introduction to Computer Science",
                "MATH201 Linear Algebra and Calculus",
            ]
        )
        cs101_result = results[0]
        math_result = results[1]

        # CS101 chunk should have more significant BM25 entries for that term
        cs101_max = max(cs101_result["values"]) if cs101_result["values"] else 0.0
        math_max = max(math_result["values"]) if math_result["values"] else 0.0
        # CS101-specific term should score higher in CS101 text
        assert cs101_max >= math_max, (
            "CS101 chunk should have equal or higher sparse score for CS101 term"
        )

    def test_vocab_persisted_to_file(self, temp_dir, corpus_texts):
        """Test vocabulary is saved to data/bm25_vocab/{collection_name}.pkl."""
        vocab_path = Path(temp_dir) / "bm25_vocab" / "test_collection.pkl"
        node = SparseEmbeddingNode(
            vocab_dir=str(Path(temp_dir) / "bm25_vocab"),
            collection_name="test_collection",
        )
        node.fit(corpus_texts, force=True)
        assert vocab_path.exists(), f"Vocab not persisted at {vocab_path}"

    def test_vocab_loaded_from_cache(self, temp_dir, corpus_texts):
        """Test vocabulary is loaded from persisted cache on second run."""
        vocab_dir = str(Path(temp_dir) / "bm25_vocab")
        node1 = SparseEmbeddingNode(vocab_dir=vocab_dir, collection_name="cached_test")
        node1.fit(corpus_texts, force=True)

        # Second run should load from cache
        node2 = SparseEmbeddingNode(vocab_dir=vocab_dir, collection_name="cached_test")
        node2.fit(corpus_texts)  # Should load from cache, not re-fit
        assert node2._is_fitted

    def test_batch_embed(self, node):
        """Test batch embedding produces valid sparse vectors."""
        texts = ["CS101", "MATH201", "PHYS101"]
        results = node.embed(texts)
        assert len(results) == 3
        for result in results:
            assert "indices" in result
            assert "values" in result

    def test_embed_without_fit_raises_error(self, temp_dir):
        """Test embedding before fitting raises RuntimeError."""
        node = SparseEmbeddingNode(vocab_dir=temp_dir, collection_name="unfitted")
        # Don't fit
        with pytest.raises(RuntimeError, match="not fitted"):
            node.embed(["test"])

    def test_collection_name_required_for_persistence(self, temp_dir):
        """Test collection_name is required for vocab persistence."""
        node = SparseEmbeddingNode(vocab_dir=temp_dir)
        with pytest.raises(ValueError, match="collection_name required"):
            node.fit(["test"])


# ============================================================
# IP-019: Embedding Cache Tests
# ============================================================


class TestEmbeddingCache:
    """Test SQLite-based embedding cache."""

    @pytest.fixture
    def cache_path(self):
        """Create temporary cache file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield str(Path(tmpdir) / "test_cache.db")

    @pytest.fixture
    def cache(self, cache_path):
        """Create embedding cache."""
        c = EmbeddingCache(cache_path=cache_path, ttl_days=30)
        yield c
        c.close()  # Ensure connection is closed before temp cleanup

    @pytest.fixture
    def sample_dense(self):
        """Sample dense vector."""
        return [0.1] * 1024

    @pytest.fixture
    def sample_sparse(self):
        """Sample sparse vector."""
        return {"indices": [0.0, 1.0, 2.0], "values": [0.5, 0.3, 0.2]}

    def test_put_and_get(self, cache, sample_dense, sample_sparse):
        """Test cache stores and retrieves embeddings."""
        chunk_hash = "test-hash-123"
        cache.put(
            chunk_hash,
            sample_dense,
            sample_sparse["indices"],
            sample_sparse["values"],
        )
        result = cache.get(chunk_hash)
        assert result is not None
        assert len(result["dense_vector"]) == len(sample_dense)
        assert result["dim"] == 1024

    def test_cache_miss_returns_none(self, cache):
        """Test cache miss returns None."""
        result = cache.get("nonexistent-hash")
        assert result is None

    def test_cache_hit_miss_stats(self, cache, sample_dense, sample_sparse):
        """Test cache hit/miss ratio is tracked."""
        cache.put(
            "hash1", sample_dense, sample_sparse["indices"], sample_sparse["values"]
        )
        cache.get("hash1")  # hit
        cache.get("hash2")  # miss
        cache.get("hash3")  # miss

        stats = cache.stats()
        assert stats["hit_count"] == 1
        assert stats["miss_count"] == 2
        assert abs(stats["hit_ratio"] - 1 / 3) < 0.01

    def test_cache_ttl_expiration(self, cache_path, sample_dense, sample_sparse):
        """Test cache entries older than TTL are invalidated."""
        import time

        cache = EmbeddingCache(cache_path=cache_path, ttl_days=30)
        try:
            chunk_hash = "expiring-hash"
            cache.put(
                chunk_hash,
                sample_dense,
                sample_sparse["indices"],
                sample_sparse["values"],
            )

            # Manually set cached_at to expired time
            conn = cache._connect()
            expired_time = time.time() - (31 * 86400)  # 31 days ago
            conn.execute(
                "UPDATE embedding_cache SET cached_at = ? WHERE chunk_hash = ?",
                (expired_time, chunk_hash),
            )
            conn.commit()

            # Should be treated as miss
            result = cache.get(chunk_hash)
            assert result is None
        finally:
            cache.close()

    def test_delete_removes_entry(self, cache, sample_dense, sample_sparse):
        """Test delete removes cached entry."""
        chunk_hash = "delete-me"
        cache.put(
            chunk_hash,
            sample_dense,
            sample_sparse["indices"],
            sample_sparse["values"],
        )
        cache.delete(chunk_hash)
        result = cache.get(chunk_hash)
        assert result is None

    def test_invalidate_expired_entries(self, cache_path, sample_dense, sample_sparse):
        """Test invalidate_expired removes old entries."""
        import time

        cache = EmbeddingCache(cache_path=cache_path, ttl_days=30)
        try:
            cache.put(
                "hash1", sample_dense, sample_sparse["indices"], sample_sparse["values"]
            )
            cache.put(
                "hash2", sample_dense, sample_sparse["indices"], sample_sparse["values"]
            )

            # Expire hash2
            conn = cache._connect()
            expired_time = time.time() - (31 * 86400)
            conn.execute(
                "UPDATE embedding_cache SET cached_at = ? WHERE chunk_hash = ?",
                (expired_time, "hash2"),
            )
            conn.commit()

            deleted = cache.invalidate_expired()
            assert deleted == 1
            assert cache.get("hash1") is not None
            assert cache.get("hash2") is None
        finally:
            cache.close()

    def test_context_manager(self, cache_path):
        """Test cache works as context manager."""
        with EmbeddingCache(cache_path=cache_path) as cache:
            cache.put("hash1", [0.1] * 1024, [0.0], [1.0])
            result = cache.get("hash1")
            assert result is not None

    def test_serialise_deserialise_vector_roundtrip(self, cache):
        """Test vector serialisation/deserialisation preserves values."""
        original = [0.1, 0.2, 0.3, -0.5, 100.0]
        serialised = cache._serialise_vector(original)
        deserialised = cache._deserialise_vector(serialised)
        assert deserialised == pytest.approx(original, abs=1e-6)


# ============================================================
# IP-020: Embedding Quality Sanity Check Tests
# ============================================================


class TestEmbeddingQualityChecker:
    """Test embedding quality validation."""

    @pytest.fixture
    def checker(self):
        """Create quality checker."""
        return EmbeddingQualityChecker()

    def test_valid_dense_vector_passes(self, checker):
        """Test valid dense vector passes all checks."""
        vector = [0.1] * 4096
        result = checker.check_dense(vector, "valid-chunk")
        assert result.passed
        assert result.failures == []

    def test_zero_vector_rejected(self, checker):
        """Check 1: dense vector norm > 0.01 — zero vectors rejected."""
        vector = [0.0] * 4096
        result = checker.check_dense(vector, "zero-chunk")
        assert not result.passed
        assert any("norm_too_low" in f for f in result.failures)

    def test_nan_vector_rejected(self, checker):
        """Check 2: no NaN values in dense vector."""
        vector = [0.1] * 4095 + [float("nan")]
        result = checker.check_dense(vector, "nan-chunk")
        assert not result.passed
        assert any("NaN" in f for f in result.failures)

    def test_inf_vector_rejected(self, checker):
        """Check 2: no Inf values in dense vector."""
        vector = [0.1] * 4095 + [float("inf")]
        result = checker.check_dense(vector, "inf-chunk")
        assert not result.passed
        assert any("Inf" in f for f in result.failures)

    def test_valid_sparse_vector_passes(self, checker):
        """Test valid sparse vector passes."""
        sparse = {"indices": [0, 1, 2], "values": [0.5, 0.3, 0.2]}
        result = checker.check_sparse(sparse, "valid-sparse")
        assert result.passed

    def test_empty_sparse_vector_rejected(self, checker):
        """Check 3: sparse vector has at least 1 non-zero entry."""
        sparse_empty = {"indices": [], "values": []}
        result = checker.check_sparse(sparse_empty, "empty-sparse")
        assert not result.passed
        assert any("no_non_zero_entries" in f for f in result.failures)

    def test_zero_value_sparse_vector_rejected(self, checker):
        """Test sparse vector with all zero values is rejected."""
        sparse_zeros = {"indices": [0, 1, 2], "values": [0.0, 0.0, 0.0]}
        result = checker.check_sparse(sparse_zeros, "zero-sparse")
        assert not result.passed

    def test_combined_check_passes(self, checker):
        """Test combined check passes for valid vectors."""
        dense = [0.1] * 4096
        sparse = {"indices": [0, 1], "values": [0.5, 0.3]}
        result = checker.check(dense, sparse, "valid-chunk")
        assert result.passed

    def test_combined_check_raises_on_dense_failure(self, checker):
        """Test combined check raises DegenerateVectorError on dense failure."""
        dense = [0.0] * 4096
        sparse = {"indices": [0, 1], "values": [0.5, 0.3]}
        with pytest.raises(DegenerateVectorError):
            checker.check(dense, sparse, "bad-dense")

    def test_combined_check_raises_on_sparse_failure(self, checker):
        """Test combined check raises DegenerateVectorError on sparse failure."""
        dense = [0.1] * 4096
        sparse = {"indices": [], "values": []}
        with pytest.raises(DegenerateVectorError):
            checker.check(dense, sparse, "bad-sparse")

    def test_failure_rate_logged_and_tracked(self, checker):
        """Test failure rate is tracked across multiple checks."""
        # 1 valid, 1 invalid
        checker.check_dense([0.1] * 4096, "valid")
        checker.check_dense([0.0] * 4096, "invalid")

        stats = checker.stats()
        assert stats["total_checked"] == 2
        assert stats["total_failed"] == 1

    def test_alert_threshold_triggered(self, checker):
        """Test alert is triggered when > 1% of batch fails."""
        # Use very low threshold to trigger alert easily
        strict_checker = EmbeddingQualityChecker(alert_threshold=0.01)

        # 100 checks, 2 fail (2% > 1%)
        for i in range(98):
            strict_checker.check_dense([0.1] * 4096, f"valid-{i}")
        for i in range(2):
            strict_checker.check_dense([0.0] * 4096, f"invalid-{i}")

        stats = strict_checker.stats()
        assert stats["alert_triggered"]
        assert stats["failure_rate"] == 0.02

    def test_raise_on_failure_norm_error(self, checker):
        """Test raise_on_failure raises DenseVectorNormError for low norm."""
        result = checker.check_dense([0.0] * 4096, "zero-chunk")
        with pytest.raises(DenseVectorNormError):
            checker.raise_on_failure(result, "zero-chunk")

    def test_raise_on_failure_nan_error(self, checker):
        """Test raise_on_failure raises DenseVectorNaNError for NaN."""
        vector = [0.1] * 4095 + [float("nan")]
        result = checker.check_dense(vector, "nan-chunk")
        with pytest.raises(DenseVectorNaNError):
            checker.raise_on_failure(result, "nan-chunk")

    def test_raise_on_sparse_failure_error(self, checker):
        """Test raise_on_sparse_failure raises SparseVectorEmptyError."""
        result = checker.check_sparse({"indices": [], "values": []}, "empty-sparse")
        with pytest.raises(SparseVectorEmptyError):
            checker.raise_on_sparse_failure(result, "empty-sparse")

    def test_quality_check_result_structure(self, checker):
        """Test QualityCheckResult has expected fields."""
        result = checker.check_dense([0.1] * 4096, "test")
        assert hasattr(result, "chunk_id")
        assert hasattr(result, "passed")
        assert hasattr(result, "failures")
        assert hasattr(result, "details")
        assert result.chunk_id == "test"
        assert result.passed is True


# ============================================================
# Integration Tests: Full Embedding Pipeline
# ============================================================


class TestEmbeddingPipelineIntegration:
    """Integration tests for the full embedding pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def _make_mock_embedding_response(self, texts, dim=4096):
        """Create a mock OpenAI embeddings API response."""
        mock_response = MagicMock()
        mock_data_list = []
        for i, _ in enumerate(texts):
            mock_data = MagicMock()
            mock_data.embedding = [0.1 + i * 0.01] * dim
            mock_data_list.append(mock_data)
        mock_response.data = mock_data_list
        return mock_response

    def test_dense_then_quality_check(self, temp_dir):
        """Test dense embedding generation followed by quality check."""
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = self._make_mock_embedding_response(
            ["CS101 Introduction to Computer Science"], dim=4096
        )
        with patch("embedding.dense_node.OpenAI", return_value=mock_client):
            node = DenseEmbeddingNode()
            node._get_client()
            vectors = node.embed(["CS101 Introduction to Computer Science"])
            assert len(vectors) == 1

            # Quality check should pass
            checker = EmbeddingQualityChecker()
            sparse_dummy = {"indices": [0, 1], "values": [0.5, 0.3]}
            result = checker.check(vectors[0], sparse_dummy, "chunk-0")
            assert result.passed

    def test_cache_integration(self, temp_dir):
        """Test cache stores and retrieves embedding results."""
        cache_path = str(Path(temp_dir) / "test.db")
        cache = EmbeddingCache(cache_path=cache_path)
        try:
            dense = [0.1] * 4096
            sparse = {"indices": [0.0, 1.0], "values": [0.5, 0.3]}

            cache.put("hash-1", dense, sparse["indices"], sparse["values"])
            result = cache.get("hash-1")

            assert result is not None
            assert len(result["dense_vector"]) == 4096
        finally:
            cache.close()

    def test_full_flow_embed_cache_quality(self, temp_dir):
        """Test full flow: embed -> quality check -> cache -> retrieve."""
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = self._make_mock_embedding_response(
            ["CS101 Computer Science"], dim=4096
        )
        with patch("embedding.dense_node.OpenAI", return_value=mock_client):
            # Dense
            node = DenseEmbeddingNode()
            node._get_client()
            dense_vec = node.embed_single("CS101 Computer Science")

            # Sparse (manual for integration test)
            sparse_vec = {"indices": [0.0, 1.0, 2.0], "values": [0.5, 0.3, 0.1]}

            # Quality check
            checker = EmbeddingQualityChecker()
            result = checker.check(dense_vec, sparse_vec, "chunk-0")
            assert result.passed

            # Cache
            cache_path = str(Path(temp_dir) / "full_flow.db")
            cache = EmbeddingCache(cache_path=cache_path)
            cache.put("chunk-0", dense_vec, sparse_vec["indices"], sparse_vec["values"])

            # Retrieve
            cached = cache.get("chunk-0")
            assert cached is not None
            assert len(cached["dense_vector"]) == 4096
            cache.close()
