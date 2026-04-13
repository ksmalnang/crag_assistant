"""Tests for IP-029: Typed error hierarchy."""

import json

from ingestion.errors_base import (
    ChunkError,
    EmbeddingNodeError,
    HealthCheckNodeError,
    IngestionError,
    IntakeNodeError,
    MetadataError,
    ParseNodeError,
    SchemaNodeError,
    UpsertNodeError,
    error_from_exception,
    serialise_errors,
)

# ─── IngestionError Tests ─────────────────────────────────────────────────────


class TestIngestionError:
    """Tests for the base IngestionError class."""

    def test_basic_error_fields(self):
        """IngestionError should have all required fields."""
        err = IngestionError(
            document_id="doc-123",
            file_path="/path/to/file.pdf",
            node="intake_node",
            reason="unsupported_format",
            details={"mime": "application/xyz"},
            message="File format not supported",
        )

        assert err.document_id == "doc-123"
        assert err.file_path == "/path/to/file.pdf"
        assert err.node == "intake_node"
        assert err.reason == "unsupported_format"
        assert err.details == {"mime": "application/xyz"}
        assert err.message == "File format not supported"

    def test_auto_generated_message(self):
        """Message should be auto-generated if not provided."""
        err = IngestionError(
            file_path="/path/to/file.pdf",
            node="parser_node",
            reason="parse_timeout",
        )

        assert "parser_node" in err.message
        assert "parse_timeout" in err.message

    def test_to_dict_includes_all_fields(self):
        """to_dict should include all serialisable fields."""
        err = IngestionError(
            document_id="doc-123",
            file_path="/path/to/file.pdf",
            node="embedding_node",
            reason="api_timeout",
            details={"retry_count": 3},
            message="Embedding API timed out",
            traceback_str="Traceback...",
        )

        d = err.to_dict()

        assert d["document_id"] == "doc-123"
        assert d["file_path"] == "/path/to/file.pdf"
        assert d["node"] == "embedding_node"
        assert d["reason"] == "api_timeout"
        assert d["details"] == {"retry_count": 3}
        assert d["message"] == "Embedding API timed out"
        assert d["traceback_str"] == "Traceback..."
        assert d["error_type"] == "IngestionError"

    def test_to_json_serialises_cleanly(self):
        """to_json should produce valid JSON."""
        err = IngestionError(
            document_id="doc-123",
            file_path="/path/to/file.pdf",
            node="intake_node",
            reason="empty_file",
        )

        json_str = err.to_json()
        data = json.loads(json_str)

        assert data["document_id"] == "doc-123"
        assert data["error_type"] == "IngestionError"

    def test_from_dict_deserialises(self):
        """from_dict should reconstruct the error."""
        data = {
            "document_id": "doc-456",
            "file_path": "/path/to/doc.pdf",
            "node": "upsert_node",
            "reason": "connection_failed",
            "details": {"host": "localhost"},
            "message": "Connection failed",
            "traceback_str": "",
            "error_type": "IngestionError",
        }

        err = IngestionError.from_dict(data)

        assert err.document_id == "doc-456"
        assert err.file_path == "/path/to/doc.pdf"
        assert err.node == "upsert_node"
        assert err.reason == "connection_failed"


# ─── Subclass Tests ───────────────────────────────────────────────────────────


class TestErrorSubclasses:
    """Tests for each error subclass."""

    def test_intake_error_has_correct_node(self):
        """IntakeNodeError should have node='intake_node'."""
        err = IntakeNodeError(
            file_path="/path/to/file.pdf",
            reason="unsupported_format",
        )
        assert err.node == "intake_node"

    def test_parse_error_has_correct_node(self):
        """ParseNodeError should have node='parser_node'."""
        err = ParseNodeError(
            file_path="/path/to/file.pdf",
            reason="parse_timeout",
        )
        assert err.node == "parser_node"

    def test_metadata_error_has_correct_node(self):
        """MetadataError should have node='metadata_resolver_node'."""
        err = MetadataError(
            file_path="/path/to/file.pdf",
            reason="unclassified",
        )
        assert err.node == "metadata_resolver_node"

    def test_chunk_error_has_correct_node(self):
        """ChunkError should have node='chunker_node'."""
        err = ChunkError(
            file_path="/path/to/file.pdf",
            reason="no_headings",
        )
        assert err.node == "chunker_node"

    def test_embedding_error_has_correct_node(self):
        """EmbeddingNodeError should have node='embedding_node'."""
        err = EmbeddingNodeError(
            file_path="/path/to/file.pdf",
            reason="api_error",
            chunk_id="chunk-001",
        )
        assert err.node == "embedding_node"
        assert err.chunk_id == "chunk-001"

    def test_upsert_error_has_correct_node(self):
        """UpsertNodeError should have node='upsert_node'."""
        err = UpsertNodeError(
            file_path="/path/to/file.pdf",
            reason="connection_failed",
            batch_index=2,
            retry_count=3,
        )
        assert err.node == "upsert_node"
        assert err.batch_index == 2
        assert err.retry_count == 3

    def test_schema_error_has_correct_node(self):
        """SchemaNodeError should have node='schema_node'."""
        err = SchemaNodeError(
            expected={"vectors": 100},
            actual={"vectors": 200},
        )
        assert err.node == "schema_node"
        assert err.expected == {"vectors": 100}
        assert err.actual == {"vectors": 200}

    def test_health_check_error_has_correct_node(self):
        """HealthCheckNodeError should have node='health_check_node'."""
        err = HealthCheckNodeError(
            check_name="count_verified",
        )
        assert err.node == "health_check_node"
        assert err.check_name == "count_verified"


# ─── Serialisation Tests ──────────────────────────────────────────────────────


class TestErrorSerialisation:
    """Tests for error serialisation."""

    def test_each_error_serialises_cleanly(self):
        """Each error subclass should serialise to dict with correct fields."""
        errors = [
            IntakeNodeError(file_path="/a.pdf", reason="empty"),
            ParseNodeError(file_path="/b.pdf", reason="timeout"),
            MetadataError(file_path="/c.pdf", reason="unclassified"),
            ChunkError(file_path="/d.pdf", reason="no_headings"),
            EmbeddingNodeError(file_path="/e.pdf", reason="api_error", chunk_id="c1"),
            UpsertNodeError(file_path="/f.pdf", reason="connection_failed"),
            SchemaNodeError(expected={"v": 1}, actual={"v": 2}),
            HealthCheckNodeError(check_name="count_check"),
        ]

        for err in errors:
            d = err.to_dict()
            assert "error_type" in d
            assert "node" in d
            assert "reason" in d
            assert "file_path" in d
            assert d["error_type"] == err.__class__.__name__

    def test_each_error_json_roundtrip(self):
        """Each error should serialise to JSON and back."""
        err = UpsertNodeError(
            file_path="/f.pdf",
            reason="connection_failed",
            batch_index=2,
            retry_count=3,
        )

        json_str = err.to_json()
        data = json.loads(json_str)

        restored = IngestionError.from_dict(data)
        assert restored.node == "upsert_node"
        assert restored.reason == "connection_failed"


# ─── Helper Function Tests ────────────────────────────────────────────────────


class TestErrorHelpers:
    """Tests for error helper functions."""

    def test_error_from_exception_converts(self):
        """error_from_exception should convert any exception."""
        exc = ValueError("Something went wrong")
        err = error_from_exception(
            exc,
            node="intake_node",
            file_path="/path/to/file.pdf",
            document_id="doc-123",
        )

        assert err.document_id == "doc-123"
        assert err.file_path == "/path/to/file.pdf"
        assert err.node == "intake_node"
        assert err.reason == "ValueError"
        assert "Something went wrong" in err.message

    def test_error_from_exception_returns_unchanged_for_ingestion_error(self):
        """If exception is already an IngestionError, return unchanged."""
        original = IngestionError(
            file_path="/a.pdf",
            node="test",
            reason="test",
        )
        result = error_from_exception(original, node="other", file_path="/b.pdf")
        assert result is original

    def test_serialise_errors_mixed_list(self):
        """serialise_errors should handle mixed list of errors."""
        errors = [
            IngestionError(file_path="/a.pdf", node="test", reason="test"),
            ValueError("Generic error"),
        ]

        result = serialise_errors(errors)

        assert len(result) == 2
        assert result[0]["error_type"] == "IngestionError"
        assert result[1]["error_type"] == "ValueError"
