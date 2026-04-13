"""Unit tests for IP3 - Metadata Resolution Epic."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from metadata.chunk_metadata import ChunkMetadata
from metadata.chunk_validation import (
    ChunkValidationError,
    ChunkValidationResult,
    ChunkValidator,
)
from metadata.document_store import DocumentMetadataStore
from metadata.resolver import (
    SEMESTER_PATTERN,
    DocType,
    DocumentStatus,
    Faculty,
    FilenameMetadata,
)

# ============================================================
# IP-010: Metadata Resolver Tests
# ============================================================


class TestFacultyEnum:
    """Test Faculty enum."""

    def test_faculty_values(self):
        """Test all faculty values are defined."""
        assert Faculty.ENGINEERING == "engineering"
        assert Faculty.SCIENCE == "science"
        assert Faculty.ARTS == "arts"
        assert Faculty.BUSINESS == "business"
        assert Faculty.MEDICINE == "medicine"
        assert Faculty.LAW == "law"
        assert Faculty.EDUCATION == "education"
        assert Faculty.OTHER == "other"


class TestDocTypeEnum:
    """Test DocType enum."""

    def test_doctype_values(self):
        """Test all doc type values are defined."""
        assert DocType.LECTURE == "lecture"
        assert DocType.TUTORIAL == "tutorial"
        assert DocType.LAB == "lab"
        assert DocType.ASSIGNMENT == "assignment"
        assert DocType.EXAM == "exam"
        assert DocType.PROJECT == "project"
        assert DocType.THESIS == "thesis"
        assert DocType.SYLLABUS == "syllabus"
        assert DocType.READING == "reading"
        assert DocType.OTHER == "other"


class TestSemesterPattern:
    """Test semester regex pattern."""

    def test_valid_semesters(self):
        """Test valid semester formats."""
        assert SEMESTER_PATTERN.match("2024-S1")
        assert SEMESTER_PATTERN.match("2024-S2")
        assert SEMESTER_PATTERN.match("2023-S1")
        assert SEMESTER_PATTERN.match("2025-S2")

    def test_invalid_semesters(self):
        """Test invalid semester formats."""
        assert not SEMESTER_PATTERN.match("2024-S3")
        assert not SEMESTER_PATTERN.match("2024-S0")
        assert not SEMESTER_PATTERN.match("24-S1")
        assert not SEMESTER_PATTERN.match("2024S1")
        assert not SEMESTER_PATTERN.match("unknown")


class TestFilenameMetadata:
    """Test filename metadata parsing."""

    def test_valid_filename(self, tmp_path):
        """Test parsing valid filename convention."""
        file_path = tmp_path / "engineering_2024-S1_lecture_introduction.pdf"
        file_path.write_text("test content")
        file_bytes = b"test content"

        metadata = FilenameMetadata.from_filename(file_path, file_bytes)

        assert metadata.faculty == Faculty.ENGINEERING
        assert metadata.semester == "2024-S1"
        assert metadata.doc_type == DocType.LECTURE
        assert metadata.display_name == "introduction"
        assert metadata.source_name == "Introduction"
        assert metadata.unclassified is False
        assert metadata.document_id == hashlib.sha256(file_bytes).hexdigest()

    def test_filename_with_underscores_in_display_name(self, tmp_path):
        """Test display name with underscores."""
        file_path = tmp_path / "science_2024-S2_lab_final_exam.pdf"
        file_path.write_text("test")
        file_bytes = b"test"

        metadata = FilenameMetadata.from_filename(file_path, file_bytes)

        assert metadata.display_name == "final_exam"
        assert metadata.source_name == "Final Exam"

    def test_unrecognised_faculty(self, tmp_path):
        """Test unrecognised faculty defaults to OTHER with warning."""
        file_path = tmp_path / "unknown_faculty_2024-S1_lecture_test.pdf"
        file_path.write_text("test")
        file_bytes = b"test"

        metadata = FilenameMetadata.from_filename(file_path, file_bytes)

        assert metadata.faculty == Faculty.OTHER
        assert metadata.unclassified is True

    def test_unrecognised_doc_type(self, tmp_path):
        """Test unrecognised doc_type defaults to OTHER with warning."""
        file_path = tmp_path / "engineering_2024-S1_unknown_test.pdf"
        file_path.write_text("test")
        file_bytes = b"test"

        metadata = FilenameMetadata.from_filename(file_path, file_bytes)

        assert metadata.doc_type == DocType.OTHER
        assert metadata.unclassified is True

    def test_invalid_semester(self, tmp_path):
        """Test invalid semester format defaults to 'unknown'."""
        file_path = tmp_path / "engineering_invalid-S3_lecture_test.pdf"
        file_path.write_text("test")
        file_bytes = b"test"

        metadata = FilenameMetadata.from_filename(file_path, file_bytes)

        assert metadata.semester == "unknown"
        assert metadata.unclassified is True

    def test_malformed_filename_fallback(self, tmp_path):
        """Test malformed filename uses fallback values."""
        file_path = tmp_path / "bad_filename.pdf"
        file_path.write_text("test")
        file_bytes = b"test"

        metadata = FilenameMetadata.from_filename(file_path, file_bytes)

        assert metadata.faculty == Faculty.OTHER
        assert metadata.semester == "unknown"
        assert metadata.doc_type == DocType.OTHER
        assert metadata.unclassified is True

    def test_document_id_from_bytes(self, tmp_path):
        """Test document_id is SHA256 of file bytes."""
        file_path = tmp_path / "engineering_2024-S1_lecture_test.pdf"
        file_bytes = b"test content for hashing"
        file_path.write_bytes(file_bytes)

        metadata = FilenameMetadata.from_filename(file_path, file_bytes)

        expected_id = hashlib.sha256(file_bytes).hexdigest()
        assert metadata.document_id == expected_id

    def test_document_id_stability(self, tmp_path):
        """Test document_id is stable across renames."""
        file_path1 = tmp_path / "old_name.pdf"
        file_path2 = tmp_path / "new_name.pdf"
        file_bytes = b"same content"
        file_path1.write_bytes(file_bytes)
        file_path2.write_bytes(file_bytes)

        metadata1 = FilenameMetadata.from_filename(file_path1, file_bytes)
        metadata2 = FilenameMetadata.from_filename(file_path2, file_bytes)

        # Same content = same document_id
        assert metadata1.document_id == metadata2.document_id

    def test_partial_filename_missing_segments(self, tmp_path):
        """Test filename with missing segments."""
        # Only 2 segments instead of 4
        file_path = tmp_path / "engineering_2024-S1.pdf"
        file_path.write_text("test")
        file_bytes = b"test"

        metadata = FilenameMetadata.from_filename(file_path, file_bytes)

        assert metadata.faculty == Faculty.ENGINEERING
        assert metadata.semester == "2024-S1"
        assert metadata.doc_type == DocType.OTHER  # Missing
        assert metadata.unclassified is True

    def test_all_valid_faculties(self, tmp_path):
        """Test parsing all valid faculty values."""
        for faculty in Faculty:
            if faculty == Faculty.OTHER:
                continue

            file_path = tmp_path / f"{faculty.value}_2024-S1_lecture_test.pdf"
            file_path.write_text("test")
            file_bytes = b"test"

            metadata = FilenameMetadata.from_filename(file_path, file_bytes)
            assert metadata.faculty == faculty

    def test_all_valid_doctypes(self, tmp_path):
        """Test parsing all valid doc_type values."""
        for doc_type in DocType:
            if doc_type == DocType.OTHER:
                continue

            file_path = tmp_path / f"engineering_2024-S1_{doc_type.value}_test.pdf"
            file_path.write_text("test")
            file_bytes = b"test"

            metadata = FilenameMetadata.from_filename(file_path, file_bytes)
            assert metadata.doc_type == doc_type

    def test_display_name_title_casing(self, tmp_path):
        """Test display_name is converted to title case."""
        file_path = tmp_path / "engineering_2024-S1_lecture_machine_learning_basics.pdf"
        file_path.write_text("test")
        file_bytes = b"test"

        metadata = FilenameMetadata.from_filename(file_path, file_bytes)

        assert metadata.source_name == "Machine Learning Basics"

    def test_filename_without_file_bytes_warning(self, tmp_path, caplog):
        """Test warning when file_bytes not provided."""
        file_path = tmp_path / "engineering_2024-S1_lecture_test.pdf"
        file_path.write_text("test")

        metadata = FilenameMetadata.from_filename(file_path)

        assert "file_bytes not provided" in caplog.text
        # Should still work with path-based hash
        assert metadata.document_id is not None


# ============================================================
# IP-011: Document Metadata Store Tests
# ============================================================


class TestDocumentMetadataStore:
    """Test document metadata store."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create metadata store."""
        db_path = tmp_path / "test_metadata.db"
        return DocumentMetadataStore(db_path=db_path)

    def test_upsert_new_document(self, store):
        """Test inserting new document."""
        store.upsert_document(
            document_id="doc-123",
            source_path="/path/to/file.pdf",
            source_name="Test Document",
            faculty=Faculty.ENGINEERING,
            doc_type=DocType.LECTURE,
            semester="2024-S1",
            chunk_count=10,
            status=DocumentStatus.PENDING,
        )

        doc = store.get_document("doc-123")
        assert doc is not None
        assert doc["document_id"] == "doc-123"
        assert doc["faculty"] == "engineering"
        assert doc["status"] == "pending"
        assert doc["chunk_count"] == 10

    def test_upsert_update_existing(self, store):
        """Test updating existing document."""
        store.upsert_document(
            document_id="doc-123",
            source_path="/path/to/file.pdf",
            source_name="Test",
            faculty=Faculty.ENGINEERING,
            doc_type=DocType.LECTURE,
            semester="2024-S1",
            status=DocumentStatus.PENDING,
        )

        # Update status
        store.upsert_document(
            document_id="doc-123",
            source_path="/path/to/file.pdf",
            source_name="Test Updated",
            faculty=Faculty.ENGINEERING,
            doc_type=DocType.LECTURE,
            semester="2024-S1",
            status=DocumentStatus.INGESTED,
        )

        doc = store.get_document("doc-123")
        assert doc["source_name"] == "Test Updated"
        assert doc["status"] == "ingested"

    def test_mark_ingested(self, store):
        """Test marking document as ingested."""
        store.upsert_document(
            document_id="doc-123",
            source_path="/path/to/file.pdf",
            source_name="Test",
            faculty=Faculty.ENGINEERING,
            doc_type=DocType.LECTURE,
            semester="2024-S1",
            status=DocumentStatus.PENDING,
        )

        store.mark_ingested("doc-123", chunk_count=15)

        doc = store.get_document("doc-123")
        assert doc["status"] == "ingested"
        assert doc["chunk_count"] == 15
        assert doc["ingested_at"] is not None

    def test_mark_failed(self, store):
        """Test marking document as failed."""
        store.upsert_document(
            document_id="doc-123",
            source_path="/path/to/file.pdf",
            source_name="Test",
            faculty=Faculty.ENGINEERING,
            doc_type=DocType.LECTURE,
            semester="2024-S1",
            status=DocumentStatus.PENDING,
        )

        store.mark_failed("doc-123")

        doc = store.get_document("doc-123")
        assert doc["status"] == "failed"

    def test_query_by_faculty(self, store):
        """Test querying documents by faculty."""
        store.upsert_document(
            document_id="doc-1",
            source_path="/path/eng.pdf",
            source_name="Engineering",
            faculty=Faculty.ENGINEERING,
            doc_type=DocType.LECTURE,
            semester="2024-S1",
        )
        store.upsert_document(
            document_id="doc-2",
            source_path="/path/sci.pdf",
            source_name="Science",
            faculty=Faculty.SCIENCE,
            doc_type=DocType.LAB,
            semester="2024-S1",
        )

        results = store.query_documents(faculty=Faculty.ENGINEERING)
        assert len(results) == 1
        assert results[0]["faculty"] == "engineering"

    def test_query_by_semester(self, store):
        """Test querying documents by semester."""
        store.upsert_document(
            document_id="doc-1",
            source_path="/path/1.pdf",
            source_name="Doc 1",
            faculty=Faculty.ENGINEERING,
            doc_type=DocType.LECTURE,
            semester="2024-S1",
        )
        store.upsert_document(
            document_id="doc-2",
            source_path="/path/2.pdf",
            source_name="Doc 2",
            faculty=Faculty.ENGINEERING,
            doc_type=DocType.LECTURE,
            semester="2024-S2",
        )

        results = store.query_documents(semester="2024-S1")
        assert len(results) == 1

    def test_query_by_status(self, store):
        """Test querying documents by status."""
        store.upsert_document(
            document_id="doc-1",
            source_path="/path/1.pdf",
            source_name="Doc 1",
            faculty=Faculty.ENGINEERING,
            doc_type=DocType.LECTURE,
            semester="2024-S1",
            status=DocumentStatus.INGESTED,
        )
        store.upsert_document(
            document_id="doc-2",
            source_path="/path/2.pdf",
            source_name="Doc 2",
            faculty=Faculty.ENGINEERING,
            doc_type=DocType.LECTURE,
            semester="2024-S1",
            status=DocumentStatus.FAILED,
        )

        results = store.query_documents(status=DocumentStatus.FAILED)
        assert len(results) == 1

    def test_status_transitions_valid(self, store):
        """Test valid status transitions."""
        # pending -> ingested
        store.upsert_document(
            document_id="doc-1",
            source_path="/path/1.pdf",
            source_name="Doc 1",
            faculty=Faculty.ENGINEERING,
            doc_type=DocType.LECTURE,
            semester="2024-S1",
            status=DocumentStatus.PENDING,
        )
        store.update_status("doc-1", DocumentStatus.INGESTED)

        doc = store.get_document("doc-1")
        assert doc["status"] == "ingested"

    def test_status_transitions_invalid(self, store):
        """Test invalid status transitions raise error."""
        store.upsert_document(
            document_id="doc-1",
            source_path="/path/1.pdf",
            source_name="Doc 1",
            faculty=Faculty.ENGINEERING,
            doc_type=DocType.LECTURE,
            semester="2024-S1",
            status=DocumentStatus.PENDING,
        )

        # Cannot go from pending -> stale directly
        with pytest.raises(ValueError):
            store.update_status("doc-1", DocumentStatus.STALE)

    def test_reingest_marks_old_as_stale(self, store, caplog):
        """Test re-ingestion marks old document as stale before updating."""
        import logging

        caplog.set_level(logging.INFO)

        store.upsert_document(
            document_id="doc-123",
            source_path="/path/file.pdf",
            source_name="Test",
            faculty=Faculty.ENGINEERING,
            doc_type=DocType.LECTURE,
            semester="2024-S1",
            chunk_count=10,
            status=DocumentStatus.INGESTED,
        )

        # Re-ingest: should mark as stale first, then update to pending
        store.upsert_document(
            document_id="doc-123",
            source_path="/path/file.pdf",
            source_name="Test",
            faculty=Faculty.ENGINEERING,
            doc_type=DocType.LECTURE,
            semester="2024-S1",
            chunk_count=12,
            status=DocumentStatus.PENDING,
        )

        # Check that stale marking was logged
        assert "marked as stale for re-ingestion" in caplog.text

        # Final status should be pending (the new status we're setting)
        doc = store.get_document("doc-123")
        assert doc["status"] == "pending"
        assert doc["chunk_count"] == 12

    def test_get_stale_documents(self, store):
        """Test retrieving stale documents."""
        store.upsert_document(
            document_id="doc-1",
            source_path="/path/1.pdf",
            source_name="Doc 1",
            faculty=Faculty.ENGINEERING,
            doc_type=DocType.LECTURE,
            semester="2024-S1",
            status=DocumentStatus.STALE,
        )
        store.upsert_document(
            document_id="doc-2",
            source_path="/path/2.pdf",
            source_name="Doc 2",
            faculty=Faculty.ENGINEERING,
            doc_type=DocType.LECTURE,
            semester="2024-S1",
            status=DocumentStatus.INGESTED,
        )

        stale_docs = store.get_stale_documents()
        assert len(stale_docs) == 1
        assert stale_docs[0]["status"] == "stale"

    def test_delete_document(self, store):
        """Test deleting document."""
        store.upsert_document(
            document_id="doc-123",
            source_path="/path/file.pdf",
            source_name="Test",
            faculty=Faculty.ENGINEERING,
            doc_type=DocType.LECTURE,
            semester="2024-S1",
        )

        result = store.delete_document("doc-123")
        assert result is True

        doc = store.get_document("doc-123")
        assert doc is None

    def test_delete_nonexistent_document(self, store):
        """Test deleting nonexistent document."""
        result = store.delete_document("nonexistent")
        assert result is False


# ============================================================
# IP-012: ChunkMetadata Validation Tests
# ============================================================


class TestChunkMetadata:
    """Test ChunkMetadata Pydantic model."""

    def test_valid_chunk_metadata(self):
        """Test creating valid chunk metadata."""
        metadata = ChunkMetadata(
            document_id="doc-123",
            chunk_index=0,
            page_start=1,
            page_end=2,
            section_title="Introduction",
            heading_path="Introduction",
            char_start=0,
            char_end=100,
            token_count=50,
        )

        assert metadata.document_id == "doc-123"
        assert metadata.chunk_index == 0
        assert metadata.page_start == 1
        assert metadata.page_end == 2
        assert metadata.section_title == "Introduction"

    def test_missing_required_field_raises(self):
        """Test missing required field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ChunkMetadata(
                chunk_index=0,  # Missing document_id
                page_start=1,
            )

        assert "document_id" in str(exc_info.value)

    def test_negative_chunk_index_raises(self):
        """Test negative chunk_index raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ChunkMetadata(
                document_id="doc-123",
                chunk_index=-1,
                page_start=1,
            )

        assert "chunk_index" in str(exc_info.value)

    def test_page_end_less_than_page_start_raises(self):
        """Test page_end < page_start raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ChunkMetadata(
                document_id="doc-123",
                chunk_index=0,
                page_start=5,
                page_end=3,
            )

        assert "page_end" in str(exc_info.value).lower()

    def test_empty_document_id_raises(self):
        """Test empty document_id raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ChunkMetadata(
                document_id="",
                chunk_index=0,
                page_start=1,
            )

        assert "document_id" in str(exc_info.value)

    def test_heading_path_required_when_section_title_set(self):
        """Test heading_path must be set when section_title is set."""
        with pytest.raises(ValidationError) as exc_info:
            ChunkMetadata(
                document_id="doc-123",
                chunk_index=0,
                page_start=1,
                section_title="Introduction",
                heading_path=None,
            )

        assert "heading_path" in str(exc_info.value).lower()

    def test_valid_semester_format(self):
        """Test valid semester format in custom validation."""
        # This would be validated by custom validator if we add semester field
        metadata = ChunkMetadata(
            document_id="doc-123",
            chunk_index=0,
            page_start=1,
        )

        assert metadata is not None

    def test_char_end_less_than_char_start_raises(self):
        """Test char_end < char_start raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ChunkMetadata(
                document_id="doc-123",
                chunk_index=0,
                page_start=1,
                char_start=100,
                char_end=50,
            )

        assert "char_end" in str(exc_info.value).lower()

    def test_to_dict_serialization(self):
        """Test to_dict serialization."""
        metadata = ChunkMetadata(
            document_id="doc-123",
            chunk_index=0,
            page_start=1,
            section_title="Intro",
            heading_path="Intro",
        )

        data = metadata.to_dict()
        assert isinstance(data, dict)
        assert data["document_id"] == "doc-123"
        assert data["chunk_index"] == 0


class TestChunkValidation:
    """Test chunk validation with failure threshold."""

    def test_validate_valid_chunk(self):
        """Test validating a valid chunk."""
        validator = ChunkValidator()

        chunk_data = {
            "document_id": "doc-123",
            "chunk_index": 0,
            "page_start": 1,
        }

        metadata, error = validator.validate_chunk(chunk_data, "doc-123", 0)

        assert metadata is not None
        assert error is None

    def test_validate_invalid_chunk(self):
        """Test validating an invalid chunk."""
        validator = ChunkValidator()

        chunk_data = {
            "chunk_index": 0,  # Missing document_id
            "page_start": 1,
        }

        metadata, error = validator.validate_chunk(chunk_data, "doc-123", 0)

        assert metadata is None
        assert error is not None
        assert error.document_id == "doc-123"
        assert error.chunk_index == 0

    def test_validate_document_chunks_all_valid(self):
        """Test validating all valid chunks."""
        validator = ChunkValidator()

        chunks = [
            {"document_id": "doc-123", "chunk_index": 0, "page_start": 1},
            {"document_id": "doc-123", "chunk_index": 1, "page_start": 2},
            {"document_id": "doc-123", "chunk_index": 2, "page_start": 3},
        ]

        result = validator.validate_document_chunks("doc-123", chunks)

        assert result.is_valid is True
        assert len(result.valid_chunks) == 3
        assert len(result.failed_chunks) == 0
        assert result.should_abort is False

    def test_validate_document_chunks_some_failures_below_threshold(self):
        """Test validating chunks with some failures below threshold."""
        validator = ChunkValidator(failure_threshold=0.20)

        chunks = [
            {"document_id": "doc-123", "chunk_index": 0, "page_start": 1},
            {"document_id": "doc-123", "chunk_index": 1, "page_start": 2},
            {"document_id": "doc-123", "chunk_index": 2, "page_start": 3},
            {"document_id": "doc-123", "chunk_index": 3, "page_start": 4},
            {"chunk_index": 4, "page_start": 5},  # Invalid - missing document_id
        ]

        result = validator.validate_document_chunks("doc-123", chunks)

        assert result.is_valid is False
        assert len(result.valid_chunks) == 4
        assert len(result.failed_chunks) == 1
        assert result.failure_rate == 0.2  # 1/5 = 20%
        assert result.should_abort is False  # Exactly at threshold, not exceeding

    def test_validate_document_chunks_abort_exceeds_threshold(self):
        """Test aborting when failures exceed threshold."""
        validator = ChunkValidator(failure_threshold=0.20)

        chunks = [
            {"document_id": "doc-123", "chunk_index": 0, "page_start": 1},
            {"chunk_index": 1, "page_start": 2},  # Invalid
            {"chunk_index": 2, "page_start": 3},  # Invalid
            {"chunk_index": 3, "page_start": 4},  # Invalid
        ]

        result = validator.validate_document_chunks("doc-123", chunks)

        assert result.failure_rate == 0.75  # 3/4 = 75%
        assert result.should_abort is True

    def test_log_validation_errors(self):
        """Test logging validation errors."""
        validator = ChunkValidator()

        error = ChunkValidationError(
            document_id="doc-123",
            chunk_index=0,
            field="document_id",
            error="Field required",
        )

        result = ChunkValidationResult(
            document_id="doc-123",
            total_chunks=1,
            failed_chunks=[error],
        )

        errors = validator.log_validation_errors(result)

        assert len(errors) == 1
        assert errors[0]["document_id"] == "doc-123"
        assert errors[0]["field"] == "document_id"

    def test_validation_result_failure_rate(self):
        """Test failure rate calculation."""
        result = ChunkValidationResult(
            document_id="doc-123",
            total_chunks=10,
            failed_chunks=[MagicMock() for _ in range(3)],
        )

        assert result.failure_rate == 0.3

    def test_validation_result_zero_chunks(self):
        """Test failure rate with zero chunks."""
        result = ChunkValidationResult(
            document_id="doc-123",
            total_chunks=0,
        )

        assert result.failure_rate == 0.0
        assert result.should_abort is False

    def test_custom_failure_threshold(self):
        """Test custom failure threshold."""
        validator = ChunkValidator(failure_threshold=0.50)  # 50% threshold

        chunks = [
            {"document_id": "doc-123", "chunk_index": 0, "page_start": 1},
            {"chunk_index": 1, "page_start": 2},  # Invalid
            {"chunk_index": 2, "page_start": 3},  # Invalid
        ]

        result = validator.validate_document_chunks("doc-123", chunks)

        assert result.failure_rate == 2 / 3  # ~66%
        assert result.should_abort is True  # Exceeds 50% threshold


class TestMetadataIntegration:
    """Integration tests for metadata resolution."""

    def test_full_metadata_resolution_workflow(self, tmp_path):
        """Test complete metadata resolution workflow."""
        # Create file
        file_path = tmp_path / "engineering_2024-S1_lecture_introduction.pdf"
        file_bytes = b"PDF content here"
        file_path.write_bytes(file_bytes)

        # Parse filename
        metadata = FilenameMetadata.from_filename(file_path, file_bytes)

        assert metadata.faculty == Faculty.ENGINEERING
        assert metadata.semester == "2024-S1"
        assert metadata.doc_type == DocType.LECTURE
        assert metadata.source_name == "Introduction"
        assert metadata.unclassified is False

    def test_metadata_store_integration(self, tmp_path):
        """Test metadata store integration."""
        file_path = tmp_path / "science_2024-S2_lab_experiment.pdf"
        file_bytes = b"Lab content"
        file_path.write_bytes(file_bytes)

        # Parse metadata
        metadata = FilenameMetadata.from_filename(file_path, file_bytes)

        # Store in database
        db_path = tmp_path / "test.db"
        store = DocumentMetadataStore(db_path=db_path)

        store.upsert_document(
            document_id=metadata.document_id,
            source_path=str(file_path),
            source_name=metadata.source_name,
            faculty=metadata.faculty,
            doc_type=metadata.doc_type,
            semester=metadata.semester,
            status=DocumentStatus.PENDING,
        )

        # Retrieve and verify
        doc = store.get_document(metadata.document_id)
        assert doc is not None
        assert doc["faculty"] == "science"
        assert doc["semester"] == "2024-S2"

    def test_chunk_validation_integration(self):
        """Test chunk validation integration."""
        validator = ChunkValidator(failure_threshold=0.20)

        # Mix of valid and invalid chunks
        chunks = [
            {"document_id": "doc-123", "chunk_index": 0, "page_start": 1},
            {"document_id": "doc-123", "chunk_index": 1, "page_start": 2},
            {"document_id": "doc-123", "chunk_index": 2, "page_start": 3},
            {
                "document_id": "doc-123",
                "chunk_index": 3,
                "page_start": 5,
                "page_end": 3,
            },  # Invalid: page_end < page_start
        ]

        result = validator.validate_document_chunks("doc-123", chunks)

        assert len(result.valid_chunks) == 3
        assert len(result.failed_chunks) == 1
        assert result.failure_rate == 0.25
        assert result.should_abort is True  # 25% > 20% threshold
