"""Chunk validation with Pydantic and failure threshold handling."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import ValidationError

from .chunk_metadata import ChunkMetadata

logger = logging.getLogger(__name__)


@dataclass
class ChunkValidationError:
    """Represents a single chunk validation error."""

    document_id: str
    chunk_index: int
    field: str
    error: str


@dataclass
class ChunkValidationResult:
    """
    Result of validating chunks in a document.

    Tracks failed chunks and determines if document should be aborted.
    """

    document_id: str
    total_chunks: int
    failed_chunks: list[ChunkValidationError] = field(default_factory=list)
    valid_chunks: list[ChunkMetadata] = field(default_factory=list)
    failure_threshold: float = 0.20  # 20% default

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as fraction of total chunks."""
        if self.total_chunks == 0:
            return 0.0
        return len(self.failed_chunks) / self.total_chunks

    @property
    def should_abort(self) -> bool:
        """Check if failure rate exceeds threshold."""
        return self.failure_rate > self.failure_threshold

    @property
    def is_valid(self) -> bool:
        """Check if all chunks are valid."""
        return len(self.failed_chunks) == 0


class ChunkValidator:
    """
    Validates ChunkMetadata instances with Pydantic.

    Supports configurable failure threshold (default 20%).
    Documents exceeding threshold are marked as failed.
    """

    def __init__(self, failure_threshold: float = 0.20):
        """
        Initialize the chunk validator.

        Args:
            failure_threshold: Maximum acceptable failure rate (0.0-1.0)
        """
        self.failure_threshold = failure_threshold

    def validate_chunk(
        self, chunk_data: dict[str, Any], document_id: str, chunk_index: int
    ) -> tuple[Optional[ChunkMetadata], Optional[ChunkValidationError]]:
        """
        Validate a single chunk and return ChunkMetadata or error.

        Args:
            chunk_data: Dictionary with chunk data
            document_id: Document identifier
            chunk_index: Chunk index

        Returns:
            Tuple of (ChunkMetadata or None, ChunkValidationError or None)
        """
        try:
            # Pydantic validation - raises ValidationError immediately on bad data
            metadata = ChunkMetadata(**chunk_data)
            return metadata, None

        except ValidationError as e:
            # Extract field and error message
            error_str = str(e)

            # Try to identify which field failed
            field_name = "unknown"
            for error_detail in e.errors():
                if "loc" in error_detail and error_detail["loc"]:
                    field_name = str(error_detail["loc"][0])
                    break

            error = ChunkValidationError(
                document_id=document_id,
                chunk_index=chunk_index,
                field=field_name,
                error=error_str,
            )

            logger.warning(
                f"Chunk validation error for document {document_id}, "
                f"chunk {chunk_index}: {field_name} - {error_str}"
            )

            return None, error

    def validate_document_chunks(
        self,
        document_id: str,
        chunks: list[dict[str, Any]],
    ) -> ChunkValidationResult:
        """
        Validate all chunks in a document.

        Args:
            document_id: Document identifier
            chunks: List of chunk data dictionaries

        Returns:
            ChunkValidationResult with valid and failed chunks
        """
        result = ChunkValidationResult(
            document_id=document_id,
            total_chunks=len(chunks),
            failure_threshold=self.failure_threshold,
        )

        for chunk_index, chunk_data in enumerate(chunks):
            metadata, error = self.validate_chunk(
                chunk_data=chunk_data,
                document_id=document_id,
                chunk_index=chunk_index,
            )

            if metadata:
                result.valid_chunks.append(metadata)
            if error:
                result.failed_chunks.append(error)

        # Log summary
        if result.failed_chunks:
            logger.warning(
                f"Document {document_id}: {len(result.failed_chunks)}/{result.total_chunks} "
                f"chunks failed validation ({result.failure_rate:.1%} failure rate)"
            )

            if result.should_abort:
                logger.error(
                    f"Document {document_id}: Aborting - failure rate "
                    f"{result.failure_rate:.1%} exceeds threshold {self.failure_threshold:.1%}"
                )
            else:
                logger.warning(
                    f"Document {document_id}: Continuing with "
                    f"{len(result.valid_chunks)} valid chunks "
                    f"({len(result.failed_chunks)} failed chunks skipped)"
                )

        return result

    def log_validation_errors(
        self, result: ChunkValidationResult
    ) -> list[dict[str, Any]]:
        """
        Log validation errors in structured format.

        Args:
            result: ChunkValidationResult

        Returns:
            List of error dictionaries for logging
        """
        errors = []

        for error in result.failed_chunks:
            error_dict = {
                "document_id": error.document_id,
                "chunk_index": error.chunk_index,
                "field": error.field,
                "error": error.error,
            }
            errors.append(error_dict)
            logger.error(
                f"Validation error: document_id={error.document_id}, "
                f"chunk_index={error.chunk_index}, "
                f"field={error.field}, "
                f"error={error.error}"
            )

        return errors
