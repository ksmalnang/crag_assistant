"""Metadata resolution and validation package."""

from .chunk_metadata import ChunkMetadata
from .chunk_validation import (
    ChunkValidationError,
    ChunkValidationResult,
    ChunkValidator,
)
from .document_store import DocumentMetadataStore
from .resolver import (
    DocType,
    DocumentStatus,
    Faculty,
    FilenameMetadata,
)

__all__ = [
    # Resolver
    "Faculty",
    "DocType",
    "DocumentStatus",
    "FilenameMetadata",
    # Document Store
    "DocumentMetadataStore",
    # Chunk Metadata
    "ChunkMetadata",
    # Chunk Validation
    "ChunkValidationError",
    "ChunkValidationResult",
    "ChunkValidator",
]
