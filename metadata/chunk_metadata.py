"""ChunkMetadata Pydantic model with strict validation."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class ChunkMetadata(BaseModel):
    """
    Metadata for a single document chunk.

    Strictly validated at construction time.
    ValidationError raised immediately on bad data.

    Note: heading_path is a list representing the hierarchy from root to
    the chunk's immediate heading. section_title is auto-derived from
    heading_path[-1] at serialization time.
    """

    # Required fields
    document_id: str = Field(
        ..., min_length=1, description="Document identifier (SHA256 hash)"
    )
    chunk_index: int = Field(
        ..., ge=0, description="Chunk index within document (0-based, sequential)"
    )
    page_start: int = Field(..., ge=0, description="Starting page number")

    # Optional fields with defaults
    page_end: Optional[int] = Field(
        default=None, ge=0, description="Ending page number (None if single page)"
    )
    heading_path: List[str] = Field(
        default_factory=list,
        description="Full ancestor list from root to chunk's immediate heading",
    )
    char_start: int = Field(
        default=0, ge=0, description="Character offset start in document"
    )
    char_end: int = Field(
        default=0, ge=0, description="Character offset end in document"
    )
    token_count: int = Field(default=0, ge=0, description="Number of tokens in chunk")
    is_low_confidence: bool = Field(
        default=False,
        description="Flag for low OCR confidence pages",
    )
    unclassified: bool = Field(
        default=False,
        description="Flag for unclassified/malformed filename chunks",
    )
    table_index: Optional[int] = Field(
        default=None, description="Table index reference for table-containing chunks"
    )
    page_image_path: Optional[str] = Field(
        default=None, description="Path to page image (PDFs only)"
    )
    overlap_context: Optional[str] = Field(
        default=None,
        description="Overlap context from adjacent chunk (not embedded, not cited)",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Chunk metadata creation timestamp",
    )

    @property
    def section_title(self) -> str:
        """
        Derive section_title from heading_path[-1].

        Never set independently - always derived from heading_path.
        Returns empty string for fallback chunks with no heading_path.
        """
        return self.heading_path[-1] if self.heading_path else ""

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, v: str) -> str:
        """Validate document_id is not empty."""
        if not v or not v.strip():
            raise ValueError("document_id cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_page_range(self) -> "ChunkMetadata":
        """Validate that page_end >= page_start if both are set."""
        if self.page_end is not None and self.page_end < self.page_start:
            raise ValueError(
                f"page_end ({self.page_end}) cannot be less than page_start ({self.page_start})"
            )
        return self

    @model_validator(mode="after")
    def validate_char_range(self) -> "ChunkMetadata":
        """Validate that char_end >= char_start."""
        if self.char_end < self.char_start:
            raise ValueError(
                f"char_end ({self.char_end}) cannot be less than char_start ({self.char_start})"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        section_title is auto-derived from heading_path[-1].
        """
        data = self.model_dump(mode="json")
        # Auto-derive section_title from heading_path
        data["section_title"] = self.section_title
        return data
