"""Chunking package for document chunk extraction."""

from typing import Optional

from .chunkers import (
    ChunkingConfig,
    HeadingHierarchyChunker,
    SlidingWindowChunker,
    TableAwareChunker,
    TextSegment,
)

__all__ = [
    "ChunkingConfig",
    "HeadingHierarchyChunker",
    "SlidingWindowChunker",
    "TableAwareChunker",
    "TextSegment",
]


def create_chunker(
    has_headings: bool = True,
    has_tables: bool = False,
    config: Optional[ChunkingConfig] = None,
):
    """
    Factory function to create appropriate chunker.

    Args:
        has_headings: Whether document has heading structure
        has_tables: Whether document contains tables
        config: Chunking configuration

    Returns:
        Appropriate chunker instance
    """
    if has_tables:
        return TableAwareChunker(config)
    elif has_headings:
        return HeadingHierarchyChunker(config)
    else:
        return SlidingWindowChunker(config)
