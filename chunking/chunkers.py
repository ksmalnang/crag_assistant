"""Document chunking strategies with heading-hierarchy, sliding window, table-aware, and overlap injection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from metadata.chunk_metadata import ChunkMetadata
from parsing.docling_parser import ParsedDocument, StructureNode, TableData

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies."""

    # Token limits
    max_chunk_tokens: int = 1024
    # Sliding window fallback
    window_size: int = 512
    window_stride: int = 448  # 64-token overlap
    # Overlap context injection
    overlap_tokens: int = 64  # 0 disables overlap
    # Token estimation (chars per token approximation)
    chars_per_token: float = 4.0

    @property
    def overlap_disabled(self) -> bool:
        """Check if overlap injection is disabled."""
        return self.overlap_tokens <= 0


@dataclass
class TextSegment:
    """Represents a text segment with metadata."""

    text: str
    char_start: int
    char_end: int
    page_start: int
    page_end: Optional[int] = None
    heading_path: List[str] = field(default_factory=list)
    table_index: Optional[int] = None
    is_table: bool = False


class HeadingHierarchyChunker:
    """
    Primary chunking strategy: splits documents at heading boundaries.

    Uses structure tree from IP-006 to create chunks that respect
    the document's hierarchical heading structure.

    Features:
    - Split at H1/H2/H3 boundaries
    - heading_path = full ancestor list from root to chunk's heading
    - section_title derived from heading_path[-1]
    - Oversized sections split mid-section with sliding window
    - Parent heading_path retained on sub-chunks
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize the chunker.

        Args:
            config: Chunking configuration (uses defaults if not provided)
        """
        self.config = config or ChunkingConfig()

    def chunk(self, parsed_doc: ParsedDocument) -> List[ChunkMetadata]:
        """
        Chunk a parsed document using heading hierarchy.

        Args:
            parsed_doc: ParsedDocument from DoclingParser

        Returns:
            List of ChunkMetadata objects
        """
        if not parsed_doc.has_headings:
            logger.warning(
                f"Document {parsed_doc.document_id} has no headings. "
                f"Use SlidingWindowChunker instead."
            )
            return []

        # Extract text segments from structure tree
        segments = self._extract_text_segments(parsed_doc)

        # Create chunks from segments
        chunks = self._create_chunks_from_segments(segments, parsed_doc)

        # Apply overlap context injection
        if not self.config.overlap_disabled:
            chunks = self._inject_overlap_context(chunks, parsed_doc)

        return chunks

    def _extract_text_segments(self, parsed_doc: ParsedDocument) -> List[TextSegment]:
        """
        Extract text segments from parsed document based on structure tree.

        Args:
            parsed_doc: ParsedDocument with structure_tree and text_content

        Returns:
            List of TextSegment objects
        """
        segments = []
        text_content = parsed_doc.text_content

        # Traverse structure tree to extract segments
        for node in parsed_doc.structure_tree:
            node_segments = self._traverse_structure_node(
                node, text_content, parsed_doc
            )
            segments.extend(node_segments)

        return segments

    def _traverse_structure_node(
        self,
        node: StructureNode,
        text_content: str,
        parsed_doc: ParsedDocument,
        parent_path: Optional[List[str]] = None,
    ) -> List[TextSegment]:
        """
        Recursively traverse structure node to extract text segments.

        Args:
            node: StructureNode from document
            text_content: Full document text
            parsed_doc: ParsedDocument
            parent_path: Parent heading path

        Returns:
            List of TextSegment objects
        """
        segments = []
        parent_path = parent_path or []

        # Build heading path for this node
        current_path = parent_path + [node.title]

        # Estimate token count for this section
        # Find text between this heading and next heading (or end of doc)
        section_text = self._extract_section_text(node, text_content, parsed_doc)

        if not section_text or not section_text.strip():
            # Skip headings with no body text
            logger.debug(f"Skipping heading '{node.title}' with no body text")
            # Still process children
            for child in node.children:
                child_segments = self._traverse_structure_node(
                    child, text_content, parsed_doc, current_path
                )
                segments.extend(child_segments)
            return segments

        # Create segment for this section
        token_count = self._estimate_tokens(section_text)

        segment = TextSegment(
            text=section_text,
            char_start=node.char_start,
            char_end=node.char_start + len(section_text),
            page_start=node.page_start,
            page_end=node.page_start,  # Will be updated if multi-page
            heading_path=current_path,
            is_table=False,
        )

        # Check if section exceeds token limit
        if token_count > self.config.max_chunk_tokens:
            # Split section using sliding window
            sub_segments = self._split_oversized_section(segment, current_path)
            segments.extend(sub_segments)
        else:
            segments.append(segment)

        # Process children
        for child in node.children:
            child_segments = self._traverse_structure_node(
                child, text_content, parsed_doc, current_path
            )
            segments.extend(child_segments)

        return segments

    def _extract_section_text(
        self,
        node: StructureNode,
        text_content: str,
        parsed_doc: ParsedDocument,
    ) -> str:
        """
        Extract text content for a section node.

        Args:
            node: StructureNode
            text_content: Full document text
            parsed_doc: ParsedDocument

        Returns:
            Section text content
        """
        # Simple extraction: use character offsets if available
        if node.char_start >= 0 and node.char_start < len(text_content):
            # Find end position (next heading or end of text)
            char_end = min(
                node.char_start + 5000,  # Reasonable section size
                len(text_content),
            )
            return text_content[node.char_start : char_end]

        # Fallback: return empty string
        return ""

    def _split_oversized_section(
        self, segment: TextSegment, heading_path: List[str]
    ) -> List[TextSegment]:
        """
        Split oversized section using sliding window approach.

        Args:
            segment: TextSegment to split
            heading_path: Parent heading path to retain

        Returns:
            List of sub-segments
        """
        sub_segments = []
        text = segment.text
        chars_per_chunk = int(
            self.config.max_chunk_tokens * self.config.chars_per_token
        )

        start = 0
        while start < len(text):
            end = min(start + chars_per_chunk, len(text))

            # Try to split at sentence boundary
            split_pos = self._find_sentence_boundary(text, start, end)
            if split_pos:
                end = split_pos

            chunk_text = text[start:end]

            sub_segment = TextSegment(
                text=chunk_text,
                char_start=segment.char_start + start,
                char_end=segment.char_start + end,
                page_start=segment.page_start,
                page_end=segment.page_end,
                heading_path=heading_path,  # Retain parent path
                is_table=False,
            )
            sub_segments.append(sub_segment)

            start = end

        return sub_segments

    def _find_sentence_boundary(
        self, text: str, start: int, target_end: int
    ) -> Optional[int]:
        """
        Find sentence boundary near target position.

        Args:
            text: Full text
            start: Start position
            target_end: Target end position

        Returns:
            Position to split at, or None
        """
        # Look backwards from target_end for sentence boundary
        search_start = max(start, target_end - 200)

        for i in range(target_end, search_start, -1):
            if i < len(text) and text[i] in [".", "!", "?", "\n"]:
                return i + 1

        return None

    def _create_chunks_from_segments(
        self, segments: List[TextSegment], parsed_doc: ParsedDocument
    ) -> List[ChunkMetadata]:
        """
        Create ChunkMetadata from text segments.

        Args:
            segments: List of TextSegment
            parsed_doc: ParsedDocument for metadata

        Returns:
            List of ChunkMetadata
        """
        chunks = []

        for idx, segment in enumerate(segments):
            token_count = self._estimate_tokens(segment.text)

            # Skip 0-token chunks (headings with no body)
            if token_count == 0 and not segment.text.strip():
                continue

            chunk = ChunkMetadata(
                document_id=parsed_doc.document_id,
                chunk_index=idx,
                page_start=segment.page_start,
                page_end=segment.page_end,
                heading_path=segment.heading_path,
                char_start=segment.char_start,
                char_end=segment.char_end,
                token_count=token_count,
                table_index=segment.table_index,
                is_low_confidence=parsed_doc.is_scanned_pdf,
                unclassified=False,
            )
            chunks.append(chunk)

        return chunks

    def _inject_overlap_context(
        self, chunks: List[ChunkMetadata], parsed_doc: ParsedDocument
    ) -> List[ChunkMetadata]:
        """
        Inject overlap context from adjacent chunks.

        Overlap rules:
        - Only from same heading level or parent
        - Never crosses H1 boundaries
        - Token budget counted against max_chunk_tokens

        Args:
            chunks: List of chunks
            parsed_doc: ParsedDocument

        Returns:
            Updated chunks with overlap context
        """
        if not chunks:
            return chunks

        overlap_chars = int(self.config.overlap_tokens * self.config.chars_per_token)

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]

            # Check if we can inject overlap (same H1 parent)
            if self._can_inject_overlap(prev_chunk, curr_chunk):
                # Get overlap text from end of previous chunk
                overlap_start = max(0, prev_chunk.char_end - overlap_chars)
                overlap_text = parsed_doc.text_content[
                    overlap_start : prev_chunk.char_end
                ]

                # Store as overlap context (not embedded, not cited)
                curr_chunk.overlap_context = overlap_text

        return chunks

    def _can_inject_overlap(
        self, prev_chunk: ChunkMetadata, curr_chunk: ChunkMetadata
    ) -> bool:
        """
        Check if overlap can be injected between two chunks.

        Never crosses H1 boundaries.

        Args:
            prev_chunk: Previous chunk
            curr_chunk: Current chunk

        Returns:
            True if overlap can be injected
        """
        # Cannot inject if either chunk has no heading_path
        if not prev_chunk.heading_path or not curr_chunk.heading_path:
            return False

        # Check H1 boundary (first element of heading_path)
        prev_h1 = prev_chunk.heading_path[0]
        curr_h1 = curr_chunk.heading_path[0]

        # Never cross H1 boundaries
        return prev_h1 == curr_h1

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text content

        Returns:
            Estimated token count
        """
        if not text:
            return 0
        return int(len(text) / self.config.chars_per_token)


class SlidingWindowChunker:
    """
    Fallback chunking strategy: sliding window with overlap.

    Triggered when has_headings=False on parsed document.
    Uses fixed window size with configurable stride for overlap.
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize the chunker.

        Args:
            config: Chunking configuration (uses defaults if not provided)
        """
        self.config = config or ChunkingConfig()

    def chunk(self, parsed_doc: ParsedDocument) -> List[ChunkMetadata]:
        """
        Chunk a flat document using sliding window.

        Args:
            parsed_doc: ParsedDocument without headings

        Returns:
            List of ChunkMetadata objects
        """
        if parsed_doc.has_headings:
            logger.warning(
                f"Document {parsed_doc.document_id} has headings. "
                f"Use HeadingHierarchyChunker instead."
            )

        text = parsed_doc.text_content
        if not text or not text.strip():
            return []

        chunks = []
        chunk_index = 0

        # Calculate window and stride in characters
        window_chars = int(self.config.window_size * self.config.chars_per_token)
        stride_chars = int(self.config.window_stride * self.config.chars_per_token)

        start = 0
        while start < len(text):
            end = min(start + window_chars, len(text))

            # Try to split at sentence boundary
            split_pos = self._find_sentence_boundary(text, start, end)
            if split_pos:
                end = split_pos

            chunk_text = text[start:end]
            token_count = self._estimate_tokens(chunk_text)

            # Create chunk with empty heading_path and section_title
            chunk = ChunkMetadata(
                document_id=parsed_doc.document_id,
                chunk_index=chunk_index,
                page_start=self._char_to_page(start, parsed_doc),
                page_end=self._char_to_page(end, parsed_doc),
                heading_path=[],  # Fallback chunks have no heading path
                char_start=start,
                char_end=end,
                token_count=token_count,
                is_low_confidence=parsed_doc.is_scanned_pdf,
            )
            chunks.append(chunk)

            chunk_index += 1
            start = end

        logger.info(
            f"Created {len(chunks)} sliding window chunks for document "
            f"{parsed_doc.document_id}"
        )

        return chunks

    def _find_sentence_boundary(
        self, text: str, start: int, target_end: int
    ) -> Optional[int]:
        """Find sentence boundary near target position."""
        search_start = max(start, target_end - 200)

        for i in range(target_end, search_start, -1):
            if i < len(text) and text[i] in [".", "!", "?", "\n"]:
                return i + 1

        return None

    def _char_to_page(self, char_pos: int, parsed_doc: ParsedDocument) -> int:
        """
        Convert character offset to page number.

        Args:
            char_pos: Character position
            parsed_doc: ParsedDocument

        Returns:
            Page number (0 if unknown)
        """
        # Simplified: assume uniform distribution across pages
        if parsed_doc.page_count == 0:
            return 0

        chars_per_page = len(parsed_doc.text_content) / parsed_doc.page_count
        if chars_per_page == 0:
            return 0

        return int(char_pos / chars_per_page)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        if not text:
            return 0
        return int(len(text) / self.config.chars_per_token)


class TableAwareChunker:
    """
    Table-aware chunking strategy.

    Ensures tables are chunked intelligently:
    - Small tables kept intact within token limit
    - Large tables split at row boundaries only
    - table_index reference in metadata
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize the chunker.

        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig()
        self.heading_chunker = HeadingHierarchyChunker(config)

    def chunk(self, parsed_doc: ParsedDocument) -> List[ChunkMetadata]:
        """
        Chunk document with table awareness.

        Args:
            parsed_doc: ParsedDocument with tables

        Returns:
            List of ChunkMetadata
        """
        if not parsed_doc.tables:
            # No tables, use standard chunker
            if parsed_doc.has_headings:
                return self.heading_chunker.chunk(parsed_doc)
            else:
                return SlidingWindowChunker(self.config).chunk(parsed_doc)

        # Chunk with table awareness
        chunks = self._chunk_with_tables(parsed_doc)
        return chunks

    def _chunk_with_tables(self, parsed_doc: ParsedDocument) -> List[ChunkMetadata]:
        """
        Create chunks respecting table boundaries.

        Args:
            parsed_doc: ParsedDocument with tables

        Returns:
            List of ChunkMetadata
        """
        chunks = []
        chunk_index = 0

        # Get standard chunks first
        if parsed_doc.has_headings:
            chunks = self.heading_chunker.chunk(parsed_doc)
            chunk_index = len(chunks)

        # Process tables
        for table in parsed_doc.tables:
            table_chunks = self._chunk_table(table, parsed_doc, chunk_index)
            chunks.extend(table_chunks)
            chunk_index += len(table_chunks)

        return chunks

    def _chunk_table(
        self,
        table: TableData,
        parsed_doc: ParsedDocument,
        start_index: int,
    ) -> List[ChunkMetadata]:
        """
        Chunk a table respecting row boundaries.

        Args:
            table: TableData from parsed document
            parsed_doc: ParsedDocument
            start_index: Starting chunk index

        Returns:
            List of ChunkMetadata for table
        """
        chunks = []

        # Check if table fits within token limit
        table_tokens = self._estimate_tokens(table.markdown_repr)

        if table_tokens <= self.config.max_chunk_tokens:
            # Table fits in one chunk
            chunk = ChunkMetadata(
                document_id=parsed_doc.document_id,
                chunk_index=start_index,
                page_start=table.page_start,
                page_end=table.page_end,
                heading_path=[],  # Tables don't have heading path
                char_start=0,
                char_end=0,
                token_count=table_tokens,
                table_index=table.table_index,
            )
            chunks.append(chunk)
        else:
            # Split table at row boundaries
            chunks = self._split_table_by_rows(table, parsed_doc, start_index)

        return chunks

    def _split_table_by_rows(
        self,
        table: TableData,
        parsed_doc: ParsedDocument,
        start_index: int,
    ) -> List[ChunkMetadata]:
        """
        Split large table at row boundaries.

        Args:
            table: TableData
            parsed_doc: ParsedDocument
            start_index: Starting chunk index

        Returns:
            List of ChunkMetadata
        """
        chunks = []
        current_rows = []
        current_tokens = 0
        chunk_idx = start_index

        for row_idx, row in enumerate(table.rows):
            row_text = " | ".join(str(cell) for cell in row)
            row_tokens = self._estimate_tokens(row_text)

            # Check if adding this row exceeds limit
            if (
                current_tokens + row_tokens > self.config.max_chunk_tokens
                and current_rows
            ):
                # Create chunk with accumulated rows
                chunk_text = self._rows_to_markdown(current_rows, table.caption)
                chunk = ChunkMetadata(
                    document_id=parsed_doc.document_id,
                    chunk_index=chunk_idx,
                    page_start=table.page_start,
                    page_end=table.page_end,
                    heading_path=[],
                    char_start=0,
                    char_end=0,
                    token_count=current_tokens,
                    table_index=table.table_index,
                )
                chunks.append(chunk)
                chunk_idx += 1

                # Start new chunk with current row
                current_rows = [row]
                current_tokens = row_tokens
            else:
                current_rows.append(row)
                current_tokens += row_tokens

        # Don't forget remaining rows
        if current_rows:
            chunk_text = self._rows_to_markdown(current_rows, table.caption)
            chunk = ChunkMetadata(
                document_id=parsed_doc.document_id,
                chunk_index=chunk_idx,
                page_start=table.page_start,
                page_end=table.page_end,
                heading_path=[],
                char_start=0,
                char_end=0,
                token_count=current_tokens,
                table_index=table.table_index,
            )
            chunks.append(chunk)

        return chunks

    def _rows_to_markdown(
        self, rows: List[List[str]], caption: Optional[str] = None
    ) -> str:
        """Convert rows to markdown representation."""
        parts = []

        if caption:
            parts.append(f"**Table: {caption}**\n")

        if rows:
            # Header
            parts.append("| " + " | ".join(str(cell) for cell in rows[0]) + " |")
            parts.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
            # Data rows
            for row in rows[1:]:
                parts.append("| " + " | ".join(str(cell) for cell in row) + " |")

        return "\n".join(parts)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        if not text:
            return 0
        return int(len(text) / self.config.chars_per_token)
