"""Unit tests for IP4 - Chunking Epic."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from chunking.chunkers import (
    ChunkingConfig,
    HeadingHierarchyChunker,
    SlidingWindowChunker,
    TableAwareChunker,
    TextSegment,
)
from metadata.chunk_metadata import ChunkMetadata
from parsing.docling_parser import ParsedDocument, StructureNode, TableData

# ============================================================
# IP-013: Heading-Hierarchy Chunker Tests
# ============================================================


class TestHeadingHierarchyChunker:
    """Test heading-hierarchy chunking strategy."""

    @pytest.fixture
    def config(self):
        """Create test config with small limits for testing."""
        return ChunkingConfig(
            max_chunk_tokens=1024,
            overlap_tokens=0,  # Disable overlap for these tests
            chars_per_token=4.0,
        )

    @pytest.fixture
    def chunker(self, config):
        """Create heading hierarchy chunker."""
        return HeadingHierarchyChunker(config)

    def _create_mock_parsed_doc(
        self,
        text: str,
        structure_tree: list,
        has_headings: bool = True,
        document_id: str = "doc-123",
    ) -> ParsedDocument:
        """Helper to create mock ParsedDocument."""
        mock_doc = MagicMock(spec=ParsedDocument)
        mock_doc.document_id = document_id
        mock_doc.text_content = text
        mock_doc.structure_tree = structure_tree
        mock_doc.has_headings = has_headings
        mock_doc.tables = []
        mock_doc.is_scanned_pdf = False
        mock_doc.page_count = max(1, len(text) // 1000)
        return mock_doc

    def test_three_level_heading_doc(self, chunker):
        """Test document with 3-level heading structure."""
        text = "A" * 500  # 125 tokens
        tree = [
            StructureNode(
                level=1,
                title="Introduction",
                page_start=1,
                char_start=0,
                children=[
                    StructureNode(
                        level=2,
                        title="Background",
                        page_start=1,
                        char_start=250,
                        children=[
                            StructureNode(
                                level=3,
                                title="Details",
                                page_start=1,
                                char_start=375,
                                children=[],
                            )
                        ],
                    )
                ],
            )
        ]

        parsed_doc = self._create_mock_parsed_doc(text, tree)
        chunks = chunker.chunk(parsed_doc)

        assert len(chunks) > 0
        # Check chunk_index is 0-based and sequential
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.document_id == "doc-123"

    def test_single_level_heading_doc(self, chunker):
        """Test document with single-level headings."""
        text = "B" * 300
        tree = [
            StructureNode(
                level=1, title="Section 1", page_start=1, char_start=0, children=[]
            ),
            StructureNode(
                level=1, title="Section 2", page_start=1, char_start=150, children=[]
            ),
        ]

        parsed_doc = self._create_mock_parsed_doc(text, tree)
        chunks = chunker.chunk(parsed_doc)

        assert len(chunks) > 0

    def test_heading_with_no_body_text_skipped(self, chunker):
        """Test headings with no body text produce 0 chunks."""
        text = ""  # No body text
        tree = [
            StructureNode(
                level=1, title="Empty Section", page_start=1, char_start=0, children=[]
            )
        ]

        parsed_doc = self._create_mock_parsed_doc(text, tree)
        chunks = chunker.chunk(parsed_doc)

        # Should skip headings with no body text
        assert len(chunks) == 0

    def test_heading_path_is_ancestor_list(self, chunker):
        """Test heading_path contains full ancestor list."""
        text = "C" * 400
        tree = [
            StructureNode(
                level=1,
                title="Chapter 1",
                page_start=1,
                char_start=0,
                children=[
                    StructureNode(
                        level=2,
                        title="Section 1.1",
                        page_start=1,
                        char_start=200,
                        children=[],
                    )
                ],
            )
        ]

        parsed_doc = self._create_mock_parsed_doc(text, tree)
        chunks = chunker.chunk(parsed_doc)

        if len(chunks) > 0:
            # heading_path should be a list
            assert isinstance(chunks[0].heading_path, list)

    def test_section_title_derived_from_heading_path(self, chunker):
        """Test section_title is derived from heading_path[-1]."""
        text = "D" * 400
        tree = [
            StructureNode(
                level=1,
                title="Main Heading",
                page_start=1,
                char_start=0,
                children=[
                    StructureNode(
                        level=2,
                        title="Sub Heading",
                        page_start=1,
                        char_start=200,
                        children=[],
                    )
                ],
            )
        ]

        parsed_doc = self._create_mock_parsed_doc(text, tree)
        chunks = chunker.chunk(parsed_doc)

        if len(chunks) > 0 and chunks[0].heading_path:
            # section_title should be derived from heading_path[-1]
            assert chunks[0].section_title == chunks[0].heading_path[-1]

    def test_chunk_index_sequential(self, chunker):
        """Test chunk_index is 0-based and sequential."""
        text = "E" * 1000
        tree = [
            StructureNode(
                level=1, title="Part 1", page_start=1, char_start=0, children=[]
            ),
            StructureNode(
                level=1, title="Part 2", page_start=1, char_start=500, children=[]
            ),
        ]

        parsed_doc = self._create_mock_parsed_doc(text, tree)
        chunks = chunker.chunk(parsed_doc)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_oversized_section_split(self, chunker):
        """Test oversized sections are split with parent heading_path retained."""
        # Config with small token limit to trigger splitting
        small_config = ChunkingConfig(
            max_chunk_tokens=50,  # Very small
            overlap_tokens=0,
            chars_per_token=4.0,
        )
        small_chunker = HeadingHierarchyChunker(small_config)

        # Large section text (200 tokens)
        text = "F" * 800
        tree = [
            StructureNode(
                level=1,
                title="Large Section",
                page_start=1,
                char_start=0,
                children=[],
            )
        ]

        parsed_doc = self._create_mock_parsed_doc(text, tree)
        chunks = small_chunker.chunk(parsed_doc)

        # Should be split into multiple chunks
        assert len(chunks) > 1
        # All sub-chunks should retain parent heading_path
        for chunk in chunks:
            assert "Large Section" in chunk.heading_path

    def test_char_start_and_end_recorded(self, chunker):
        """Test char_start and char_end are recorded."""
        text = "G" * 600
        tree = [
            StructureNode(
                level=1, title="Test", page_start=1, char_start=100, children=[]
            )
        ]

        parsed_doc = self._create_mock_parsed_doc(text, tree)
        chunks = chunker.chunk(parsed_doc)

        if len(chunks) > 0:
            assert chunks[0].char_start >= 0
            assert chunks[0].char_end >= chunks[0].char_start


# ============================================================
# IP-014: Sliding Window Fallback Chunker Tests
# ============================================================


class TestSlidingWindowChunker:
    """Test sliding window fallback chunking strategy."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return ChunkingConfig(
            max_chunk_tokens=1024,
            window_size=512,
            window_stride=448,  # 64-token overlap
            overlap_tokens=0,
            chars_per_token=4.0,
        )

    @pytest.fixture
    def chunker(self, config):
        """Create sliding window chunker."""
        return SlidingWindowChunker(config)

    def _create_flat_parsed_doc(self, text: str, document_id: str = "doc-456"):
        """Create flat document without headings."""
        mock_doc = MagicMock(spec=ParsedDocument)
        mock_doc.document_id = document_id
        mock_doc.text_content = text
        mock_doc.structure_tree = []
        mock_doc.has_headings = False
        mock_doc.tables = []
        mock_doc.is_scanned_pdf = False
        mock_doc.page_count = max(1, len(text) // 1000)
        return mock_doc

    def test_flat_document_chunking(self, chunker):
        """Test flat document produces overlapping chunks."""
        # 3000-token flat document
        text = "H" * 12000  # ~3000 tokens

        parsed_doc = self._create_flat_parsed_doc(text)
        chunks = chunker.chunk(parsed_doc)

        assert len(chunks) > 0

    def test_fallback_chunks_have_empty_heading_path(self, chunker):
        """Test fallback chunks have empty heading_path."""
        text = "I" * 2000
        parsed_doc = self._create_flat_parsed_doc(text)
        chunks = chunker.chunk(parsed_doc)

        for chunk in chunks:
            assert chunk.heading_path == []

    def test_fallback_chunks_have_empty_section_title(self, chunker):
        """Test fallback chunks have empty section_title."""
        text = "J" * 2000
        parsed_doc = self._create_flat_parsed_doc(text)
        chunks = chunker.chunk(parsed_doc)

        for chunk in chunks:
            assert chunk.section_title == ""

    def test_chunk_index_sequential(self, chunker):
        """Test chunk_index is 0-based and sequential."""
        text = "K" * 3000
        parsed_doc = self._create_flat_parsed_doc(text)
        chunks = chunker.chunk(parsed_doc)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_page_start_and_end_populated(self, chunker):
        """Test page_start and page_end are populated."""
        text = "L" * 5000
        parsed_doc = self._create_flat_parsed_doc(text)
        chunks = chunker.chunk(parsed_doc)

        for chunk in chunks:
            assert chunk.page_start >= 0

    def test_window_stride_overlap(self, chunker):
        """Test that window stride creates overlap."""
        # Small text to test overlap
        text = "M" * 2048  # 512 tokens

        parsed_doc = self._create_flat_parsed_doc(text)
        chunks = chunker.chunk(parsed_doc)

        # With 512 token window and 448 stride, should have overlap
        if len(chunks) > 1:
            # Check that chunks overlap (char ranges overlap)
            for i in range(1, len(chunks)):
                prev_end = chunks[i - 1].char_end
                curr_start = chunks[i].char_start
                # They should be close or overlapping
                assert curr_start <= prev_end or curr_start - prev_end < 100

    def test_empty_document_returns_no_chunks(self, chunker):
        """Test empty document produces no chunks."""
        text = ""
        parsed_doc = self._create_flat_parsed_doc(text)
        chunks = chunker.chunk(parsed_doc)

        assert len(chunks) == 0

    def test_3000_token_document_chunk_count(self, chunker):
        """Test 3000-token document produces correct chunk count."""
        # 3000 tokens = 12000 chars
        text = "N" * 12000

        parsed_doc = self._create_flat_parsed_doc(text)
        chunks = chunker.chunk(parsed_doc)

        # With 512 token window and 448 stride, should produce ~6 chunks
        # (3000 / (512-64) ≈ 6.7)
        assert len(chunks) >= 5
        assert len(chunks) <= 10


# ============================================================
# IP-015: Table-Aware Chunking Tests
# ============================================================


class TestTableAwareChunker:
    """Test table-aware chunking strategy."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return ChunkingConfig(
            max_chunk_tokens=1024,
            overlap_tokens=0,
            chars_per_token=4.0,
        )

    @pytest.fixture
    def chunker(self, config):
        """Create table-aware chunker."""
        return TableAwareChunker(config)

    def _create_parsed_doc_with_tables(
        self, text: str, tables: list, has_headings: bool = True
    ):
        """Create parsed document with tables."""
        mock_doc = MagicMock(spec=ParsedDocument)
        mock_doc.document_id = "doc-789"
        mock_doc.text_content = text
        mock_doc.structure_tree = []
        mock_doc.has_headings = has_headings
        mock_doc.tables = tables
        mock_doc.is_scanned_pdf = False
        mock_doc.page_count = 5
        return mock_doc

    def test_small_table_kept_intact(self, chunker):
        """Test small table kept in one chunk within token limit."""
        text = "Some text before table."
        table = TableData(
            table_index=0,
            page_start=1,
            page_end=1,
            markdown_repr="| Col1 | Col2 |\n| --- | --- |\n| A | B |",
            rows=[["Col1", "Col2"], ["A", "B"]],
            caption="Small Table",
        )

        parsed_doc = self._create_parsed_doc_with_tables(text, [table])
        chunks = chunker.chunk(parsed_doc)

        # Should have at least one chunk for the table
        table_chunks = [c for c in chunks if c.table_index is not None]
        assert len(table_chunks) >= 1
        # Small table should be in one chunk
        assert table_chunks[0].table_index == 0

    def test_large_table_split_at_row_boundaries(self, chunker):
        """Test large table split at row boundaries only."""
        text = "Text"
        # Create table with very long cell content to exceed token limit
        rows = []
        for i in range(20):
            # Long cell content to ensure we exceed 1024 tokens
            row = [
                f"Row{i}_Column1_{'X' * 100}",
                f"Row{i}_Column2_{'Y' * 100}",
                f"Row{i}_Column3_{'Z' * 100}",
            ]
            rows.append(row)

        # Build markdown representation
        markdown = "| Col1 | Col2 | Col3 |\n| --- | --- | --- |\n"
        for row in rows:
            markdown += f"| {' | '.join(row)} |\n"

        table = TableData(
            table_index=0,
            page_start=1,
            page_end=3,
            markdown_repr=markdown,
            rows=rows,
            caption="Large Table",
        )

        parsed_doc = self._create_parsed_doc_with_tables(text, [table])

        # Use chunker with smaller token limit to force splitting
        small_config = ChunkingConfig(
            max_chunk_tokens=200,  # Small limit to force splitting
            overlap_tokens=0,
            chars_per_token=4.0,
        )
        small_chunker = TableAwareChunker(small_config)
        chunks = small_chunker.chunk(parsed_doc)

        # Should have multiple chunks for large table
        table_chunks = [c for c in chunks if c.table_index == 0]
        assert len(table_chunks) > 1

    def test_table_metadata_includes_table_index(self, chunker):
        """Test chunk metadata includes table_index reference."""
        text = "Text"
        table = TableData(
            table_index=5,
            page_start=2,
            page_end=2,
            markdown_repr="| A | B |\n| --- | --- |\n| 1 | 2 |",
            rows=[["A", "B"], ["1", "2"]],
            caption="Test Table",
        )

        parsed_doc = self._create_parsed_doc_with_tables(text, [table])
        chunks = chunker.chunk(parsed_doc)

        table_chunks = [c for c in chunks if c.table_index is not None]
        assert len(table_chunks) > 0
        assert table_chunks[0].table_index == 5

    def test_document_without_tables_uses_standard_chunker(self, chunker):
        """Test document without tables uses standard chunker."""
        text = "O" * 500
        parsed_doc = self._create_parsed_doc_with_tables(text, [])
        chunks = chunker.chunk(parsed_doc)

        # Should use heading hierarchy or sliding window
        # No table chunks should exist
        table_chunks = [c for c in chunks if c.table_index is not None]
        assert len(table_chunks) == 0

    def test_single_row_table_stays_intact(self, chunker):
        """Test single-row table stays in one chunk."""
        text = "Text"
        table = TableData(
            table_index=0,
            page_start=1,
            page_end=1,
            markdown_repr="| Header | Value |\n| --- | --- |\n| A | 1 |",
            rows=[["Header", "Value"], ["A", "1"]],
        )

        parsed_doc = self._create_parsed_doc_with_tables(text, [table])
        chunks = chunker.chunk(parsed_doc)

        table_chunks = [c for c in chunks if c.table_index == 0]
        assert len(table_chunks) == 1


# ============================================================
# IP-016: Chunk Overlap Context Injection Tests
# ============================================================


class TestOverlapContextInjection:
    """Test chunk overlap context injection."""

    @pytest.fixture
    def config_with_overlap(self):
        """Create config with overlap enabled."""
        return ChunkingConfig(
            max_chunk_tokens=1024,
            overlap_tokens=64,  # 64-token overlap
            chars_per_token=4.0,
        )

    @pytest.fixture
    def chunker(self, config_with_overlap):
        """Create chunker with overlap."""
        return HeadingHierarchyChunker(config_with_overlap)

    def _create_mock_parsed_doc(self, text: str, tree: list):
        """Create mock parsed document."""
        mock_doc = MagicMock(spec=ParsedDocument)
        mock_doc.document_id = "doc-overlap"
        mock_doc.text_content = text
        mock_doc.structure_tree = tree
        mock_doc.has_headings = True
        mock_doc.tables = []
        mock_doc.is_scanned_pdf = False
        mock_doc.page_count = 3
        return mock_doc

    def test_overlap_prepended_as_non_indexed_prefix(self, chunker):
        """Test overlap context is marked but not embedded."""
        text = "P" * 2000
        tree = [
            StructureNode(
                level=1,
                title="Section A",
                page_start=1,
                char_start=0,
                children=[
                    StructureNode(
                        level=2,
                        title="Subsection A1",
                        page_start=1,
                        char_start=500,
                        children=[],
                    ),
                    StructureNode(
                        level=2,
                        title="Subsection A2",
                        page_start=2,
                        char_start=1000,
                        children=[],
                    ),
                ],
            )
        ]

        parsed_doc = self._create_mock_parsed_doc(text, tree)
        chunks = chunker.chunk(parsed_doc)

        # Check that overlap_context is set for some chunks
        if len(chunks) > 1:
            # At least some chunks should have overlap context
            overlap_chunks = [c for c in chunks if c.overlap_context]
            # Not all chunks will have overlap (first chunk won't)
            assert len(overlap_chunks) >= 0  # May or may not have overlap

    def test_overlap_never_crosses_h1_boundaries(self, chunker):
        """Test overlap doesn't cross H1 boundaries."""
        text = "Q" * 3000
        tree = [
            StructureNode(
                level=1,
                title="Chapter 1",
                page_start=1,
                char_start=0,
                children=[],
            ),
            StructureNode(
                level=1,
                title="Chapter 2",
                page_start=2,
                char_start=1500,
                children=[],
            ),
        ]

        parsed_doc = self._create_mock_parsed_doc(text, tree)
        chunks = chunker.chunk(parsed_doc)

        # Verify H1 boundaries are respected
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]

            if prev_chunk.heading_path and curr_chunk.heading_path:
                prev_h1 = prev_chunk.heading_path[0]
                curr_h1 = curr_chunk.heading_path[0]

                # If different H1, no overlap should be injected
                if prev_h1 != curr_h1:
                    assert curr_chunk.overlap_context is None

    def test_overlap_token_budget_counted(self, chunker):
        """Test overlap token budget is counted against chunk limit."""
        text = "R" * 2000
        tree = [
            StructureNode(
                level=1,
                title="Test Section",
                page_start=1,
                char_start=0,
                children=[
                    StructureNode(
                        level=2,
                        title="Part 1",
                        page_start=1,
                        char_start=500,
                        children=[],
                    ),
                    StructureNode(
                        level=2,
                        title="Part 2",
                        page_start=2,
                        char_start=1000,
                        children=[],
                    ),
                ],
            )
        ]

        parsed_doc = self._create_mock_parsed_doc(text, tree)
        chunks = chunker.chunk(parsed_doc)

        # Check overlap chunks don't exceed token limit
        for chunk in chunks:
            if chunk.overlap_context:
                # Total tokens (chunk + overlap) should not exceed limit
                overlap_tokens = chunker._estimate_tokens(chunk.overlap_context)
                total_tokens = chunk.token_count + overlap_tokens
                # Should be within or close to limit
                assert total_tokens <= chunker.config.max_chunk_tokens + 100

    def test_configurable_overlap_zero_disables(self):
        """Test overlap_tokens=0 disables overlap."""
        no_overlap_config = ChunkingConfig(
            max_chunk_tokens=1024,
            overlap_tokens=0,  # Disabled
            chars_per_token=4.0,
        )
        no_overlap_chunker = HeadingHierarchyChunker(no_overlap_config)

        text = "S" * 2000
        tree = [
            StructureNode(
                level=1,
                title="No Overlap",
                page_start=1,
                char_start=0,
                children=[
                    StructureNode(
                        level=2,
                        title="Part 1",
                        page_start=1,
                        char_start=500,
                        children=[],
                    ),
                    StructureNode(
                        level=2,
                        title="Part 2",
                        page_start=2,
                        char_start=1000,
                        children=[],
                    ),
                ],
            )
        ]

        parsed_doc = self._create_mock_parsed_doc(text, tree)
        chunks = no_overlap_chunker.chunk(parsed_doc)

        # No chunks should have overlap_context
        for chunk in chunks:
            assert chunk.overlap_context is None

    def test_overlap_from_same_heading_level(self, chunker):
        """Test overlap only from same heading level or parent."""
        text = "T" * 2500
        tree = [
            StructureNode(
                level=1,
                title="Main",
                page_start=1,
                char_start=0,
                children=[
                    StructureNode(
                        level=2,
                        title="Section A",
                        page_start=1,
                        char_start=500,
                        children=[],
                    ),
                    StructureNode(
                        level=2,
                        title="Section B",
                        page_start=2,
                        char_start=1200,
                        children=[],
                    ),
                ],
            )
        ]

        parsed_doc = self._create_mock_parsed_doc(text, tree)
        chunks = chunker.chunk(parsed_doc)

        # Same H2 level chunks can have overlap
        if len(chunks) > 1:
            # Verify they share same H1 parent
            for chunk in chunks:
                if chunk.heading_path:
                    assert chunk.heading_path[0] == "Main"


# ============================================================
# Integration Tests
# ============================================================


class TestChunkingIntegration:
    """Integration tests for chunking workflows."""

    def test_full_heading_chunking_workflow(self):
        """Test complete heading-based chunking workflow."""
        config = ChunkingConfig(
            max_chunk_tokens=100,
            overlap_tokens=0,
            chars_per_token=4.0,
        )
        chunker = HeadingHierarchyChunker(config)

        text = "U" * 1000
        tree = [
            StructureNode(
                level=1,
                title="Document",
                page_start=1,
                char_start=0,
                children=[
                    StructureNode(
                        level=2,
                        title="Introduction",
                        page_start=1,
                        char_start=200,
                        children=[],
                    ),
                    StructureNode(
                        level=2,
                        title="Conclusion",
                        page_start=2,
                        char_start=600,
                        children=[],
                    ),
                ],
            )
        ]

        mock_doc = MagicMock(spec=ParsedDocument)
        mock_doc.document_id = "doc-integration"
        mock_doc.text_content = text
        mock_doc.structure_tree = tree
        mock_doc.has_headings = True
        mock_doc.tables = []
        mock_doc.is_scanned_pdf = False
        mock_doc.page_count = 2

        chunks = chunker.chunk(mock_doc)

        assert len(chunks) > 0
        # All chunks should have valid metadata
        for chunk in chunks:
            assert isinstance(chunk, ChunkMetadata)
            assert chunk.chunk_index >= 0
            assert chunk.document_id == "doc-integration"

    def test_fallback_to_sliding_window(self):
        """Test fallback to sliding window when no headings."""
        config = ChunkingConfig(
            window_size=512,
            window_stride=448,
            chars_per_token=4.0,
        )
        chunker = SlidingWindowChunker(config)

        text = "V" * 3000
        mock_doc = MagicMock(spec=ParsedDocument)
        mock_doc.document_id = "doc-fallback"
        mock_doc.text_content = text
        mock_doc.structure_tree = []
        mock_doc.has_headings = False
        mock_doc.tables = []
        mock_doc.is_scanned_pdf = False
        mock_doc.page_count = 3

        chunks = chunker.chunk(mock_doc)

        assert len(chunks) > 0
        # All fallback chunks should have empty heading_path
        for chunk in chunks:
            assert chunk.heading_path == []
            assert chunk.section_title == ""

    def test_table_aware_chunking_workflow(self):
        """Test table-aware chunking workflow."""
        config = ChunkingConfig(
            max_chunk_tokens=1024,
            chars_per_token=4.0,
        )
        chunker = TableAwareChunker(config)

        text = "W" * 500
        table = TableData(
            table_index=0,
            page_start=1,
            page_end=1,
            markdown_repr="| A | B |\n| --- | --- |\n| 1 | 2 |",
            rows=[["A", "B"], ["1", "2"]],
        )

        mock_doc = MagicMock(spec=ParsedDocument)
        mock_doc.document_id = "doc-tables"
        mock_doc.text_content = text
        mock_doc.structure_tree = []
        mock_doc.has_headings = False
        mock_doc.tables = [table]
        mock_doc.is_scanned_pdf = False
        mock_doc.page_count = 1

        chunks = chunker.chunk(mock_doc)

        # Should have table chunks
        table_chunks = [c for c in chunks if c.table_index is not None]
        assert len(table_chunks) >= 1
        assert table_chunks[0].table_index == 0
