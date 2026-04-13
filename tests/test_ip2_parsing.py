"""Unit tests for IP2 - Parsing (Docling) Epic."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from parsing.docling_parser import (
    DoclingParser,
    ParsedDocument,
    StructureNode,
    TableData,
)
from parsing.errors import (
    CorruptedDocumentError,
    ParseError,
    ParseTimeoutError,
    UnsupportedFormatParseError,
)


class TestStructureNode:
    """Test StructureNode data class."""

    def test_structure_node_creation(self):
        """Test basic structure node creation."""
        node = StructureNode(
            level=1,
            title="Introduction",
            page_start=1,
            char_start=0,
        )

        assert node.level == 1
        assert node.title == "Introduction"
        assert node.page_start == 1
        assert node.char_start == 0
        assert node.children == []

    def test_structure_node_with_children(self):
        """Test structure node with nested children."""
        child = StructureNode(level=2, title="Subsection", page_start=1, char_start=50)
        parent = StructureNode(
            level=1,
            title="Main Section",
            page_start=1,
            char_start=0,
            children=[child],
        )

        assert len(parent.children) == 1
        assert parent.children[0].title == "Subsection"


class TestTableData:
    """Test TableData data class."""

    def test_table_data_creation(self):
        """Test basic table data creation."""
        rows = [["Header1", "Header2"], ["Value1", "Value2"]]
        table = TableData(
            table_index=0,
            page_start=1,
            page_end=1,
            markdown_repr="| Header1 | Header2 |\n| --- | --- |\n| Value1 | Value2 |",
            rows=rows,
        )

        assert table.table_index == 0
        assert table.page_start == 1
        assert table.page_end == 1
        assert len(table.rows) == 2

    def test_table_data_with_caption(self):
        """Test table data with caption."""
        table = TableData(
            table_index=0,
            page_start=1,
            page_end=1,
            markdown_repr="",
            rows=[],
            caption="Sample Table",
        )

        assert table.caption == "Sample Table"


class TestParsedDocument:
    """Test ParsedDocument data class."""

    def test_parsed_document_creation(self):
        """Test basic parsed document creation."""
        doc = MagicMock()
        parsed = ParsedDocument(
            file_path="/path/to/file.pdf",
            document_id="doc-123",
            docling_doc=doc,
            structure_tree=[],
            tables=[],
        )

        assert parsed.file_path == "/path/to/file.pdf"
        assert parsed.document_id == "doc-123"
        assert parsed.has_headings is False  # Empty structure tree
        assert parsed.is_scanned_pdf is False

    def test_parsed_document_with_headings(self):
        """Test parsed document with headings."""
        doc = MagicMock()
        node = StructureNode(level=1, title="Heading", page_start=1, char_start=0)
        parsed = ParsedDocument(
            file_path="/path/to/file.pdf",
            document_id="doc-123",
            docling_doc=doc,
            structure_tree=[node],
            tables=[],
            has_headings=True,  # Explicitly set to True
        )

        assert parsed.has_headings is True

    def test_parsed_document_scanned_pdf(self):
        """Test parsed document flagged as scanned."""
        doc = MagicMock()
        parsed = ParsedDocument(
            file_path="/path/to/scanned.pdf",
            document_id="doc-123",
            docling_doc=doc,
            structure_tree=[],
            tables=[],
            is_scanned_pdf=True,
            avg_ocr_confidence=0.5,
            low_confidence_pages=[1, 2, 3],
        )

        assert parsed.is_scanned_pdf is True
        assert parsed.avg_ocr_confidence == 0.5
        assert len(parsed.low_confidence_pages) == 3


class TestDoclingParser:
    """Test DoclingParser functionality."""

    @pytest.fixture
    def parser(self):
        """Create parser instance with short timeout for testing."""
        return DoclingParser(
            timeout_seconds=5,
            ocr_enabled=False,
            export_page_images=False,
        )

    def test_parser_initialization(self, parser):
        """Test parser initialization."""
        assert parser.timeout_seconds == 5
        assert parser.ocr_enabled is False
        assert parser.export_page_images is False

    def test_parser_default_settings(self):
        """Test parser uses settings defaults."""
        parser = DoclingParser()
        assert parser.timeout_seconds == 120
        assert parser.ocr_enabled is True
        assert parser.export_page_images is False

    @patch("parsing.docling_parser._get_settings")
    @patch("parsing.docling_parser.DocumentConverter")
    def test_get_converter(self, mock_converter, mock_settings, parser):
        """Test converter creation."""
        mock_instance = MagicMock()
        mock_converter.return_value = mock_instance

        converter = parser._get_converter()

        assert converter == mock_instance
        assert parser._converter is not None

    @pytest.mark.asyncio
    async def test_parse_file_success(self, parser, tmp_path):
        """Test parsing a file successfully."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("Test content")

        # Mock conversion result
        mock_doc = MagicMock()
        mock_doc.export_to_markdown.return_value = "# Heading\nContent"
        mock_doc.pages = []
        mock_doc.body.children = []
        mock_doc.tables = []

        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_result.input.format = "PDF"

        with patch.object(parser, "_parse_sync", return_value=mock_result):
            result = await parser.parse_file(test_file, "doc-123")

        assert isinstance(result, ParsedDocument)
        assert result.file_path == str(test_file)
        assert result.document_id == "doc-123"
        assert result.parsing_time_seconds >= 0

    @pytest.mark.asyncio
    async def test_parse_file_timeout(self, parser, tmp_path):
        """Test parsing timeout."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("Test content")

        # Mock slow parsing that exceeds timeout
        def slow_parse(*args, **kwargs):
            time.sleep(10)  # Longer than timeout
            return MagicMock()

        parser.timeout_seconds = 1  # Set 1 second timeout

        with patch.object(parser, "_parse_sync", side_effect=slow_parse):
            with pytest.raises(ParseTimeoutError):
                await parser.parse_file(test_file, "doc-123")

    @pytest.mark.asyncio
    async def test_parse_file_error(self, parser, tmp_path):
        """Test parsing error handling."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("Test content")

        with patch.object(
            parser, "_parse_sync", side_effect=Exception("Parse failure")
        ):
            with pytest.raises(ParseError) as exc_info:
                await parser.parse_file(test_file, "doc-123")

        assert "Parse failure" in str(exc_info.value)

    def test_parse_sync_returns_none_raises(self, parser, tmp_path):
        """Test that None document from converter raises CorruptedDocumentError."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("Test content")

        mock_result = MagicMock()
        mock_result.document = None

        with patch.object(parser, "_get_converter") as mock_converter:
            mock_converter.return_value.convert.return_value = mock_result

            with pytest.raises(CorruptedDocumentError):
                parser._parse_sync(test_file)

    def test_extract_structure_tree_empty(self, parser):
        """Test structure tree extraction with no headings."""
        mock_doc = MagicMock()
        mock_doc.body.children = []

        tree = parser._extract_structure_tree(mock_doc)

        assert tree == []

    def test_extract_structure_tree_with_headings(self, parser):
        """Test structure tree extraction with headings."""
        mock_heading = MagicMock()
        mock_heading.label = "section-heading"
        mock_heading.text = "Introduction"
        mock_heading.level = 1
        mock_heading.page = 1
        mock_heading.char_start = 0
        mock_heading.children = []

        mock_doc = MagicMock()
        mock_doc.body.children = [mock_heading]

        tree = parser._extract_structure_tree(mock_doc)

        assert len(tree) == 1
        assert tree[0].title == "Introduction"
        assert tree[0].level == 1

    def test_extract_structure_tree_skips_blank_headings(self, parser):
        """Test that blank headings are skipped."""
        mock_heading = MagicMock()
        mock_heading.label = "section-heading"
        mock_heading.text = ""  # Blank heading
        mock_heading.level = 1
        mock_heading.page = 1
        mock_heading.char_start = 0
        mock_heading.children = []

        mock_doc = MagicMock()
        mock_doc.body.children = [mock_heading]

        tree = parser._extract_structure_tree(mock_doc)

        assert tree == []

    def test_extract_tables_empty(self, parser):
        """Test table extraction with no tables."""
        mock_doc = MagicMock()
        mock_doc.tables = []

        tables = parser._extract_tables(mock_doc)

        assert tables == []

    def test_extract_tables_with_data(self, parser):
        """Test table extraction with table data."""
        mock_table = MagicMock()
        mock_table.data = [
            ["Header1", "Header2"],
            ["Value1", "Value2"],
        ]
        mock_table.caption = "Sample Table"
        mock_table.page = 1

        mock_doc = MagicMock()
        mock_doc.tables = [mock_table]

        tables = parser._extract_tables(mock_doc)

        assert len(tables) == 1
        assert tables[0].table_index == 0
        assert len(tables[0].rows) == 2
        assert tables[0].caption == "Sample Table"

    def test_extract_tables_skips_empty_tables(self, parser):
        """Test that empty tables are skipped."""
        mock_table = MagicMock()
        mock_table.data = []  # Empty table

        mock_doc = MagicMock()
        mock_doc.tables = [mock_table]

        tables = parser._extract_tables(mock_doc)

        assert tables == []

    def test_table_to_markdown(self, parser):
        """Test table to markdown conversion."""
        mock_table = MagicMock()
        mock_table.data = [
            ["Name", "Age"],
            ["Alice", "30"],
            ["Bob", "25"],
        ]

        markdown = parser._table_to_markdown(mock_table)

        assert "Name" in markdown
        assert "Age" in markdown
        assert "Alice" in markdown
        assert "|" in markdown  # Markdown table format

    def test_table_to_markdown_with_caption(self, parser):
        """Test table to markdown with caption."""
        mock_table = MagicMock()
        mock_table.data = [["Header"]]
        caption = "Demographics"

        markdown = parser._table_to_markdown(mock_table, caption)

        assert "Demographics" in markdown
        assert "**Table:" in markdown

    def test_detect_scanned_pdf_with_text(self, parser):
        """Test scanned PDF detection with text layer."""
        mock_page1 = MagicMock()
        mock_page1.text = "A" * 500  # Lots of text
        mock_page2 = MagicMock()
        mock_page2.text = "B" * 500

        mock_doc = MagicMock()
        mock_doc.pages = [mock_page1, mock_page2]

        is_scanned = parser._detect_scanned_pdf(mock_doc)

        assert is_scanned is False  # Has text layer

    def test_detect_scanned_pdf_no_text(self, parser):
        """Test scanned PDF detection without text layer."""
        mock_page1 = MagicMock()
        mock_page1.text = ""  # No text
        mock_page2 = MagicMock()
        mock_page2.text = ""

        mock_doc = MagicMock()
        mock_doc.pages = [mock_page1, mock_page2]

        is_scanned = parser._detect_scanned_pdf(mock_doc)

        assert is_scanned is True  # Scanned PDF

    def test_detect_scanned_pdf_low_text(self, parser):
        """Test scanned PDF detection with low text content."""
        mock_page = MagicMock()
        mock_page.text = "Short"  # Very little text

        mock_doc = MagicMock()
        mock_doc.pages = [mock_page]

        is_scanned = parser._detect_scanned_pdf(mock_doc)

        assert is_scanned is True  # Below threshold

    def test_calculate_ocr_confidence(self, parser):
        """Test OCR confidence calculation."""
        mock_page1 = MagicMock()
        mock_page1.ocr_confidence = 0.8
        mock_page2 = MagicMock()
        mock_page2.ocr_confidence = 0.9

        mock_doc = MagicMock()
        mock_doc.pages = [mock_page1, mock_page2]

        avg_confidence, low_conf_pages = parser._calculate_ocr_confidence(mock_doc)

        assert abs(avg_confidence - 0.85) < 0.001
        assert low_conf_pages == []  # Both above threshold

    def test_calculate_ocr_confidence_low_pages(self, parser):
        """Test OCR confidence with low confidence pages."""
        mock_page1 = MagicMock()
        mock_page1.ocr_confidence = 0.4  # Below threshold
        mock_page2 = MagicMock()
        mock_page2.ocr_confidence = 0.8

        mock_doc = MagicMock()
        mock_doc.pages = [mock_page1, mock_page2]

        avg_confidence, low_conf_pages = parser._calculate_ocr_confidence(mock_doc)

        assert abs(avg_confidence - 0.6) < 0.001
        assert 1 in low_conf_pages  # Page 1 is low confidence

    def test_export_page_images_disabled(self, parser, tmp_path):
        """Test page image export when disabled."""
        parser.export_page_images = False

        mock_doc = MagicMock()
        mock_doc.pages = []

        images = parser._export_page_images(mock_doc, "doc-123")

        assert images == {}

    def test_parsed_document_metadata(self, parser):
        """Test parsed document includes metadata."""
        mock_doc = MagicMock()
        mock_doc.export_to_markdown.return_value = "# Test"
        mock_doc.pages = []
        mock_doc.body.children = []
        mock_doc.tables = []

        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_result.input.format = "PDF"

        async def run_test():
            return await parser._build_parsed_document(
                file_path=Path("/test.pdf"),
                document_id="doc-123",
                conv_result=mock_result,
                parsing_time=1.5,
            )

        parsed = asyncio.get_event_loop().run_until_complete(run_test())

        assert parsed.metadata["input_format"] == "PDF"
        assert parsed.parsing_time_seconds >= 0


class TestParseErrors:
    """Test parsing error classes."""

    def test_parse_error(self):
        """Test base ParseError."""
        error = ParseError(
            file_path="/path/to/file.pdf",
            reason="parse_failure",
        )

        assert error.file_path == "/path/to/file.pdf"
        assert error.reason == "parse_failure"
        assert "parse_failure" in str(error)

    def test_parse_timeout_error(self):
        """Test ParseTimeoutError."""
        error = ParseTimeoutError(
            file_path="/path/to/file.pdf",
            timeout_seconds=120,
        )

        assert error.reason == "parse_timeout"
        assert "120s" in str(error)

    def test_unsupported_format_parse_error(self):
        """Test UnsupportedFormatParseError."""
        error = UnsupportedFormatParseError(
            file_path="/path/to/file.exe",
            detected_mime="application/x-msdownload",
        )

        assert error.reason == "unsupported_format"
        assert error.detected_mime == "application/x-msdownload"

    def test_corrupted_document_error(self):
        """Test CorruptedDocumentError."""
        error = CorruptedDocumentError(
            file_path="/path/to/corrupted.pdf",
            message="File is corrupted",
        )

        assert error.reason == "corrupted_document"
        assert error.file_path == "/path/to/corrupted.pdf"


class TestDoclingParserIntegration:
    """Integration tests for DoclingParser."""

    @pytest.mark.asyncio
    async def test_full_parse_pipeline(self, tmp_path):
        """Test full parsing pipeline with mocked Docling."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("Test PDF content")

        parser = DoclingParser(
            timeout_seconds=5,
            ocr_enabled=False,
            export_page_images=False,
        )

        # Mock the entire Docling conversion
        mock_doc = MagicMock()
        mock_doc.export_to_markdown.return_value = "# Heading 1\nContent here"
        mock_doc.pages = []
        mock_doc.body.children = []
        mock_doc.tables = []

        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_result.input.format = "PDF"

        with patch.object(parser, "_parse_sync", return_value=mock_result):
            result = await parser.parse_file(test_file, "doc-test-123")

        assert isinstance(result, ParsedDocument)
        assert result.document_id == "doc-test-123"
        assert result.has_headings is False
        assert result.is_scanned_pdf is False
        assert result.tables == []

    @pytest.mark.asyncio
    async def test_parse_with_structure_tree(self, tmp_path):
        """Test parsing with heading structure extraction."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("Test content")

        parser = DoclingParser(
            timeout_seconds=5,
            ocr_enabled=False,
            export_page_images=False,
        )

        # Mock document with headings
        mock_heading = MagicMock()
        mock_heading.label = "section-heading"
        mock_heading.text = "Introduction"
        mock_heading.level = 1
        mock_heading.page = 1
        mock_heading.char_start = 0
        mock_heading.children = []

        mock_doc = MagicMock()
        mock_doc.export_to_markdown.return_value = "# Introduction\nContent"
        mock_doc.pages = []
        mock_doc.body.children = [mock_heading]
        mock_doc.tables = []

        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_result.input.format = "PDF"

        with patch.object(parser, "_parse_sync", return_value=mock_result):
            result = await parser.parse_file(test_file, "doc-123")

        assert result.has_headings is True
        assert len(result.structure_tree) == 1
        assert result.structure_tree[0].title == "Introduction"

    @pytest.mark.asyncio
    async def test_parse_with_tables(self, tmp_path):
        """Test parsing with table extraction."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("Test content")

        parser = DoclingParser(
            timeout_seconds=5,
            ocr_enabled=False,
            export_page_images=False,
        )

        # Mock document with table
        mock_table = MagicMock()
        mock_table.data = [
            ["Column1", "Column2", "Column3"],
            ["Value1", "Value2", "Value3"],
        ]
        mock_table.caption = "Sample Table"
        mock_table.page = 1

        mock_doc = MagicMock()
        mock_doc.export_to_markdown.return_value = "# Heading\nTable content"
        mock_doc.pages = []
        mock_doc.body.children = []
        mock_doc.tables = [mock_table]

        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_result.input.format = "PDF"

        with patch.object(parser, "_parse_sync", return_value=mock_result):
            result = await parser.parse_file(test_file, "doc-123")

        assert len(result.tables) == 1
        assert result.tables[0].caption == "Sample Table"
        assert len(result.tables[0].rows) == 2
        assert "Column1" in result.tables[0].markdown_repr

    @pytest.mark.asyncio
    async def test_parse_scanned_pdf_with_ocr(self, tmp_path):
        """Test parsing scanned PDF with OCR metadata."""
        test_file = tmp_path / "scanned.pdf"
        test_file.write_text("")  # Empty text layer

        parser = DoclingParser(
            timeout_seconds=5,
            ocr_enabled=True,
            export_page_images=False,
        )

        # Mock scanned PDF with OCR
        mock_page1 = MagicMock()
        mock_page1.text = ""
        mock_page1.ocr_confidence = 0.7
        mock_page2 = MagicMock()
        mock_page2.text = ""
        mock_page2.ocr_confidence = 0.5  # Low confidence

        mock_doc = MagicMock()
        mock_doc.export_to_markdown.return_value = "OCR text"
        mock_doc.pages = [mock_page1, mock_page2]
        mock_doc.body.children = []
        mock_doc.tables = []

        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_result.input.format = "PDF"

        with patch.object(parser, "_parse_sync", return_value=mock_result):
            result = await parser.parse_file(test_file, "doc-123")

        assert result.is_scanned_pdf is True
        assert result.avg_ocr_confidence == 0.6
        assert 2 in result.low_confidence_pages
