"""Docling-based parser with format-specific pipelines."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from docling.datamodel.document import ConversionResult, DoclingDocument
from docling.document_converter import DocumentConverter

from .errors import CorruptedDocumentError, ParseError, ParseTimeoutError

logger = logging.getLogger(__name__)


def _get_settings():
    """Lazy load settings to avoid import-time validation."""
    try:
        from pipeline.config import settings

        return settings
    except Exception:
        # Return default values if settings not available
        return None


@dataclass
class TableData:
    """Represents extracted table data with markdown representation."""

    table_index: int
    page_start: int
    page_end: int
    markdown_repr: str
    rows: list[list[str]]
    caption: Optional[str] = None


@dataclass
class StructureNode:
    """Represents a node in the document structure tree."""

    level: int
    title: str
    page_start: int
    char_start: int
    children: list["StructureNode"] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """
    Represents a fully parsed document with all extracted metadata.

    This is the primary output of the DoclingParser, containing:
    - The DoclingDocument object
    - Structure tree (headings hierarchy)
    - Tables (structured data)
    - Page image references (for PDFs)
    - OCR metadata (for scanned PDFs)
    """

    file_path: str
    document_id: str
    docling_doc: DoclingDocument
    structure_tree: list[StructureNode]
    tables: list[TableData]
    page_image_paths: dict[int, Path] = field(default_factory=dict)
    has_headings: bool = False
    is_scanned_pdf: bool = False
    avg_ocr_confidence: float = 1.0
    low_confidence_pages: list[int] = field(default_factory=list)
    parsing_time_seconds: float = 0.0
    page_count: int = 0
    text_content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class DoclingParser:
    """
    Docling-based parser with format-specific pipelines.

    Supports:
    - PDF: OCR enabled, table structure recovery, page image export
    - DOCX: Heading styles mapped to H1/H2/H3, embedded images, table cells
    - PPTX: Slide titles as H1, bullet text as body, slide numbers

    Output is always a DoclingDocument regardless of input format.
    """

    def __init__(
        self,
        timeout_seconds: Optional[int] = None,
        ocr_enabled: Optional[bool] = None,
        export_page_images: Optional[bool] = None,
    ):
        """
        Initialize the Docling parser.

        Args:
            timeout_seconds: Timeout per file in seconds (default: from settings)
            ocr_enabled: Whether to enable OCR for PDFs (default: from settings)
            export_page_images: Whether to export page images (default: from settings)
        """
        settings = _get_settings()
        self.timeout_seconds = timeout_seconds or (
            settings.parser_timeout_seconds if settings else 120
        )
        self.ocr_enabled = (
            ocr_enabled
            if ocr_enabled is not None
            else (settings.ocr_enabled if settings else True)
        )
        self.export_page_images = (
            export_page_images
            if export_page_images is not None
            else (settings.export_page_images if settings else False)
        )

        self._converter: Optional[DocumentConverter] = None

    def _get_converter(self) -> DocumentConverter:
        """Get or create the document converter with appropriate options."""
        if self._converter is None:
            # Create converter with default options
            # Docling will use appropriate pipelines based on input format
            self._converter = DocumentConverter()

        return self._converter

    async def parse_file(
        self,
        file_path: str | Path,
        document_id: str,
    ) -> ParsedDocument:
        """
        Parse a document file using Docling with format-specific pipelines.

        Enforces timeout to prevent hanging on large/complex documents.

        Args:
            file_path: Path to the document file
            document_id: Unique document identifier

        Returns:
            ParsedDocument with all extracted data

        Raises:
            ParseTimeoutError: If parsing exceeds timeout
            ParseError: If parsing fails
        """
        file_path = Path(file_path)
        start_time = time.time()

        try:
            # Run parsing with timeout
            conv_result = await asyncio.wait_for(
                asyncio.to_thread(self._parse_sync, file_path),
                timeout=self.timeout_seconds,
            )

            parsing_time = time.time() - start_time

            # Build parsed document
            return await self._build_parsed_document(
                file_path=file_path,
                document_id=document_id,
                conv_result=conv_result,
                parsing_time=parsing_time,
            )

        except asyncio.TimeoutError as e:
            raise ParseTimeoutError(
                file_path=str(file_path),
                timeout_seconds=self.timeout_seconds,
            ) from e
        except ParseError:
            raise
        except Exception as e:
            logger.error(f"Parsing failed for {file_path}: {e}")
            raise ParseError(
                file_path=str(file_path),
                reason="parse_error",
                message=f"Parsing failed: {e}",
            ) from e

    def _parse_sync(self, file_path: Path) -> ConversionResult:
        """
        Synchronous parsing wrapper for async execution.

        Args:
            file_path: Path to the document

        Returns:
            Docling conversion result
        """
        converter = self._get_converter()
        result = converter.convert(str(file_path))

        if result.document is None:
            raise CorruptedDocumentError(
                file_path=str(file_path),
                message=f"Docling returned None document for {file_path}",
            )

        return result

    async def _build_parsed_document(
        self,
        file_path: Path,
        document_id: str,
        conv_result: ConversionResult,
        parsing_time: float,
    ) -> ParsedDocument:
        """
        Build ParsedDocument from conversion result.

        Extracts:
        - Structure tree
        - Tables
        - Page images
        - OCR metadata

        Args:
            file_path: Original file path
            document_id: Document identifier
            conv_result: Docling conversion result
            parsing_time: Time taken to parse

        Returns:
            Fully constructed ParsedDocument
        """
        doc = conv_result.document

        # Extract text content
        text_content = (
            doc.export_to_markdown() if hasattr(doc, "export_to_markdown") else ""
        )

        # Extract structure tree
        structure_tree = self._extract_structure_tree(doc)

        # Extract tables
        tables = self._extract_tables(doc)

        # Detect if scanned PDF
        is_scanned = self._detect_scanned_pdf(doc)

        # Build page image references
        page_images = {}
        if self.export_page_images and hasattr(doc, "pages"):
            page_images = self._export_page_images(doc, document_id)

        # Calculate OCR confidence if applicable
        avg_confidence = 1.0
        low_conf_pages = []
        if is_scanned:
            avg_confidence, low_conf_pages = self._calculate_ocr_confidence(doc)

        return ParsedDocument(
            file_path=str(file_path),
            document_id=document_id,
            docling_doc=doc,
            structure_tree=structure_tree,
            tables=tables,
            page_image_paths=page_images,
            has_headings=len(structure_tree) > 0,
            is_scanned_pdf=is_scanned,
            avg_ocr_confidence=avg_confidence,
            low_confidence_pages=low_conf_pages,
            parsing_time_seconds=parsing_time,
            page_count=len(doc.pages) if hasattr(doc, "pages") else 0,
            text_content=text_content,
            metadata={
                "input_format": str(conv_result.input.format)
                if hasattr(conv_result, "input")
                else None,
            },
        )

    def _extract_structure_tree(self, doc: DoclingDocument) -> list[StructureNode]:
        """
        Extract hierarchical heading structure as a navigable tree.

        Args:
            doc: DoclingDocument object

        Returns:
            List of top-level StructureNode objects with nested children
        """
        nodes: list[StructureNode] = []

        # Extract headings from document items
        if hasattr(doc, "body") and hasattr(doc.body, "children"):
            nodes = self._traverse_body_structure(doc.body.children)

        # Log warning if no headings detected
        if not nodes:
            logger.warning("No headings detected in document")

        return nodes

    def _traverse_body_structure(self, children: list) -> list[StructureNode]:
        """
        Recursively traverse document body to build structure tree.

        Args:
            children: Document body children (sections, headings, etc.)

        Returns:
            List of StructureNode objects
        """
        nodes = []

        for child in children:
            # Skip headings with no text
            if hasattr(child, "label") and "heading" in str(child.label).lower():
                title = ""
                if hasattr(child, "text"):
                    title = child.text.strip()

                if not title:
                    logger.warning("Skipping blank heading (no text)")
                    continue

                level = 1
                if hasattr(child, "level"):
                    level = child.level

                page_start = 0
                if hasattr(child, "page") and child.page is not None:
                    page_start = child.page

                char_start = 0
                if hasattr(child, "char_start") and child.char_start is not None:
                    char_start = child.char_start

                node = StructureNode(
                    level=level,
                    title=title,
                    page_start=page_start,
                    char_start=char_start,
                )

                # Process children for nested structure
                if hasattr(child, "children") and child.children:
                    node.children = self._traverse_body_structure(child.children)

                nodes.append(node)

        return nodes

    def _extract_tables(self, doc: DoclingDocument) -> list[TableData]:
        """
        Extract tables as structured data with markdown representations.

        Args:
            doc: DoclingDocument object

        Returns:
            List of TableData objects
        """
        tables = []

        # Find all tables in document
        if hasattr(doc, "tables"):
            for idx, table in enumerate(doc.tables):
                # Skip empty tables
                if hasattr(table, "data") and table.data:
                    rows = table.data
                    if len(rows) == 0 or len(rows[0]) == 0:
                        logger.warning(f"Skipping empty table at index {idx}")
                        continue
                else:
                    continue

                # Build table data
                caption = None
                if hasattr(table, "caption"):
                    caption = table.caption

                page_start = 0
                page_end = 0
                if hasattr(table, "page"):
                    page_start = table.page
                    page_end = table.page

                # Generate markdown representation
                markdown = self._table_to_markdown(table, caption)

                table_data = TableData(
                    table_index=idx,
                    page_start=page_start,
                    page_end=page_end,
                    markdown_repr=markdown,
                    rows=rows if isinstance(rows, list) else [],
                    caption=caption,
                )
                tables.append(table_data)

        return tables

    def _table_to_markdown(self, table, caption: Optional[str] = None) -> str:
        """
        Convert table to markdown representation.

        Args:
            table: Table object from DoclingDocument
            caption: Optional table caption

        Returns:
            Markdown string
        """
        if not hasattr(table, "data") or not table.data:
            return ""

        rows = table.data
        markdown_parts = []

        # Add caption if provided
        if caption:
            markdown_parts.append(f"**Table: {caption}**\n")

        # Build markdown table
        if len(rows) > 0:
            # Header row
            header = "| " + " | ".join(str(cell) for cell in rows[0]) + " |"
            markdown_parts.append(header)

            # Separator row
            separator = "| " + " | ".join(["---"] * len(rows[0])) + " |"
            markdown_parts.append(separator)

            # Data rows
            for row in rows[1:]:
                row_str = "| " + " | ".join(str(cell) for cell in row) + " |"
                markdown_parts.append(row_str)

        return "\n".join(markdown_parts)

    def _export_page_images(
        self, doc: DoclingDocument, document_id: str
    ) -> dict[int, Path]:
        """
        Export page images as PNG files for source preview.

        Args:
            doc: DoclingDocument object
            document_id: Unique document identifier

        Returns:
            Dictionary mapping page number to image path
        """
        image_dir = Path("data/page_images") / document_id
        image_dir.mkdir(parents=True, exist_ok=True)

        page_images = {}

        if hasattr(doc, "pages"):
            for page_num, page in enumerate(doc.pages, start=1):
                if hasattr(page, "image") and page.image is not None:
                    image_path = image_dir / f"page_{page_num}.png"
                    # Save image (Docling provides PIL Image)
                    page.image.save(str(image_path))
                    page_images[page_num] = image_path

        return page_images

    def _detect_scanned_pdf(self, doc: DoclingDocument) -> bool:
        """
        Detect if PDF is scanned/image-only with no text layer.

        Detection criterion: < 100 chars per page on average.

        Args:
            doc: DoclingDocument object

        Returns:
            True if detected as scanned PDF
        """
        if not hasattr(doc, "pages") or len(doc.pages) == 0:
            return False

        total_chars = 0
        page_count = len(doc.pages)

        for page in doc.pages:
            if hasattr(page, "text"):
                total_chars += len(page.text or "")

        avg_chars_per_page = total_chars / page_count if page_count > 0 else 0
        settings = _get_settings()
        threshold = settings.scanned_pdf_char_threshold if settings else 100

        return avg_chars_per_page < threshold

    def _calculate_ocr_confidence(
        self, doc: DoclingDocument
    ) -> tuple[float, list[int]]:
        """
        Calculate OCR confidence scores per page.

        Args:
            doc: DoclingDocument object

        Returns:
            Tuple of (average confidence, list of low confidence page numbers)
        """
        if not hasattr(doc, "pages"):
            return 1.0, []

        confidences = []
        low_confidence_pages = []

        for page_num, page in enumerate(doc.pages, start=1):
            if hasattr(page, "ocr_confidence"):
                confidence = page.ocr_confidence
                confidences.append(confidence)

                settings = _get_settings()
                threshold = settings.ocr_confidence_threshold if settings else 0.6
                if confidence < threshold:
                    low_confidence_pages.append(page_num)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 1.0

        return avg_confidence, low_confidence_pages
