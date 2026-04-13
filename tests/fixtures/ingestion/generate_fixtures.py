"""Generate test fixtures for the ingestion pipeline.

Creates binary test files for all supported document types and edge cases.
Run: python tests/fixtures/ingestion/generate_fixtures.py
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent


def _write_pdf_with_content(
    path: Path,
    text: str,
    headings: list[str] | None = None,
    page_count: int = 1,
) -> None:
    """Write a minimal valid PDF with given text content."""
    objects = []

    # Object 1: Catalog
    catalog = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    objects.append(catalog)

    # Build page content
    page_contents = []
    for i in range(page_count):
        page_text = text
        if headings and i < len(headings):
            page_text = f"{headings[i]}\n\n{text}"

        # PDF stream: text content
        stream = f"BT /F1 12 Tf 50 750 Td ({page_text}) Tj ET".encode()
        stream_obj = (
            f"{3 + i * 2} 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode()
            + stream
            + b"\nendstream\nendobj\n"
        )
        page_contents.append(stream_obj)

    # Object 2: Pages
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(page_count))
    pages_obj = (
        b"2 0 obj\n<< /Type /Pages /Kids ["
        + kids.encode()
        + b"] /Count "
        + str(page_count).encode()
        + b" /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>\nendobj\n"
    )
    objects.append(pages_obj)

    # Add page content objects
    for pc in page_contents:
        objects.append(pc)

    # Build xref
    sum(len(o) for o in objects) + len(b"%PDF-1.4\n")

    # Write PDF
    with open(path, "wb") as f:
        f.write(b"%PDF-1.4\n")
        offsets = []
        offset = 5  # After %PDF-1.4\n
        for obj in objects:
            offsets.append(offset)
            f.write(obj)
            offset += len(obj)

        # xref table
        xref_start = offset
        f.write(f"xref\n0 {len(objects) + 1}\n".encode())
        f.write(b"0000000000 65535 f \n")
        for o in offsets:
            f.write(f"{o:010d} 00000 n \n".encode())
        f.write(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode())
        f.write(f"startxref\n{xref_start}\n%%EOF\n".encode())


def _write_docx_with_content(
    path: Path, text: str, headings: list[str] | None = None
) -> None:
    """Write a minimal valid DOCX with given text content."""
    # DOCX is a ZIP file with XML content
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        # [Content_Types].xml
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main.xml"/>'
            "</Types>",
        )

        # word/document.xml
        body_parts = []
        if headings:
            for h in headings:
                body_parts.append(
                    f'<w:p><w:r><w:rPr><w:b/><w:sz w:val="28"/></w:rPr><w:t>{h}</w:t></w:r></w:p>'
                )
        body_parts.append(f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>")

        zf.writestr(
            "word/document.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body>{''.join(body_parts)}</w:body></w:document>",
        )

        # word/_rels/document.xml.rels
        zf.writestr(
            "word/_rels/document.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            "</Relationships>",
        )

        # _rels/.rels
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            "</Relationships>",
        )


def generate_all_fixtures() -> None:
    """Generate all test fixtures."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. minimal_valid.pdf - Basic valid PDF with content
    _write_pdf_with_content(
        FIXTURE_DIR / "minimal_valid.pdf",
        "This is a minimal valid PDF file for testing the ingestion pipeline.",
    )

    # 2. multi_heading.pdf - PDF with multiple heading-like content
    _write_pdf_with_content(
        FIXTURE_DIR / "multi_heading.pdf",
        "Content text under the heading.",
        headings=["Introduction", "Methods", "Results", "Conclusion"],
        page_count=2,
    )

    # 3. no_headings.pdf - PDF without any heading structure
    _write_pdf_with_content(
        FIXTURE_DIR / "no_headings.pdf",
        "This document has no headings. It is just a flat text document with multiple paragraphs of content.",
    )

    # 4. valid.docx - Valid DOCX with headings
    _write_docx_with_content(
        FIXTURE_DIR / "valid.docx",
        "This is the body text of the document.",
        headings=["Engineering 2025-S1 Lecture - Introduction to Machine Learning"],
    )

    # 5. valid.pptx - Minimal valid PPTX (ZIP-based like DOCX)
    with zipfile.ZipFile(FIXTURE_DIR / "valid.pptx", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
            "</Types>",
        )
        zf.writestr(
            "ppt/presentation.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
            '<p:sldIdLst><p:sldId id="256" r:id="rId1" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/></p:sldIdLst>'
            "</p:presentation>",
        )
        zf.writestr(
            "ppt/_rels/presentation.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>'
            "</Relationships>",
        )
        zf.writestr(
            "ppt/slides/slide1.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
            '<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
            '<p:sp><p:nvSpPr><p:cNvPr id="2" name="Title"/><p:cNvSpPr><a:spLocks noGrp="1" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"/></p:cNvSpPr><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr>'
            '<p:txBody><a:bodyPr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US" sz="1800"/><a:t>Introduction to ML</a:t></a:r></a:p></p:txBody></p:sp>'
            "</p:spTree></p:cSld></p:sld>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>'
            "</Relationships>",
        )

    # 6. corrupted.docx - Invalid ZIP content (not a real DOCX)
    with zipfile.ZipFile(FIXTURE_DIR / "corrupted.docx", "w") as zf:
        zf.writestr("garbage.txt", b"This is not a valid DOCX file.")

    # 7. encrypted.pdf - PDF with encrypt marker
    _write_pdf_with_content(
        FIXTURE_DIR / "encrypted.pdf",
        "Encrypted content placeholder.",
    )
    # Add encrypt marker to make it look encrypted
    with open(FIXTURE_DIR / "encrypted.pdf", "rb") as f:
        content = f.read()
    # Insert /Encrypt dictionary before trailer
    content = content.replace(
        b"trailer",
        b"trailer\n<< /Encrypt << /Filter /Standard /V 1 /R 2 >> >>",
    )
    with open(FIXTURE_DIR / "encrypted.pdf", "wb") as f:
        f.write(content)

    # 8. oversized.pdf - Stub file that exceeds 50MB (sparse)
    oversized_path = FIXTURE_DIR / "oversized.pdf"
    with open(oversized_path, "wb") as f:
        f.write(b"%PDF-1.4\n")
        # Write enough content to exceed 50MB
        f.write(b"X" * (50 * 1024 * 1024 + 1))
        f.write(b"\n%%EOF\n")

    # Generate fixture manifest
    manifest = {
        "version": "1.0.0",
        "fixtures": [
            {
                "filename": "minimal_valid.pdf",
                "description": "Basic valid PDF with simple text content",
                "tests": [
                    "Format validation passes",
                    "Single chunk produced",
                    "No heading structure expected",
                ],
                "expected_chunks": 1,
                "expected_headings": [],
                "file_type": "happy_path",
            },
            {
                "filename": "multi_heading.pdf",
                "description": "PDF with multiple headings across 2 pages",
                "tests": [
                    "Heading structure extracted",
                    "Multiple chunks produced",
                    "Page boundaries respected",
                ],
                "expected_chunks": 4,
                "expected_headings": [
                    "Introduction",
                    "Methods",
                    "Results",
                    "Conclusion",
                ],
                "file_type": "happy_path",
            },
            {
                "filename": "no_headings.pdf",
                "description": "PDF without any heading structure (flat text)",
                "tests": [
                    "Fallback chunking used",
                    "Single chunk from flat content",
                ],
                "expected_chunks": 1,
                "expected_headings": [],
                "file_type": "edge_case",
            },
            {
                "filename": "valid.docx",
                "description": "Valid DOCX with heading structure",
                "tests": [
                    "DOCX format detected",
                    "Heading from filename convention",
                    "Body text chunked",
                ],
                "expected_chunks": 1,
                "expected_headings": [
                    "Engineering 2025-S1 Lecture - Introduction to Machine Learning"
                ],
                "file_type": "happy_path",
            },
            {
                "filename": "valid.pptx",
                "description": "Valid PPTX with slide content",
                "tests": [
                    "PPTX format detected",
                    "Slide title extracted",
                ],
                "expected_chunks": 1,
                "expected_headings": [],
                "file_type": "happy_path",
            },
            {
                "filename": "corrupted.docx",
                "description": "Invalid DOCX (ZIP with garbage content)",
                "tests": [
                    "Format validation fails or produces empty output",
                    "CorruptedDocumentError or empty content",
                ],
                "expected_chunks": 0,
                "expected_headings": [],
                "file_type": "error_path",
            },
            {
                "filename": "encrypted.pdf",
                "description": "PDF with /Encrypt dictionary",
                "tests": [
                    "Encryption detected",
                    "EncryptedFileError raised",
                ],
                "expected_chunks": 0,
                "expected_headings": [],
                "file_type": "error_path",
            },
            {
                "filename": "oversized.pdf",
                "description": "PDF stub exceeding 50MB size limit",
                "tests": [
                    "FileTooLargeError raised",
                    "Rejected before parsing",
                ],
                "expected_chunks": 0,
                "expected_headings": [],
                "file_type": "error_path",
                "size_bytes": 52428801,
            },
        ],
    }

    manifest_path = FIXTURE_DIR / "fixture_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    generate_all_fixtures()
