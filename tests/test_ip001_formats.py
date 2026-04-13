"""Unit tests for IP-001: Define Supported File Format Contract."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ingestion.errors import (
    UnsupportedFormatError,
)
from ingestion.formats import (
    SupportedFormat,
    detect_mime_type,
    get_supported_extensions,
    get_supported_mime_types,
    is_supported_extension,
    validate_file_format,
)


class TestSupportedFormat:
    """Test SupportedFormat enum."""

    def test_pdf_extension(self):
        assert SupportedFormat.PDF.extensions == [".pdf"]

    def test_docx_extension(self):
        assert SupportedFormat.DOCX.extensions == [".docx"]

    def test_pptx_extension(self):
        assert SupportedFormat.PPTX.extensions == [".pptx"]

    def test_txt_extension(self):
        assert SupportedFormat.TXT.extensions == [".txt"]

    def test_md_extension(self):
        assert SupportedFormat.MD.extensions == [".md", ".markdown"]


class TestSupportedFormatsConfig:
    """Test configuration-driven supported formats."""

    def test_get_supported_mime_types(self):
        mime_types = get_supported_mime_types()
        assert len(mime_types) == 5
        assert "application/pdf" in mime_types
        assert (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            in mime_types
        )

    def test_get_supported_extensions(self):
        extensions = get_supported_extensions()
        assert len(extensions) == 5
        assert ".pdf" in extensions
        assert ".docx" in extensions


class TestMimeTypeDetection:
    """Test MIME type detection functionality."""

    def test_detect_mime_type_pdf(self, tmp_path):
        """Test MIME detection for PDF file."""
        pdf_file = tmp_path / "test.pdf"
        # Minimal PDF header
        pdf_file.write_bytes(b"%PDF-1.4\n%Test PDF content")

        with patch("ingestion.formats.magic.Magic") as mock_magic:
            mock_instance = MagicMock()
            mock_instance.from_buffer.return_value = "application/pdf"
            mock_magic.return_value = mock_instance

            mime = detect_mime_type(pdf_file)
            assert mime == "application/pdf"

    def test_detect_mime_type_file_not_found(self):
        """Test MIME detection raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            detect_mime_type("/nonexistent/file.pdf")

    def test_detect_mime_type_txt(self, tmp_path):
        """Test MIME detection for text file."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello, world!")

        with patch("ingestion.formats.magic.Magic") as mock_magic:
            mock_instance = MagicMock()
            mock_instance.from_buffer.return_value = "text/plain"
            mock_magic.return_value = mock_instance

            mime = detect_mime_type(txt_file)
            assert mime == "text/plain"


class TestValidateFileFormat:
    """Test file format validation."""

    def test_validate_valid_pdf(self, tmp_path):
        """Test validation of valid PDF file."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\nTest content")

        with patch("ingestion.formats.detect_mime_type") as mock_detect:
            mock_detect.return_value = "application/pdf"
            result = validate_file_format(pdf_file)
            assert result == "application/pdf"

    def test_validate_valid_docx(self, tmp_path):
        """Test validation of valid DOCX file."""
        docx_file = tmp_path / "test.docx"
        docx_file.write_bytes(b"PK\x03\x04")  # ZIP header for DOCX

        with patch("ingestion.formats.detect_mime_type") as mock_detect:
            mock_detect.return_value = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            result = validate_file_format(docx_file)
            assert (
                result
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

    def test_validate_unsupported_format(self, tmp_path):
        """Test validation rejects unsupported formats."""
        exe_file = tmp_path / "test.exe"
        exe_file.write_bytes(b"MZ\x90\x00")

        with patch("ingestion.formats.detect_mime_type") as mock_detect:
            mock_detect.return_value = "application/x-msdownload"
            with pytest.raises(UnsupportedFormatError) as exc_info:
                validate_file_format(exe_file)

            assert exc_info.value.reason == "unsupported_format"
            assert exc_info.value.detected_mime == "application/x-msdownload"

    def test_validate_mime_mismatch(self, tmp_path):
        """Test validation detects MIME mismatch (e.g., .exe disguised as .pdf)."""
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"This is actually an EXE file")

        with patch("ingestion.formats.detect_mime_type") as mock_detect:
            mock_detect.return_value = "application/x-msdownload"
            with pytest.raises(UnsupportedFormatError) as exc_info:
                validate_file_format(fake_pdf)

            assert exc_info.value.reason == "unsupported_format"
            assert exc_info.value.file_path == str(fake_pdf)

    def test_validate_empty_file(self, tmp_path):
        """Test validation handles empty files."""
        empty_file = tmp_path / "empty.pdf"
        empty_file.write_bytes(b"")

        with patch("ingestion.formats.detect_mime_type") as mock_detect:
            mock_detect.return_value = "application/x-empty"
            with pytest.raises(UnsupportedFormatError):
                validate_file_format(empty_file)


class TestSupportedExtension:
    """Test extension checking."""

    def test_supported_pdf_extension(self):
        assert is_supported_extension("test.pdf") is True

    def test_supported_docx_extension(self):
        assert is_supported_extension("test.docx") is True

    def test_unsupported_extension(self):
        assert is_supported_extension("test.exe") is False

    def test_case_insensitive_extension(self):
        assert is_supported_extension("test.PDF") is True
        assert is_supported_extension("test.Docx") is True


class TestIntakeError:
    """Test IntakeError hierarchy."""

    def test_unsupported_format_error(self):
        """Test UnsupportedFormatError creation."""
        error = UnsupportedFormatError(
            file_path="/path/to/file.exe",
            detected_mime="application/x-msdownload",
        )
        assert error.reason == "unsupported_format"
        assert error.file_path == "/path/to/file.exe"
        assert "unsupported_format" in str(error)

    def test_intake_error_message(self):
        """Test IntakeError default message generation."""
        error = UnsupportedFormatError(
            file_path="/path/to/file.exe",
            detected_mime="application/x-msdownload",
        )
        assert "/path/to/file.exe" in error.message
        assert "unsupported_format" in error.message
