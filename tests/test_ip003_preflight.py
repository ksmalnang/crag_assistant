"""Unit tests for IP-003: Build File Size and Corruption Pre-Flight Checks."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from docx.opc.exceptions import PackageNotFoundError
from pypdf.errors import PdfReadError

from ingestion.errors import (
    CorruptedFileError,
    EmptyFileError,
    EncryptedFileError,
    FileTooLargeError,
)
from ingestion.preflight import (
    PreflightValidator,
    validate_file_preflight,
)


class TestPreflightValidator:
    """Test preflight validation."""

    @pytest.fixture
    def validator(self):
        """Create validator with small file size for testing."""
        return PreflightValidator(max_file_size_bytes=1024 * 1024)  # 1MB

    def test_validate_empty_file(self, validator, tmp_path):
        """Test validation rejects empty files."""
        empty_file = tmp_path / "empty.pdf"
        empty_file.write_bytes(b"")

        with pytest.raises(EmptyFileError) as exc_info:
            validator.validate(empty_file)

        assert exc_info.value.reason == "empty_file"
        assert "empty" in str(exc_info.value).lower()

    def test_validate_file_too_large(self, validator, tmp_path):
        """Test validation rejects oversized files."""
        large_file = tmp_path / "large.pdf"
        # Create file larger than 1MB limit
        large_file.write_bytes(b"x" * (1024 * 1024 + 100))

        with pytest.raises(FileTooLargeError) as exc_info:
            validator.validate(large_file)

        assert exc_info.value.reason == "file_too_large"
        assert "exceeds limit" in str(exc_info.value)

    def test_validate_pdf_corruption(self, validator, tmp_path):
        """Test validation detects corrupted PDFs."""
        corrupted_pdf = tmp_path / "corrupted.pdf"
        corrupted_pdf.write_bytes(b"This is not a valid PDF file")

        with patch("ingestion.preflight.PdfReader") as mock_reader:
            mock_reader.side_effect = PdfReadError("Invalid PDF structure")

            with pytest.raises(CorruptedFileError) as exc_info:
                validator._check_pdf_corruption(corrupted_pdf)

            assert exc_info.value.reason == "corrupted_file"
            assert "corrupted" in str(exc_info.value).lower()

    def test_validate_pdf_encrypted(self, validator, tmp_path):
        """Test validation detects password-protected PDFs."""
        encrypted_pdf = tmp_path / "encrypted.pdf"
        encrypted_pdf.write_bytes(b"%PDF-1.4 encrypted")

        with patch("ingestion.preflight.PdfReader") as mock_reader:
            mock_reader.side_effect = Exception("File is encrypted")

            with pytest.raises(EncryptedFileError) as exc_info:
                validator._check_pdf_corruption(encrypted_pdf)

            assert exc_info.value.reason == "encrypted_file"

    def test_validate_docx_corruption(self, validator, tmp_path):
        """Test validation detects corrupted DOCX files."""
        corrupted_docx = tmp_path / "corrupted.docx"
        corrupted_docx.write_bytes(b"Invalid DOCX content")

        with patch("ingestion.preflight.Document") as mock_doc:
            mock_doc.side_effect = PackageNotFoundError("Invalid package")

            with pytest.raises(CorruptedFileError) as exc_info:
                validator._check_docx_corruption(corrupted_docx)

            assert exc_info.value.reason == "corrupted_file"

    def test_validate_pdf_success(self, validator, tmp_path):
        """Test validation passes for valid PDF."""
        valid_pdf = tmp_path / "valid.pdf"
        valid_pdf.write_bytes(b"%PDF-1.4 valid content")

        with patch("ingestion.preflight.PdfReader") as mock_reader:
            mock_instance = MagicMock()
            mock_instance.pages = [MagicMock()]
            mock_reader.return_value = mock_instance

            # Should not raise
            result = validator._check_pdf_corruption(valid_pdf)
            assert result is None

    def test_validate_docx_success(self, validator, tmp_path):
        """Test validation passes for valid DOCX."""
        valid_docx = tmp_path / "valid.docx"
        valid_docx.write_bytes(b"PK\x03\x04 valid docx")

        with patch("ingestion.preflight.Document") as mock_doc:
            mock_instance = MagicMock()
            mock_instance.paragraphs = []
            mock_doc.return_value = mock_instance

            # Should not raise
            result = validator._check_docx_corruption(valid_docx)
            assert result is None


class TestEmptyFileCheck:
    """Test empty file detection."""

    def test_empty_file_rejected(self, tmp_path):
        """Test zero-byte files are rejected."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_bytes(b"")

        validator = PreflightValidator()

        with pytest.raises(EmptyFileError) as exc_info:
            validator._check_empty_file(empty_file)

        assert exc_info.value.reason == "empty_file"
        assert exc_info.value.file_path == str(empty_file)

    def test_nonempty_file_passes(self, tmp_path):
        """Test non-empty files pass check."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"Some content")

        validator = PreflightValidator()

        # Should not raise
        validator._check_empty_file(test_file)


class TestFileSizeCheck:
    """Test file size validation."""

    def test_oversized_file_rejected(self, tmp_path):
        """Test files exceeding limit are rejected."""
        large_file = tmp_path / "large.bin"
        large_file.write_bytes(b"x" * 10000)

        validator = PreflightValidator(max_file_size_bytes=1000)

        with pytest.raises(FileTooLargeError) as exc_info:
            validator._check_file_size(large_file)

        assert exc_info.value.reason == "file_too_large"
        assert "10000" in str(exc_info.value)
        assert "1000" in str(exc_info.value)

    def test_undersized_file_passes(self, tmp_path):
        """Test files under limit pass check."""
        small_file = tmp_path / "small.bin"
        small_file.write_bytes(b"x" * 100)

        validator = PreflightValidator(max_file_size_bytes=1000)

        # Should not raise
        validator._check_file_size(small_file)

    def test_default_file_size_limit(self):
        """Test default file size is 50MB."""
        validator = PreflightValidator()
        assert validator.max_file_size_bytes == 52_428_800


class TestValidateFilePreflight:
    """Test convenience function."""

    def test_validate_file_preflight_empty(self, tmp_path):
        """Test convenience function with empty file."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_bytes(b"")

        with pytest.raises(EmptyFileError):
            validate_file_preflight(empty_file)

    def test_validate_file_preflight_custom_size(self, tmp_path):
        """Test convenience function with custom size limit."""
        large_file = tmp_path / "large.bin"
        large_file.write_bytes(b"x" * 10000)

        with pytest.raises(FileTooLargeError):
            validate_file_preflight(large_file, max_file_size_bytes=1000)
