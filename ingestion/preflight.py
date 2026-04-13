"""Pre-flight validation checks for file ingestion."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .errors import (
    CorruptedFileError,
    EmptyFileError,
    EncryptedFileError,
    FileTooLargeError,
)
from .formats import detect_mime_type

logger = logging.getLogger(__name__)


class PreflightValidator:
    """
    Validates files before they enter the parsing pipeline.

    Checks:
    - File size limits
    - Zero-byte files
    - PDF corruption
    - DOCX corruption
    - Password-protected/encrypted files
    """

    def __init__(self, max_file_size_bytes: Optional[int] = None):
        """
        Initialize the preflight validator.

        Args:
            max_file_size_bytes: Maximum allowed file size in bytes (default: 50MB)
        """
        self.max_file_size_bytes = max_file_size_bytes or 52_428_800  # 50MB

    def validate(self, file_path: str | Path) -> bool:
        """
        Run all pre-flight validation checks on a file.

        Args:
            file_path: Path to the file

        Returns:
            True if all checks pass

        Raises:
            EmptyFileError: If file is zero bytes
            FileTooLargeError: If file exceeds size limit
            CorruptedFileError: If file is corrupted
            EncryptedFileError: If file is password-protected
        """
        file_path = Path(file_path)

        # Check 1: Zero-byte files
        self._check_empty_file(file_path)

        # Check 2: File size limit
        self._check_file_size(file_path)

        # Check 3: Format-specific corruption checks
        mime_type = detect_mime_type(file_path)
        base_mime = mime_type.split(";")[0].strip()

        if base_mime == "application/pdf":
            self._check_pdf_corruption(file_path)
        elif (
            base_mime
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ):
            self._check_docx_corruption(file_path)

        logger.info(f"Pre-flight validation passed: {file_path}")
        return True

    def _check_empty_file(self, file_path: Path) -> None:
        """
        Check if file is empty (zero bytes).

        Args:
            file_path: Path to the file

        Raises:
            EmptyFileError: If file size is 0 bytes
        """
        file_size = file_path.stat().st_size
        if file_size == 0:
            raise EmptyFileError(
                file_path=str(file_path),
                message=f"File is empty (0 bytes): {file_path}",
            )

    def _check_file_size(self, file_path: Path) -> None:
        """
        Check if file exceeds maximum allowed size.

        Args:
            file_path: Path to the file

        Raises:
            FileTooLargeError: If file size exceeds limit
        """
        file_size = file_path.stat().st_size
        if file_size > self.max_file_size_bytes:
            raise FileTooLargeError(
                file_path=str(file_path),
                message=f"File size ({file_size} bytes) exceeds limit ({self.max_file_size_bytes} bytes)",
            )

    def _check_pdf_corruption(self, file_path: Path) -> None:
        """
        Check PDF file integrity by attempting to open it.

        Args:
            file_path: Path to the PDF file

        Raises:
            EncryptedFileError: If PDF is password-protected
            CorruptedFileError: If PDF is corrupted
        """
        try:
            reader = PdfReader(str(file_path))
            # Try to access the first page to ensure it's readable
            if len(reader.pages) > 0:
                _ = reader.pages[0]
        except Exception as e:
            # Check if it's an encrypted/password-protected file
            error_msg = str(e).lower()
            if "encrypt" in error_msg or "password" in error_msg:
                raise EncryptedFileError(
                    file_path=str(file_path),
                    detected_mime="application/pdf",
                    message=f"PDF is password-protected: {file_path}",
                )
            # Otherwise, it's a corruption issue
            raise CorruptedFileError(
                file_path=str(file_path),
                detected_mime="application/pdf",
                message=f"PDF file is corrupted or unreadable: {file_path} - {e}",
            )

    def _check_docx_corruption(self, file_path: Path) -> None:
        """
        Check DOCX file integrity by attempting to open it.

        Args:
            file_path: Path to the DOCX file

        Raises:
            CorruptedFileError: If DOCX is corrupted
        """
        try:
            doc = Document(str(file_path))
            # Try to access some basic properties to ensure it's valid
            _ = doc.paragraphs
        except PackageNotFoundError as e:
            raise CorruptedFileError(
                file_path=str(file_path),
                detected_mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                message=f"DOCX file is corrupted or unreadable: {file_path} - {e}",
            )
        except Exception as e:
            raise CorruptedFileError(
                file_path=str(file_path),
                detected_mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                message=f"DOCX file validation failed: {file_path} - {e}",
            )


def validate_file_preflight(
    file_path: str | Path,
    max_file_size_bytes: Optional[int] = None,
) -> bool:
    """
    Convenience function to run pre-flight validation.

    Args:
        file_path: Path to the file
        max_file_size_bytes: Maximum allowed file size (optional)

    Returns:
        True if validation passes

    Raises:
        IntakeError subclass if validation fails
    """
    validator = PreflightValidator(max_file_size_bytes=max_file_size_bytes)
    return validator.validate(file_path)
