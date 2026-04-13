"""Supported file format contract and MIME type validation."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import magic

from .errors import UnsupportedFormatError


class SupportedFormat(str, Enum):
    """Enumeration of supported file formats with their MIME types and extensions."""

    PDF = "application/pdf"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    TXT = "text/plain"
    MD = "text/markdown"

    @property
    def extensions(self) -> list[str]:
        """Return file extensions associated with this format."""
        extension_map: dict[str, list[str]] = {
            SupportedFormat.PDF: [".pdf"],
            SupportedFormat.DOCX: [".docx"],
            SupportedFormat.PPTX: [".pptx"],
            SupportedFormat.TXT: [".txt"],
            SupportedFormat.MD: [".md", ".markdown"],
        }
        return extension_map.get(self, [])


# Configuration-driven supported formats - adding new formats only requires updating this dict
SUPPORTED_FORMATS: dict[str, str] = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "text/plain": ".txt",
    "text/markdown": ".md",
}


def get_supported_mime_types() -> list[str]:
    """Return list of supported MIME types from configuration."""
    return list(SUPPORTED_FORMATS.keys())


def get_supported_extensions() -> list[str]:
    """Return list of supported file extensions from configuration."""
    extensions: list[str] = []
    for ext in SUPPORTED_FORMATS.values():
        # Some entries might have multiple extensions
        extensions.append(ext)
    return extensions


def detect_mime_type(file_path: str | Path) -> str:
    """
    Detect MIME type from file bytes using python-magic, not just extension.

    Args:
        file_path: Path to the file

    Returns:
        Detected MIME type string

    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file can't be read
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read file bytes and detect MIME type
    with open(file_path, "rb") as f:
        # Read first 2048 bytes for MIME detection
        header = f.read(2048)
        mime = magic.Magic(mime=True)
        detected_mime = mime.from_buffer(header)

    return detected_mime


def validate_file_format(file_path: str | Path) -> str:
    """
    Validate that a file has a supported MIME type.

    Args:
        file_path: Path to the file

    Returns:
        Detected MIME type if valid

    Raises:
        UnsupportedFormatError: If file format is not supported
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)
    detected_mime = detect_mime_type(file_path)

    # Check if detected MIME type is in supported formats
    supported_mimes = get_supported_mime_types()

    # Handle text variants (e.g., text/plain; charset=utf-8)
    base_mime = detected_mime.split(";")[0].strip()

    if base_mime not in supported_mimes:
        raise UnsupportedFormatError(
            file_path=str(file_path),
            detected_mime=detected_mime,
        )

    return detected_mime


def is_supported_extension(file_path: str | Path) -> bool:
    """
    Quick check if file extension is supported (without MIME detection).

    This is a preliminary check before actual MIME validation.

    Args:
        file_path: Path to the file

    Returns:
        True if extension is supported
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    supported_exts = get_supported_extensions()
    return ext in supported_exts
