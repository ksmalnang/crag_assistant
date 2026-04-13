"""Typed error hierarchy for parsing validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ParseError(Exception):
    """Base error for document parsing failures."""

    file_path: str
    reason: str
    message: Optional[str] = None

    def __post_init__(self) -> None:
        if self.message is None:
            self.message = (
                f"Document parsing failed for {self.file_path}: {self.reason}"
            )
        super().__init__(self.message)


@dataclass
class ParseTimeoutError(ParseError):
    """Error raised when parsing exceeds timeout limit."""

    def __init__(
        self,
        file_path: str,
        timeout_seconds: int,
        message: Optional[str] = None,
    ) -> None:
        super().__init__(
            file_path=file_path,
            reason="parse_timeout",
            message=message
            or f"Parsing timed out after {timeout_seconds}s: {file_path}",
        )


@dataclass
class UnsupportedFormatParseError(ParseError):
    """Error raised when file format cannot be parsed."""

    def __init__(
        self,
        file_path: str,
        detected_mime: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        super().__init__(
            file_path=file_path,
            reason="unsupported_format",
            message=message,
        )
        self.detected_mime = detected_mime


@dataclass
class CorruptedDocumentError(ParseError):
    """Error raised when document is corrupted or unreadable."""

    def __init__(
        self,
        file_path: str,
        message: Optional[str] = None,
    ) -> None:
        super().__init__(
            file_path=file_path,
            reason="corrupted_document",
            message=message,
        )
