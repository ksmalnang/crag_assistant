"""Typed error hierarchy for intake validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class IntakeError(Exception):
    """Base error for intake validation failures."""

    file_path: str
    reason: str
    detected_mime: Optional[str] = None
    message: Optional[str] = None

    def __post_init__(self) -> None:
        if self.message is None:
            self.message = (
                f"Intake validation failed for {self.file_path}: {self.reason}"
            )
        super().__init__(self.message)


@dataclass
class UnsupportedFormatError(IntakeError):
    """Error raised when file format is not supported."""

    def __init__(
        self,
        file_path: str,
        detected_mime: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        super().__init__(
            file_path=file_path,
            reason="unsupported_format",
            detected_mime=detected_mime,
            message=message,
        )


@dataclass
class FileTooLargeError(IntakeError):
    """Error raised when file exceeds size limit."""

    def __init__(
        self,
        file_path: str,
        detected_mime: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        super().__init__(
            file_path=file_path,
            reason="file_too_large",
            detected_mime=detected_mime,
            message=message,
        )


@dataclass
class EmptyFileError(IntakeError):
    """Error raised when file is empty (zero bytes)."""

    def __init__(
        self,
        file_path: str,
        detected_mime: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        super().__init__(
            file_path=file_path,
            reason="empty_file",
            detected_mime=detected_mime,
            message=message,
        )


@dataclass
class CorruptedFileError(IntakeError):
    """Error raised when file is corrupted or unreadable."""

    def __init__(
        self,
        file_path: str,
        detected_mime: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        super().__init__(
            file_path=file_path,
            reason="corrupted_file",
            detected_mime=detected_mime,
            message=message,
        )


@dataclass
class EncryptedFileError(IntakeError):
    """Error raised when file is password-protected or encrypted."""

    def __init__(
        self,
        file_path: str,
        detected_mime: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        super().__init__(
            file_path=file_path,
            reason="encrypted_file",
            detected_mime=detected_mime,
            message=message,
        )
