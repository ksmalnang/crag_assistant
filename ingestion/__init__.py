"""Ingestion package."""

from .errors import (
    CorruptedFileError,
    EmptyFileError,
    EncryptedFileError,
    FileTooLargeError,
    IntakeError,
    UnsupportedFormatError,
)
from .formats import (
    SupportedFormat,
    detect_mime_type,
    get_supported_extensions,
    get_supported_mime_types,
    is_supported_extension,
    validate_file_format,
)
from .ledger import IngestionLedger
from .manifest import (
    IntakeManifest,
    ManifestEntry,
    generate_intake_manifest,
)
from .preflight import PreflightValidator, validate_file_preflight
from .watcher import FileChangeEvent, FolderWatcher, compute_file_hash

__all__ = [
    # Errors
    "IntakeError",
    "UnsupportedFormatError",
    "FileTooLargeError",
    "EmptyFileError",
    "CorruptedFileError",
    "EncryptedFileError",
    # Formats
    "SupportedFormat",
    "SUPPORTED_FORMATS",
    "detect_mime_type",
    "validate_file_format",
    "is_supported_extension",
    "get_supported_mime_types",
    "get_supported_extensions",
    # Ledger
    "IngestionLedger",
    # Manifest
    "IntakeManifest",
    "ManifestEntry",
    "generate_intake_manifest",
    # Preflight
    "PreflightValidator",
    "validate_file_preflight",
    # Watcher
    "FileChangeEvent",
    "FolderWatcher",
    "compute_file_hash",
]
