"""Ingestion package."""

from .alerter import SlackAlerter, check_and_alert
from .dead_letter import DeadLetterQueue
from .errors import (
    CorruptedFileError,
    EmptyFileError,
    EncryptedFileError,
    FileTooLargeError,
    IntakeError,
    UnsupportedFormatError,
)
from .errors_base import (
    ChunkError,
    EmbeddingNodeError,
    HealthCheckNodeError,
    IngestionError,
    IntakeNodeError,
    MetadataError,
    ParseNodeError,
    SchemaNodeError,
    UpsertNodeError,
    error_from_exception,
    serialise_errors,
)
from .formats import (
    SupportedFormat,
    detect_mime_type,
    get_supported_extensions,
    get_supported_mime_types,
    is_supported_extension,
    validate_file_format,
)
from .graph import build_ingestion_graph, compile_ingestion_graph, ingestion_graph
from .ledger import IngestionLedger
from .manifest import (
    IntakeManifest,
    ManifestEntry,
    generate_intake_manifest,
)
from .nodes import (
    chunker_node,
    embedding_node,
    health_check_node,
    intake_node,
    metadata_resolver_node,
    parser_node,
    upsert_node,
)
from .orchestrator import BatchOrchestrator
from .preflight import PreflightValidator, validate_file_preflight
from .report import IngestionReport
from .state import ErrorEntry, IngestionState
from .types_ import BatchRunSummary, DocumentResult
from .watcher import FileChangeEvent, FolderWatcher, compute_file_hash

__all__ = [
    # Alerter
    "SlackAlerter",
    "check_and_alert",
    # Dead Letter Queue
    "DeadLetterQueue",
    # Errors (unified hierarchy - errors_base.py)
    "IngestionError",
    "IntakeNodeError",
    "ParseNodeError",
    "MetadataError",
    "ChunkError",
    "EmbeddingNodeError",
    "UpsertNodeError",
    "SchemaNodeError",
    "HealthCheckNodeError",
    "error_from_exception",
    "serialise_errors",
    # Errors (legacy intake errors)
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
    # Graph
    "build_ingestion_graph",
    "compile_ingestion_graph",
    "ingestion_graph",
    # Ledger
    "IngestionLedger",
    # Manifest
    "IntakeManifest",
    "ManifestEntry",
    "generate_intake_manifest",
    # Nodes
    "intake_node",
    "parser_node",
    "metadata_resolver_node",
    "chunker_node",
    "embedding_node",
    "upsert_node",
    "health_check_node",
    # Orchestrator
    "BatchOrchestrator",
    "BatchRunSummary",
    "DocumentResult",
    # Preflight
    "PreflightValidator",
    "validate_file_preflight",
    # Report
    "IngestionReport",
    # State
    "IngestionState",
    "ErrorEntry",
    # Types
    "BatchRunSummary",
    "DocumentResult",
    # Watcher
    "FileChangeEvent",
    "FolderWatcher",
    "compute_file_hash",
]
