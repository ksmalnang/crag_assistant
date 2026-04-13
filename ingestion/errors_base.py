"""Unified typed error hierarchy for the ingestion pipeline.

All ingestion errors inherit from IngestionError, which provides:
- document_id, file_path, node, reason, details: dict
- to_dict() for JSON serialisation (dead letter queue, reports)
- from_dict() for deserialisation (retry from dead letter queue)

The existing per-module error hierarchies (ingestion/errors.py,
parsing/errors.py, embedding/errors.py, vector_store/errors.py) are
preserved for backward compatibility.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from typing import Any

# ─── Base Error ───────────────────────────────────────────────────────────────


@dataclass
class IngestionError(Exception):
    """
    Base class for all ingestion pipeline errors.

    Attributes:
        document_id: Unique document identifier (may be empty if unknown).
        file_path: Path to the file being processed.
        node: Name of the ingestion node that raised the error.
        reason: Machine-readable error code.
        details: Additional context as key-value pairs.
        message: Human-readable error message (auto-generated if None).
        traceback_str: Full traceback string for debugging.
    """

    document_id: str = ""
    file_path: str = ""
    node: str = "unknown"
    reason: str = "unknown"
    details: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    traceback_str: str = ""

    def __post_init__(self) -> None:
        if not self.message:
            self.message = (
                f"Ingestion error in {self.node} for {self.file_path}: {self.reason}"
            )
        if not self.traceback_str:
            self.traceback_str = traceback.format_exc().strip()
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Serialise error to dictionary for JSON storage."""
        return {
            "document_id": self.document_id,
            "file_path": self.file_path,
            "node": self.node,
            "reason": self.reason,
            "details": self.details,
            "message": self.message,
            "traceback_str": self.traceback_str,
            "error_type": self.__class__.__name__,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise error to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IngestionError":
        """Deserialise error from dictionary."""
        error_type = data.pop("error_type", "IngestionError")
        error_cls = _ERROR_REGISTRY.get(error_type, cls)
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in data.items() if k in allowed}
        return error_cls(**kwargs)


# Registry for error subclass deserialisation
_ERROR_REGISTRY: dict[str, type[IngestionError]] = {}


def register_error_class(cls: type[IngestionError]) -> type[IngestionError]:
    """Decorator to register an error class for from_dict() deserialisation."""
    _ERROR_REGISTRY[cls.__name__] = cls
    return cls


# ─── Pipeline-Level Error Subclasses ─────────────────────────────────────────


@register_error_class
@dataclass
class IntakeNodeError(IngestionError):
    """Error raised during intake/validation stage."""

    node: str = field(default="intake_node", init=False)


@register_error_class
@dataclass
class ParseNodeError(IngestionError):
    """Error raised during document parsing stage."""

    node: str = field(default="parser_node", init=False)


@register_error_class
@dataclass
class MetadataError(IngestionError):
    """Error raised during metadata resolution stage."""

    node: str = field(default="metadata_resolver_node", init=False)


@register_error_class
@dataclass
class ChunkError(IngestionError):
    """Error raised during document chunking stage."""

    node: str = field(default="chunker_node", init=False)


@register_error_class
@dataclass
class EmbeddingNodeError(IngestionError):
    """Error raised during embedding generation stage."""

    node: str = field(default="embedding_node", init=False)
    chunk_id: str = ""


@register_error_class
@dataclass
class UpsertNodeError(IngestionError):
    """Error raised during vector store upsert stage."""

    node: str = field(default="upsert_node", init=False)
    batch_index: int = 0
    retry_count: int = 0

    def __init__(
        self,
        document_id: str = "",
        file_path: str = "",
        reason: str = "unknown",
        details: dict[str, Any] = None,
        message: str = "",
        traceback_str: str = "",
        batch_index: int = 0,
        retry_count: int = 0,
        **_kwargs: Any,
    ) -> None:
        self.batch_index = batch_index
        self.retry_count = retry_count
        super().__init__(
            document_id=document_id,
            file_path=file_path,
            reason=reason,
            details=details or {},
            message=message,
            traceback_str=traceback_str,
        )
        # Set node after super().__init__() to override the default
        self.node = "upsert_node"


@register_error_class
@dataclass
class SchemaNodeError(IngestionError):
    """Error raised when Qdrant collection schema mismatches expected config."""

    node: str = field(default="schema_node", init=False)
    expected: dict[str, Any] = field(default_factory=dict)
    actual: dict[str, Any] = field(default_factory=dict)


@register_error_class
@dataclass
class HealthCheckNodeError(IngestionError):
    """Error raised when post-upsert health check fails."""

    node: str = field(default="health_check_node", init=False)
    check_name: str = ""


# ─── Helper Functions ─────────────────────────────────────────────────────────


def error_from_exception(
    exc: Exception,
    node: str,
    file_path: str,
    document_id: str = "",
) -> IngestionError:
    """
    Convert any exception to an IngestionError.

    If the exception is already an IngestionError, it is returned unchanged.

    Args:
        exc: The exception to convert.
        node: Name of the ingestion node.
        file_path: Path to the file being processed.
        document_id: Document identifier.

    Returns:
        An IngestionError instance.
    """
    if isinstance(exc, IngestionError):
        return exc

    return IngestionError(
        document_id=document_id,
        file_path=file_path,
        node=node,
        reason=type(exc).__name__,
        details={"original_error": str(exc), "original_type": type(exc).__name__},
        message=str(exc),
    )


def serialise_errors(errors: list[Exception]) -> list[dict[str, Any]]:
    """Serialise a list of errors to dictionaries."""
    result = []
    for err in errors:
        if isinstance(err, IngestionError):
            result.append(err.to_dict())
        else:
            result.append(
                {
                    "error_type": type(err).__name__,
                    "message": str(err),
                    "traceback_str": "",
                }
            )
    return result
