"""Typed error hierarchy for vector store operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StoreError(Exception):
    """Base error for vector store operations."""

    reason: str
    collection_name: Optional[str] = None
    message: Optional[str] = field(default=None, repr=False)

    def __init__(
        self,
        reason: str,
        collection_name: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        self.reason = reason
        self.collection_name = collection_name
        self.message = message or f"Vector store error: {self.reason}"
        super().__init__(self.message)


@dataclass
class SchemaError(StoreError):
    """Error raised when collection schema does not match expected config."""

    expected: Optional[dict] = None
    actual: Optional[dict] = None

    def __init__(
        self,
        reason: str = "schema_mismatch",
        collection_name: Optional[str] = None,
        message: Optional[str] = None,
        expected: Optional[dict] = None,
        actual: Optional[dict] = None,
    ) -> None:
        self.expected = expected
        self.actual = actual
        if message is None:
            message = (
                f"Schema mismatch for collection '{collection_name}': "
                f"expected={expected}, actual={actual}"
            )
        super().__init__(
            reason=reason,
            collection_name=collection_name,
            message=message,
        )


@dataclass
class UpsertError(StoreError):
    """Error raised when upsert fails after retries."""

    batch_index: int = 0
    retry_count: int = 0

    def __init__(
        self,
        reason: str = "upsert_failed",
        collection_name: Optional[str] = None,
        message: Optional[str] = None,
        batch_index: int = 0,
        retry_count: int = 0,
    ) -> None:
        self.batch_index = batch_index
        self.retry_count = retry_count
        if message is None:
            message = (
                f"Upsert failed for batch {batch_index} "
                f"after {retry_count} retries: {reason}"
            )
        super().__init__(
            reason=reason,
            collection_name=collection_name,
            message=message,
        )


@dataclass
class DeletionError(StoreError):
    """Error raised when stale chunk deletion fails."""

    document_id: Optional[str] = None

    def __init__(
        self,
        reason: str = "deletion_failed",
        collection_name: Optional[str] = None,
        message: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> None:
        self.document_id = document_id
        if message is None:
            message = (
                f"Failed to delete stale chunks for document '{document_id}' "
                f"in collection '{collection_name}': {reason}"
            )
        super().__init__(
            reason=reason,
            collection_name=collection_name,
            message=message,
        )


@dataclass
class HealthCheckError(StoreError):
    """Error raised when post-upsert health check fails."""

    check_name: Optional[str] = None
    details: Optional[dict] = None

    def __init__(
        self,
        reason: str = "health_check_failed",
        collection_name: Optional[str] = None,
        message: Optional[str] = None,
        check_name: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        self.check_name = check_name
        self.details = details
        if message is None:
            message = (
                f"Health check '{check_name}' failed for "
                f"collection '{collection_name}': {reason}"
            )
        super().__init__(
            reason=reason,
            collection_name=collection_name,
            message=message,
        )
