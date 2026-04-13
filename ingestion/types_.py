"""Shared types for ingestion orchestration.

Extracted to avoid circular imports between orchestrator and report modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentResult:
    """Result of processing a single document."""

    file_path: str
    document_id: str | None = None
    status: str = "pending"  # pending, processing, success, skipped, failed
    chunks_created: int = 0
    vectors_upserted: int = 0
    error_reason: str | None = None
    error_node: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float = 0.0


@dataclass
class BatchRunSummary:
    """Aggregated results from a batch ingestion run."""

    run_id: str
    started_at: str
    completed_at: str | None = None
    total_files: int = 0
    ingested: int = 0
    skipped: int = 0
    skipped_unchanged: int = 0
    skipped_error: int = 0
    failed: int = 0
    total_chunks_created: int = 0
    total_vectors_upserted: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)
    document_results: list[DocumentResult] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "total_files": self.total_files,
            "ingested": self.ingested,
            "skipped": self.skipped,
            "skipped_unchanged": self.skipped_unchanged,
            "skipped_error": self.skipped_error,
            "failed": self.failed,
            "total_chunks_created": self.total_chunks_created,
            "total_vectors_upserted": self.total_vectors_upserted,
            "errors": self.errors,
        }
