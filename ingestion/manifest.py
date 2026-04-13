"""Intake manifest generator for ingestion runs."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ManifestEntry:
    """Represents a single file entry in the ingestion manifest."""

    def __init__(
        self,
        file_path: str,
        document_id: str,
        file_size_bytes: int,
        detected_mime: str,
        status: str,  # 'queued', 'skipped', 'error'
        queued_at: Optional[str] = None,
        error_reason: Optional[str] = None,
    ):
        """
        Initialize a manifest entry.

        Args:
            file_path: Absolute path to the file
            document_id: Unique document identifier
            file_size_bytes: File size in bytes
            detected_mime: Detected MIME type
            status: Entry status (queued, skipped, error)
            queued_at: ISO timestamp when queued
            error_reason: Reason for error status
        """
        self.file_path = file_path
        self.document_id = document_id
        self.file_size_bytes = file_size_bytes
        self.detected_mime = detected_mime
        self.status = status
        self.queued_at = queued_at or datetime.now(timezone.utc).isoformat()
        self.error_reason = error_reason

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to dictionary."""
        data: dict[str, Any] = {
            "file_path": self.file_path,
            "document_id": self.document_id,
            "file_size_bytes": self.file_size_bytes,
            "detected_mime": self.detected_mime,
            "status": self.status,
            "queued_at": self.queued_at,
        }
        if self.error_reason:
            data["error_reason"] = self.error_reason
        return data


class IntakeManifest:
    """
    Generates and manages JSON manifests for ingestion runs.

    Each manifest records intake decisions before parsing begins,
    stored in a structured log directory.
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        manifest_dir: Optional[str | Path] = None,
    ):
        """
        Initialize the manifest generator.

        Args:
            run_id: Unique run identifier (UUID4 if not provided)
            manifest_dir: Directory to store manifests (default: logs/ingestion/manifests)
        """
        self.run_id = run_id or str(uuid.uuid4())
        self.manifest_dir = Path(manifest_dir or "logs/ingestion/manifests")
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.entries: list[ManifestEntry] = []
        self.created_at = datetime.now(timezone.utc).isoformat()

    def add_entry(
        self,
        file_path: str,
        document_id: str,
        file_size_bytes: int,
        detected_mime: str,
        status: str,
        error_reason: Optional[str] = None,
    ) -> ManifestEntry:
        """
        Add an entry to the manifest.

        Args:
            file_path: Absolute path to the file
            document_id: Unique document identifier
            file_size_bytes: File size in bytes
            detected_mime: Detected MIME type
            status: Entry status (queued, skipped, error)
            error_reason: Reason for error status

        Returns:
            The created ManifestEntry
        """
        entry = ManifestEntry(
            file_path=file_path,
            document_id=document_id,
            file_size_bytes=file_size_bytes,
            detected_mime=detected_mime,
            status=status,
            error_reason=error_reason,
        )
        self.entries.append(entry)
        return entry

    def add_queued_entry(
        self,
        file_path: str,
        document_id: str,
        file_size_bytes: int,
        detected_mime: str,
    ) -> ManifestEntry:
        """Add a file that is queued for ingestion."""
        return self.add_entry(
            file_path=file_path,
            document_id=document_id,
            file_size_bytes=file_size_bytes,
            detected_mime=detected_mime,
            status="queued",
        )

    def add_skipped_entry(
        self,
        file_path: str,
        document_id: str,
        file_size_bytes: int,
        detected_mime: str,
    ) -> ManifestEntry:
        """Add a file that was skipped (already ingested)."""
        return self.add_entry(
            file_path=file_path,
            document_id=document_id,
            file_size_bytes=file_size_bytes,
            detected_mime=detected_mime,
            status="skipped",
        )

    def add_error_entry(
        self,
        file_path: str,
        document_id: str,
        file_size_bytes: int,
        detected_mime: str,
        error_reason: str,
    ) -> ManifestEntry:
        """Add a file that encountered an error during intake."""
        return self.add_entry(
            file_path=file_path,
            document_id=document_id,
            file_size_bytes=file_size_bytes,
            detected_mime=detected_mime,
            status="error",
            error_reason=error_reason,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to dictionary."""
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "total_entries": len(self.entries),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def save(self) -> Path:
        """
        Save manifest to JSON file.

        Returns:
            Path to the saved manifest file
        """
        manifest_path = self.manifest_dir / f"{self.run_id}.json"

        manifest_data = self.to_dict()

        # Write atomically by writing to temp file first
        temp_path = manifest_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2, ensure_ascii=False)
            # Rename is atomic on most platforms
            temp_path.rename(manifest_path)
            logger.info(f"Manifest saved: {manifest_path}")
            return manifest_path
        except Exception as e:
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
            raise e

    def get_summary(self) -> dict[str, int]:
        """
        Get summary counts of manifest entries by status.

        Returns:
            Dictionary with counts per status
        """
        summary: dict[str, int] = {}
        for entry in self.entries:
            summary[entry.status] = summary.get(entry.status, 0) + 1
        return summary


def generate_intake_manifest(
    entries: list[dict[str, Any]],
    run_id: Optional[str] = None,
    manifest_dir: Optional[str | Path] = None,
) -> Path:
    """
    Convenience function to generate and save an intake manifest.

    Args:
        entries: List of entry dictionaries with keys:
            - file_path
            - document_id
            - file_size_bytes
            - detected_mime
            - status (queued, skipped, error)
            - error_reason (optional)
        run_id: Unique run identifier (UUID4 if not provided)
        manifest_dir: Directory to store manifests

    Returns:
        Path to the saved manifest file
    """
    manifest = IntakeManifest(run_id=run_id, manifest_dir=manifest_dir)

    for entry_data in entries:
        manifest.add_entry(**entry_data)

    return manifest.save()
