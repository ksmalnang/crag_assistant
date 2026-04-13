"""Dead letter queue for failed ingestion documents.

Stores failed documents with error metadata in /data/dead_letter/{run_id}/,
supporting CLI-based retry operations.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ingestion.errors_base import IngestionError

logger = logging.getLogger(__name__)


class DeadLetterQueue:
    """
    Dead letter queue for failed ingestion documents.

    When a document fails during ingestion, it is copied to the dead letter
    directory alongside an error metadata JSON file. This enables operational
    recovery via the retry CLI command.

    Structure:
        data/dead_letter/{run_id}/
        ├── {original_filename}          # Copy of the failed file
        └── {original_filename}.error.json  # Error metadata
    """

    DEFAULT_DIR = Path("data/dead_letter")

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the dead letter queue.

        Args:
            base_dir: Base directory for dead letter storage.
                      Defaults to data/dead_letter.
        """
        self.base_dir = base_dir or self.DEFAULT_DIR

    def _run_dir(self, run_id: str) -> Path:
        """Get the run-specific dead letter directory."""
        return self.base_dir / run_id

    def store(
        self,
        run_id: str,
        file_path: str,
        error: IngestionError,
    ) -> Path:
        """
        Store a failed document in the dead letter queue.

        Copies the original file and writes an accompanying error JSON.

        Args:
            run_id: The ingestion run ID.
            file_path: Path to the failed file.
            error: The IngestionError that caused the failure.

        Returns:
            Path to the error JSON file.
        """
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        original = Path(file_path)
        filename = original.name

        # Copy the failed file
        dest_file = run_dir / filename
        shutil.copy2(original, dest_file)

        # Write error metadata
        error_data = error.to_dict()
        error_data["failed_at"] = datetime.now(timezone.utc).isoformat()
        error_data["original_filename"] = filename

        error_path = run_dir / f"{filename}.error.json"
        error_path.write_text(
            json.dumps(error_data, indent=2, default=str),
            encoding="utf-8",
        )

        logger.info(
            f"Dead letter stored: {error_path} "
            f"(file: {dest_file}, error: {error.reason})"
        )
        return error_path

    def list_entries(self, run_id: Optional[str] = None) -> list[dict[str, Any]]:
        """
        List all dead letter entries.

        Args:
            run_id: Filter by specific run ID. If None, lists all runs.

        Returns:
            List of error metadata dictionaries.
        """
        entries = []

        if run_id:
            dirs = [self._run_dir(run_id)] if self._run_dir(run_id).exists() else []
        else:
            dirs = sorted(self.base_dir.iterdir()) if self.base_dir.exists() else []

        for run_dir in dirs:
            if not run_dir.is_dir():
                continue
            for error_file in run_dir.glob("*.error.json"):
                try:
                    data = json.loads(error_file.read_text(encoding="utf-8"))
                    data["_error_file"] = str(error_file)
                    data["_file_path"] = str(error_file.with_suffix(""))
                    entries.append(data)
                except Exception as e:
                    logger.warning(f"Failed to read dead letter {error_file}: {e}")

        return entries

    def get_entry(self, error_file_path: str) -> Optional[dict[str, Any]]:
        """
        Get a specific dead letter entry by error file path.

        Args:
            error_file_path: Path to the .error.json file.

        Returns:
            Error metadata dictionary or None if not found.
        """
        path = Path(error_file_path)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["_error_file"] = str(path)
            data["_file_path"] = str(path.with_suffix(""))
            return data
        except Exception as e:
            logger.warning(f"Failed to read dead letter {path}: {e}")
            return None

    def remove_entry(self, error_file_path: str) -> bool:
        """
        Remove a dead letter entry (both file and error JSON).

        Args:
            error_file_path: Path to the .error.json file.

        Returns:
            True if successfully removed.
        """
        path = Path(error_file_path)
        file_path = path.with_suffix("")  # Remove .error.json

        removed = False
        if file_path.exists():
            file_path.unlink()
            removed = True
        if path.exists():
            path.unlink()
            removed = True

        return removed

    def remove_run(self, run_id: str) -> bool:
        """
        Remove all dead letter entries for a run.

        Args:
            run_id: The ingestion run ID.

        Returns:
            True if the run directory was removed.
        """
        run_dir = self._run_dir(run_id)
        if run_dir.exists():
            shutil.rmtree(run_dir)
            return True
        return False
