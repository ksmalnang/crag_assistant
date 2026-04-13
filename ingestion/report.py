"""Post-run ingestion report generator.

Generates a comprehensive JSON report after each ingestion run,
with human-readable summary output and CLI support.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ingestion.types_ import BatchRunSummary

logger = logging.getLogger(__name__)


class IngestionReport:
    """
    Generates and manages JSON reports for ingestion runs.

    Each report records ingestion outcomes after a run completes,
    stored in a structured log directory.
    """

    REPORT_DIR = Path("logs/ingestion/runs")

    def __init__(
        self,
        run_id: str,
        summary: Optional[BatchRunSummary] = None,
        report_dir: Optional[str | Path] = None,
    ):
        """
        Initialize the ingestion report.

        Args:
            run_id: Unique run identifier.
            summary: Batch run summary to report on.
            report_dir: Directory to store reports (default: logs/ingestion/runs).
        """
        self.run_id = run_id
        self.summary = summary
        self.report_dir = Path(report_dir or self.REPORT_DIR)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.generated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        if self.summary is None:
            return {
                "run_id": self.run_id,
                "generated_at": self.generated_at,
                "status": "no_summary",
            }

        return {
            "run_id": self.summary.run_id,
            "started_at": self.summary.started_at,
            "completed_at": self.summary.completed_at,
            "duration_seconds": round(self.summary.duration_seconds, 2),
            "total_files": self.summary.total_files,
            "ingested": self.summary.ingested,
            "skipped": self.summary.skipped,
            "skipped_unchanged": self.summary.skipped_unchanged,
            "skipped_error": self.summary.skipped_error,
            "failed": self.summary.failed,
            "total_chunks_created": self.summary.total_chunks_created,
            "total_vectors_upserted": self.summary.total_vectors_upserted,
            "errors": self.summary.errors,
            "generated_at": self.generated_at,
        }

    def save(self) -> Path:
        """
        Save report to JSON file.

        Returns:
            Path to the saved report file.
        """
        report_path = self.report_dir / f"{self.run_id}.json"

        report_data = self.to_dict()

        # Write atomically by writing to temp file first
        temp_path = report_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            temp_path.rename(report_path)
            logger.info(f"Ingestion report saved: {report_path}")
            return report_path
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e

    @classmethod
    def load_last_report(cls) -> Optional[dict[str, Any]]:
        """
        Load the most recent ingestion report.

        Returns:
            Report dictionary or None if no reports exist.
        """
        report_dir = cls.REPORT_DIR
        if not report_dir.exists():
            return None

        # Find most recent .json file
        json_files = sorted(
            report_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not json_files:
            return None

        last_report_path = json_files[0]
        try:
            with open(last_report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load report {last_report_path}: {e}")
            return None

    @classmethod
    def load_report(cls, run_id: str) -> Optional[dict[str, Any]]:
        """
        Load a specific ingestion report by run_id.

        Args:
            run_id: Run ID to load.

        Returns:
            Report dictionary or None if not found.
        """
        report_path = cls.REPORT_DIR / f"{run_id}.json"
        if not report_path.exists():
            return None

        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load report {report_path}: {e}")
            return None

    @classmethod
    def print_last_report(cls) -> None:
        """Print the last ingestion report in human-readable format."""
        report = cls.load_last_report()
        if report is None:
            print("No ingestion reports found.")  # noqa: T201
            return

        cls._print_report(report)

    @classmethod
    def print_report(cls, run_id: str) -> None:
        """Print a specific ingestion report in human-readable format."""
        report = cls.load_report(run_id)
        if report is None:
            print(f"Report not found for run_id: {run_id}")  # noqa: T201
            return

        cls._print_report(report)

    @staticmethod
    def _print_report(report: dict[str, Any]) -> None:
        """Print a report dictionary in human-readable format."""
        print("\n" + "=" * 60)  # noqa: T201
        print("Ingestion Report")  # noqa: T201
        print("=" * 60)  # noqa: T201
        print(f"  Run ID:         {report.get('run_id', 'N/A')}")  # noqa: T201
        print(f"  Started:        {report.get('started_at', 'N/A')}")  # noqa: T201
        print(f"  Completed:      {report.get('completed_at', 'N/A')}")  # noqa: T201
        print(f"  Duration:       {report.get('duration_seconds', 'N/A')}s")  # noqa: T201
        print(f"  Total files:    {report.get('total_files', 0)}")  # noqa: T201
        print(f"  Ingested:       {report.get('ingested', 0)}")  # noqa: T201
        print(f"  Skipped:        {report.get('skipped', 0)}")  # noqa: T201
        if report.get("skipped_unchanged", 0) > 0:
            print(f"    (unchanged:  {report.get('skipped_unchanged', 0)})")  # noqa: T201
        if report.get("skipped_error", 0) > 0:
            print(f"    (error:      {report.get('skipped_error', 0)})")  # noqa: T201
        print(f"  Failed:         {report.get('failed', 0)}")  # noqa: T201
        print(f"  Chunks:         {report.get('total_chunks_created', 0)}")  # noqa: T201
        print(f"  Vectors:        {report.get('total_vectors_upserted', 0)}")  # noqa: T201

        errors = report.get("errors", [])
        if errors:
            print(f"\n  Errors ({len(errors)}):")  # noqa: T201
            for err in errors:
                print(  # noqa: T201
                    f"    - {err.get('file', 'unknown')}: "
                    f"{err.get('reason', 'unknown')} "
                    f"[{err.get('node', 'unknown')}]"
                )

        print("=" * 60)  # noqa: T201
