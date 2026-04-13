"""Chunk quality validation report for dry-run ingestion.

Generates a comprehensive report of chunk quality metrics
without performing expensive embedding or upsert operations.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from metadata.chunk_metadata import ChunkMetadata

logger = logging.getLogger(__name__)

# Threshold for suspiciously short chunks
SHORT_CHUNK_THRESHOLD = 20  # tokens


class ChunkQualityReport:
    """
    Generates chunk quality reports for dry-run validation.

    Metrics:
    - Total chunk count
    - Average/min/max token_count
    - Percentage of chunks with heading_path
    - Percentage of unclassified chunks
    - Faculty/doc_type distribution
    - Suspiciously short chunks flagged
    """

    REPORT_DIR = Path("logs/ingestion")

    def __init__(
        self,
        run_id: str,
        chunks: list[ChunkMetadata],
        metadata: Optional[dict[str, Any]] = None,
        report_dir: Optional[str | Path] = None,
    ):
        """
        Initialize the chunk quality report.

        Args:
            run_id: Unique run identifier.
            chunks: List of ChunkMetadata objects.
            metadata: Document metadata (faculty, doc_type, etc.).
            report_dir: Directory to store reports.
        """
        self.run_id = run_id
        self.chunks = chunks
        self.metadata = metadata or {}
        self.report_dir = Path(report_dir or self.REPORT_DIR)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.generated_at = datetime.now(timezone.utc).isoformat()

    def generate(self) -> dict[str, Any]:
        """Generate the chunk quality report."""
        if not self.chunks:
            return {
                "run_id": self.run_id,
                "generated_at": self.generated_at,
                "status": "no_chunks",
                "total_chunks": 0,
            }

        token_counts = [c.token_count for c in self.chunks]
        chunks_with_headings = sum(1 for c in self.chunks if c.heading_path)
        unclassified = sum(1 for c in self.chunks if c.unclassified)
        short_chunks = sum(
            1 for c in self.chunks if c.token_count < SHORT_CHUNK_THRESHOLD
        )

        total = len(self.chunks)

        report = {
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "status": "dry_run",
            "total_chunks": total,
            "token_stats": {
                "avg_token_count": round(sum(token_counts) / total, 2),
                "min_token_count": min(token_counts),
                "max_token_count": max(token_counts),
            },
            "heading_path_stats": {
                "chunks_with_headings": chunks_with_headings,
                "percentage": round(chunks_with_headings / total * 100, 1),
            },
            "unclassified_stats": {
                "unclassified_chunks": unclassified,
                "percentage": round(unclassified / total * 100, 1),
            },
            "short_chunks": {
                "count": short_chunks,
                "percentage": round(short_chunks / total * 100, 1),
                "threshold": SHORT_CHUNK_THRESHOLD,
                "chunks": [
                    {
                        "chunk_index": c.chunk_index,
                        "token_count": c.token_count,
                        "heading_path": c.heading_path,
                    }
                    for c in self.chunks
                    if c.token_count < SHORT_CHUNK_THRESHOLD
                ],
            },
            "metadata_distribution": {
                "faculty": self.metadata.get("faculty", "unknown"),
                "doc_type": self.metadata.get("doc_type", "unknown"),
                "semester": self.metadata.get("semester", "unknown"),
            },
        }

        return report

    def save(self) -> Path:
        """Save the report to JSON file."""
        report = self.generate()
        report_path = self.report_dir / f"dry_run_{self.run_id}.json"

        temp_path = report_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            temp_path.rename(report_path)
            logger.info(f"Dry-run report saved: {report_path}")
            return report_path
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e

    @classmethod
    def print_report(cls, report: dict[str, Any]) -> None:
        """Print a chunk quality report in human-readable format."""
        print("\n" + "=" * 60)  # noqa: T201
        print("Chunk Quality Report (Dry Run)")  # noqa: T201
        print("=" * 60)  # noqa: T201
        print(f"  Run ID:         {report.get('run_id', 'N/A')}")  # noqa: T201
        print(f"  Status:         {report.get('status', 'N/A')}")  # noqa: T201
        print(f"  Total chunks:   {report.get('total_chunks', 0)}")  # noqa: T201

        token_stats = report.get("token_stats", {})
        if token_stats:
            print("\n  Token Statistics:")  # noqa: T201
            print(f"    Average:      {token_stats.get('avg_token_count', 'N/A')}")  # noqa: T201
            print(f"    Min:          {token_stats.get('min_token_count', 'N/A')}")  # noqa: T201
            print(f"    Max:          {token_stats.get('max_token_count', 'N/A')}")  # noqa: T201

        heading_stats = report.get("heading_path_stats", {})
        if heading_stats:
            print("\n  Heading Path Coverage:")  # noqa: T201
            print(f"    With headings: {heading_stats.get('chunks_with_headings', 0)}")  # noqa: T201
            print(f"    Percentage:    {heading_stats.get('percentage', 0)}%")  # noqa: T201

        short_chunks = report.get("short_chunks", {})
        if short_chunks and short_chunks.get("count", 0) > 0:
            print(f"    Count: {short_chunks.get('count', 0)}")  # noqa: T201
            print(f"    Percentage: {short_chunks.get('percentage', 0)}%")  # noqa: T201
            for _chunk in short_chunks.get("chunks", [])[:5]:
                pass  # noqa: T201

        metadata = report.get("metadata_distribution", {})
        if metadata:
            print("\n  Metadata Distribution:")  # noqa: T201
            print(f"    Faculty:  {metadata.get('faculty', 'N/A')}")  # noqa: T201
            print(f"    Doc Type: {metadata.get('doc_type', 'N/A')}")  # noqa: T201
            print(f"    Semester: {metadata.get('semester', 'N/A')}")  # noqa: T201

        print("=" * 60)  # noqa: T201
