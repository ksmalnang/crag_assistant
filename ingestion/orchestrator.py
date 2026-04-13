"""Batch orchestrator for multi-document ingestion runs.

Implements async batch orchestration for processing multiple documents
concurrently with bounded concurrency and progress tracking.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from ingestion.alerter import check_and_alert
from ingestion.dead_letter import DeadLetterQueue
from ingestion.errors_base import IngestionError
from ingestion.formats import is_supported_extension
from ingestion.graph import compile_ingestion_graph
from ingestion.ledger import IngestionLedger
from ingestion.manifest import IntakeManifest
from ingestion.report import IngestionReport
from ingestion.state import IngestionState
from ingestion.types_ import BatchRunSummary, DocumentResult

logger = logging.getLogger(__name__)


class BatchOrchestrator:
    """
    Async batch orchestrator for multi-document ingestion.

    Uses asyncio.Semaphore for bounded concurrency.
    """

    def __init__(
        self,
        concurrency: int = 4,
        incremental: bool = True,
        force_full: bool = False,
    ):
        """
        Initialize the batch orchestrator.

        Args:
            concurrency: Max concurrent documents (default: 4).
            incremental: Whether to skip unchanged documents (default: True).
            force_full: Force re-ingestion of all files regardless of ledger.
        """
        self.concurrency = concurrency
        self.incremental = incremental
        self.force_full = force_full
        self.semaphore = asyncio.Semaphore(concurrency)
        self.ledger = IngestionLedger()
        self.graph = compile_ingestion_graph()

    def _collect_files(self, folder: str) -> list[str]:
        """
        Collect all supported files from a folder.

        Args:
            folder: Path to folder to scan.

        Returns:
            List of absolute file paths.
        """
        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        files = []
        for f in folder_path.iterdir():
            if f.is_file() and is_supported_extension(str(f)):
                files.append(str(f.resolve()))

        return sorted(files)

    def _should_skip(self, file_path: str) -> tuple[bool, str]:
        """
        Check if a file should be skipped based on ledger state.

        Args:
            file_path: Absolute file path.

        Returns:
            Tuple of (should_skip, skip_reason).
        """
        if self.force_full:
            return False, ""

        if not self.incremental:
            return False, ""

        entry = self.ledger.get_entry(file_path)
        if entry and entry.get("status") == "ingested":
            return True, "unchanged"

        return False, ""

    async def ingest_document(self, file_path: str, run_id: str) -> DocumentResult:
        """
        Ingest a single document through the LangGraph pipeline.

        Args:
            file_path: Absolute path to the file.
            run_id: Run ID for tracking.

        Returns:
            DocumentResult with processing outcome.
        """
        async with self.semaphore:
            result = DocumentResult(
                file_path=file_path,
                started_at=datetime.now(timezone.utc).isoformat(),
            )

            start_time = time.time()

            try:
                # Check if should skip
                should_skip, skip_reason = self._should_skip(file_path)
                if should_skip:
                    result.status = "skipped"
                    result.error_reason = f"skipped ({skip_reason})"
                    result.completed_at = datetime.now(timezone.utc).isoformat()
                    result.duration_seconds = time.time() - start_time
                    return result

                # Build initial state
                initial_state: IngestionState = {
                    "run_id": run_id,
                    "file_path": file_path,
                    "document_id": "",
                    "docling_doc": None,
                    "structure_tree": [],
                    "metadata": {},
                    "chunks": [],
                    "dense_vectors": [],
                    "sparse_vectors": [],
                    "upsert_count": 0,
                    "errors": [],
                    "status": "pending",
                }

                # Run through LangGraph
                final_state = await self.graph.ainvoke(initial_state)

                # Extract results
                result.document_id = final_state.get("document_id")
                result.status = final_state.get("status", "completed")
                result.chunks_created = len(final_state.get("chunks", []))
                result.vectors_upserted = final_state.get("upsert_count", 0)

                # Check for errors
                errors = final_state.get("errors", [])
                if errors:
                    last_error = errors[-1]
                    result.error_reason = last_error.get("message", "unknown")
                    result.error_node = last_error.get("node")
                    if result.status != "failed":
                        result.status = "failed"

                if result.status in ("completed", "processing"):
                    result.status = "success"

                    # Update ledger
                    if result.document_id:
                        file_bytes = Path(file_path).read_bytes()
                        content_hash = hashlib.sha256(file_bytes).hexdigest()
                        self.ledger.mark_ingested(
                            file_path=file_path,
                            document_id=result.document_id,
                            content_hash=content_hash,
                        )

            except Exception as e:
                result.status = "failed"
                result.error_reason = str(e)
                logger.error(f"[run:{run_id}] Failed to ingest {file_path}: {e}")

            result.completed_at = datetime.now(timezone.utc).isoformat()
            result.duration_seconds = time.time() - start_time

            return result

    async def run_batch(self, folder: str) -> BatchRunSummary:
        """
        Run batch ingestion on all files in a folder.

        Args:
            folder: Path to folder containing documents.

        Returns:
            BatchRunSummary with aggregated results.
        """
        run_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc).isoformat()
        start_time = time.time()

        logger.info(f"[run:{run_id}] Starting batch ingestion from {folder}")

        # Collect files
        files = self._collect_files(folder)
        total = len(files)

        if total == 0:
            logger.warning(f"No supported files found in {folder}")
            return BatchRunSummary(
                run_id=run_id,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc).isoformat(),
                total_files=0,
                duration_seconds=time.time() - start_time,
            )

        logger.info(f"[run:{run_id}] Found {total} files to process")

        # Create manifest
        manifest = IntakeManifest(run_id=run_id)

        # Queue all files in manifest
        for f in files:
            file_path_obj = Path(f)
            file_bytes = file_path_obj.read_bytes()
            doc_id = hashlib.sha256(file_bytes).hexdigest()
            manifest.add_queued_entry(
                file_path=f,
                document_id=doc_id,
                file_size_bytes=len(file_bytes),
                detected_mime="application/octet-stream",
            )

        # Run ingestion with progress bar
        results: list[DocumentResult] = []

        with tqdm(
            total=total,
            desc=f"Ingesting (concurrency={self.concurrency})",
            unit="file",
        ) as pbar:
            tasks = [
                asyncio.create_task(self.ingest_document(f, run_id)) for f in files
            ]

            for task in asyncio.as_completed(tasks):
                result = await task
                results.append(result)

                if result.status == "success":
                    pbar.set_postfix_str(
                        f"ok={sum(1 for r in results if r.status == 'success')}, "
                        f"err={sum(1 for r in results if r.status == 'failed')}"
                    )
                elif result.status == "skipped":
                    pbar.set_postfix_str(
                        f"skip={sum(1 for r in results if r.status == 'skipped')}"
                    )

                pbar.update(1)

        # Aggregate results
        summary = self._aggregate_results(
            run_id=run_id,
            started_at=started_at,
            results=results,
            duration=time.time() - start_time,
        )

        # Save manifest
        manifest.save()

        # Store failed documents in dead letter queue
        dlq = DeadLetterQueue()
        for result in results:
            if result.status == "failed" and result.error_reason:
                error = IngestionError(
                    document_id=result.document_id or "",
                    file_path=result.file_path,
                    node=result.error_node or "unknown",
                    reason=result.error_reason or "unknown",
                    message=result.error_reason or "Unknown error",
                )
                dlq.store(run_id=run_id, file_path=result.file_path, error=error)

        # Generate and save report
        report = IngestionReport(
            run_id=run_id,
            summary=summary,
        )
        report.save()

        # Check alert conditions and send if needed
        check_and_alert(
            run_id=run_id,
            total_files=summary.total_files,
            failed_count=summary.failed,
            errors=summary.errors,
        )

        # Print human-readable summary
        self._print_summary(summary)

        return summary

    def _aggregate_results(
        self,
        run_id: str,
        started_at: str,
        results: list[DocumentResult],
        duration: float,
    ) -> BatchRunSummary:
        """Aggregate individual document results into summary."""
        ingested = sum(1 for r in results if r.status == "success")
        skipped = sum(1 for r in results if r.status == "skipped")
        skipped_unchanged = sum(
            1
            for r in results
            if r.status == "skipped" and "unchanged" in (r.error_reason or "")
        )
        skipped_error = sum(
            1
            for r in results
            if r.status == "skipped" and "error" in (r.error_reason or "")
        )
        failed = sum(1 for r in results if r.status == "failed")

        errors = []
        for r in results:
            if r.status == "failed":
                errors.append(
                    {
                        "file": r.file_path,
                        "reason": r.error_reason or "unknown",
                        "node": r.error_node or "unknown",
                    }
                )

        return BatchRunSummary(
            run_id=run_id,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
            total_files=len(results),
            ingested=ingested,
            skipped=skipped,
            skipped_unchanged=skipped_unchanged,
            skipped_error=skipped_error,
            failed=failed,
            total_chunks_created=sum(r.chunks_created for r in results),
            total_vectors_upserted=sum(r.vectors_upserted for r in results),
            errors=errors,
            document_results=results,
            duration_seconds=duration,
        )

    @staticmethod
    def _print_summary(summary: BatchRunSummary) -> None:
        """Print human-readable summary to stdout."""
        print("\n" + "=" * 60)  # noqa: T201
        print("Ingestion Run Summary")  # noqa: T201
        print("=" * 60)  # noqa: T201
        print(f"  Run ID:         {summary.run_id}")  # noqa: T201
        print(f"  Duration:       {summary.duration_seconds:.1f}s")  # noqa: T201
        print(f"  Total files:    {summary.total_files}")  # noqa: T201
        print(f"  Ingested:       {summary.ingested}")  # noqa: T201
        print(f"  Skipped:        {summary.skipped}")  # noqa: T201
        if summary.skipped_unchanged > 0:
            print(f"    (unchanged:  {summary.skipped_unchanged})")  # noqa: T201
        if summary.skipped_error > 0:
            print(f"    (error:      {summary.skipped_error})")  # noqa: T201
        print(f"  Failed:         {summary.failed}")  # noqa: T201
        print(f"  Chunks:         {summary.total_chunks_created}")  # noqa: T201
        print(f"  Vectors:        {summary.total_vectors_upserted}")  # noqa: T201

        if summary.errors:
            print(f"\n  Errors ({len(summary.errors)}):")  # noqa: T201
            for err in summary.errors:
                print(  # noqa: T201
                    f"    - {err['file']}: {err['reason']} [{err['node']}]"
                )

        print("=" * 60)  # noqa: T201
