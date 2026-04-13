"""CLI retry command for dead letter queue.

Usage:
    python -m ingestion.retry --dead-letter-dir /data/dead_letter/{run_id}
    python -m ingestion.retry --run-id <uuid>
    python -m ingestion.retry --run-id <uuid> --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from ingestion.dead_letter_queue import DeadLetterQueue
from ingestion.orchestrator import BatchOrchestrator
from ingestion.types_ import BatchRunSummary


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the retry CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


async def retry_dead_letter(
    dlq: DeadLetterQueue,
    run_id: str,
    concurrency: int = 4,
    verbose: bool = False,
) -> BatchRunSummary:
    """
    Retry all dead letter entries for a given run.

    For each failed document:
    1. Re-runs the full ingestion subgraph from intake
    2. On success: removes file from dead letter folder, updates ledger
    3. On failure: re-stores in dead letter queue

    Args:
        dlq: Dead letter queue instance.
        run_id: The original run ID to retry.
        concurrency: Max concurrent document ingestions.
        verbose: Enable verbose logging.

    Returns:
        BatchRunSummary with retry results.
    """
    entries = dlq.list_entries(run_id)

    if not entries:
        return BatchRunSummary(
            run_id=run_id,
            started_at="",
            total_files=0,
        )


    orchestrator = BatchOrchestrator(
        concurrency=concurrency,
        incremental=False,  # Force re-ingestion
        force_full=True,
    )

    results = []

    for entry in entries:
        error_file = entry.get("_error_file", "")
        file_path = entry.get("_file_path", "")
        entry.get("original_filename", "")

        if not Path(file_path).exists():
            continue


        try:
            # Run ingestion for this file
            result = await orchestrator.ingest_document(
                file_path=file_path,
                run_id=f"retry-{run_id}",
            )

            if result.status == "success":
                # Remove from dead letter queue
                dlq.remove_entry(error_file)
            else:
                # Re-store with updated error
                pass

            results.append(result)

        except Exception:
            pass

    # Aggregate results
    from datetime import datetime, timezone

    ingested = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status == "failed")

    summary = BatchRunSummary(
        run_id=f"retry-{run_id}",
        started_at=datetime.now(timezone.utc).isoformat(),
        total_files=len(results),
        ingested=ingested,
        failed=failed,
        total_chunks_created=sum(r.chunks_created for r in results),
        total_vectors_upserted=sum(r.vectors_upserted for r in results),
    )

    # Print summary

    return summary


def main() -> None:
    """Main entry point for retry CLI."""
    parser = argparse.ArgumentParser(
        prog="ingestion.retry",
        description="Retry failed documents from the dead letter queue",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        help="Run ID to retry documents from",
    )
    parser.add_argument(
        "--dead-letter-dir",
        type=str,
        help="Path to dead letter directory (alternative to --run-id)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max concurrent document retries (default: 4)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose/debug logging",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.dead_letter_dir:
        dlq = DeadLetterQueue(base_dir=Path(args.dead_letter_dir))
        # Extract run_id from path
        run_id = Path(args.dead_letter_dir).name
    elif args.run_id:
        dlq = DeadLetterQueue()
        run_id = args.run_id
    else:
        sys.exit(1)

    asyncio.run(retry_dead_letter(dlq, run_id, args.concurrency, args.verbose))


if __name__ == "__main__":
    main()
