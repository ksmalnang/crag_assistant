"""CLI entry point for ingestion runs.

Usage:
    python -m ingestion.run --folder ./docs --concurrency 4
    python -m ingestion.run --folder ./docs --incremental
    python -m ingestion.run --folder ./docs --full
    python -m ingestion.run --folder ./docs --dry-run
    python -m ingestion.run --report-only
    python -m ingestion.run --report-only --run-id <uuid>
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from ingestion.orchestrator import BatchOrchestrator
from ingestion.report import IngestionReport


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the ingestion CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="ingestion.run",
        description="Campus AI Assistant – Document Ingestion CLI",
        epilog="Examples:\n"
        "  python -m ingestion.run --folder ./docs --concurrency 4\n"
        "  python -m ingestion.run --folder ./docs --incremental\n"
        "  python -m ingestion.run --folder ./docs --full\n"
        "  python -m ingestion.run --report-only\n"
        "  python -m ingestion.run --report-only --run-id <uuid>\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--folder",
        type=str,
        help="Path to folder containing documents to ingest",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max concurrent document ingestions (default: 4)",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        default=True,
        help="Enable incremental mode: skip unchanged documents (default: True)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        default=False,
        help="Force full re-ingestion of all documents regardless of ledger state",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Dry-run mode: skip embedding and upsert, generate chunk quality report",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        default=False,
        help="Print last run report without triggering ingestion",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Specific run ID to load report for (use with --report-only)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose/debug logging",
    )

    return parser


async def run_ingestion(args: argparse.Namespace) -> None:
    """Run the ingestion batch process."""
    if not args.folder:
        print("Error: --folder is required for ingestion runs.")  # noqa: T201
        sys.exit(1)

    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"Error: Folder not found: {args.folder}")  # noqa: T201
        sys.exit(1)

    orchestrator = BatchOrchestrator(
        concurrency=args.concurrency,
        incremental=args.incremental and not args.full,
        force_full=args.full,
        dry_run=args.dry_run,
    )

    summary = await orchestrator.run_batch(str(folder_path))

    if summary.failed > 0:
        sys.exit(1)


def run_report_only(args: argparse.Namespace) -> None:
    """Print report without running ingestion."""
    if args.run_id:
        IngestionReport.print_report(args.run_id)
    else:
        IngestionReport.print_last_report()


def main() -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.report_only:
        run_report_only(args)
    else:
        asyncio.run(run_ingestion(args))


if __name__ == "__main__":
    main()
