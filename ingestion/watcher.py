"""Local folder watcher for detecting file changes via content hashing."""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .ledger import IngestionLedger

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: str | Path) -> str:
    """
    Compute SHA256 hash of file contents.

    Args:
        file_path: Path to the file

    Returns:
        SHA256 hex digest string
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


class FileChangeEvent:
    """Represents a detected file change event."""

    def __init__(
        self,
        file_path: str,
        event_type: str,  # 'new', 'modified', 'unchanged', 'deleted'
        content_hash: Optional[str] = None,
        document_id: Optional[str] = None,
    ):
        self.file_path = file_path
        self.event_type = event_type
        self.content_hash = content_hash
        self.document_id = document_id or str(uuid.uuid4())

    def __repr__(self) -> str:
        return f"FileChangeEvent({self.event_type}, {self.file_path})"


class IntakeFileHandler(FileSystemEventHandler):
    """
    Watchdog file handler that tracks changes via content hashing.

    Monitors a directory for file system events and uses SHA256 hashing
    to detect actual content changes rather than relying on mtime.
    """

    def __init__(
        self,
        ledger: IngestionLedger,
        supported_extensions: Optional[set[str]] = None,
    ):
        """
        Initialize the file handler.

        Args:
            ledger: IngestionLedger instance for tracking state
            supported_extensions: Set of supported file extensions (e.g., {'.pdf', '.docx'})
        """
        super().__init__()
        self.ledger = ledger
        self.supported_extensions = supported_extensions or {
            ".pdf",
            ".docx",
            ".pptx",
            ".txt",
            ".md",
        }
        self.events: list[FileChangeEvent] = []

    def _is_supported_file(self, path: str) -> bool:
        """Check if file has a supported extension."""
        ext = Path(path).suffix.lower()
        return ext in self.supported_extensions

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events."""
        if event.is_directory:
            return

        path = event.src_path
        if not self._is_supported_file(path):
            return

        self._process_file(path, "new")

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return

        path = event.src_path
        if not self._is_supported_file(path):
            return

        self._process_file(path, "modified")

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion events."""
        if event.is_directory:
            return

        path = event.src_path
        if not self._is_supported_file(path):
            return

        # Log warning for deleted files
        logger.warning(f"File deleted from watch directory: {path}")
        self.events.append(FileChangeEvent(path, "deleted"))

    def _process_file(self, path: str, event_type: str) -> None:
        """
        Process a file event by computing hash and comparing with ledger.

        Args:
            path: File path
            event_type: Type of event ('new' or 'modified')
        """
        try:
            current_hash = compute_file_hash(path)
            ledger_entry = self.ledger.get_entry(path)

            if ledger_entry is None:
                # New file not in ledger - queue for ingestion
                logger.info(f"New file detected: {path}")
                doc_id = str(uuid.uuid4())
                self.ledger.upsert_entry(path, doc_id, current_hash, status="pending")
                self.events.append(FileChangeEvent(path, "new", current_hash, doc_id))

            elif ledger_entry["content_hash"] != current_hash:
                # File content changed - flag for re-ingestion
                logger.info(f"File modified: {path}")
                self.ledger.flag_for_reingestion(path)
                self.ledger.upsert_entry(
                    path,
                    ledger_entry["document_id"],
                    current_hash,
                    status="reingest_pending",
                )
                self.events.append(
                    FileChangeEvent(
                        path, "modified", current_hash, ledger_entry["document_id"]
                    )
                )

            else:
                # File unchanged - skip
                logger.debug(f"File unchanged, skipping: {path}")
                self.events.append(
                    FileChangeEvent(
                        path, "unchanged", current_hash, ledger_entry["document_id"]
                    )
                )

        except Exception as e:
            logger.error(f"Error processing file {path}: {e}")
            self.events.append(FileChangeEvent(path, "error"))


class FolderWatcher:
    """
    Watches a configured directory for file changes using content hashing.

    Scans recursively under WATCH_DIR and maintains state in a SQLite ledger.
    """

    def __init__(
        self,
        watch_dir: str | Path,
        ledger: Optional[IngestionLedger] = None,
        supported_extensions: Optional[set[str]] = None,
    ):
        """
        Initialize the folder watcher.

        Args:
            watch_dir: Directory to watch (will be created if it doesn't exist)
            ledger: IngestionLedger instance (created if not provided)
            supported_extensions: Set of supported file extensions
        """
        self.watch_dir = Path(watch_dir)
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = ledger or IngestionLedger()
        self.supported_extensions = supported_extensions or {
            ".pdf",
            ".docx",
            ".pptx",
            ".txt",
            ".md",
        }

        self._observer: Optional[Observer] = None
        self._file_handler: Optional[IntakeFileHandler] = None

    def _collect_files(self) -> list[Path]:
        """
        Recursively collect all files in watch directory.

        Returns:
            List of Path objects
        """
        files = []
        for root, _dirs, filenames in os.walk(self.watch_dir):
            for filename in filenames:
                file_path = Path(root) / filename
                if file_path.suffix.lower() in self.supported_extensions:
                    files.append(file_path)
        return files

    def scan(self) -> list[FileChangeEvent]:
        """
        Perform a full scan of the watch directory.

        Compares current file hashes with ledger state to detect:
        - New files: queue for ingestion
        - Unchanged files: skip with log
        - Modified files: flag for re-ingestion
        - Deleted files: warn only

        Returns:
            List of FileChangeEvent objects
        """
        events: list[FileChangeEvent] = []

        # Get current files
        current_files = self._collect_files()
        current_paths = {str(f) for f in current_files}

        # Get ledger entries
        ledger_hashes = self.ledger.get_entries_for_hash_check()
        ledger_paths = set(ledger_hashes.keys())

        # Process current files
        for file_path in current_files:
            file_path_str = str(file_path)
            try:
                current_hash = compute_file_hash(file_path)

                if file_path_str not in ledger_hashes:
                    # New file - queue for ingestion
                    logger.info(f"New file detected: {file_path_str}")
                    doc_id = str(uuid.uuid4())
                    self.ledger.upsert_entry(
                        file_path_str, doc_id, current_hash, status="pending"
                    )
                    events.append(
                        FileChangeEvent(file_path_str, "new", current_hash, doc_id)
                    )

                elif ledger_hashes[file_path_str] != current_hash:
                    # File content changed - flag for re-ingestion
                    logger.info(f"File modified: {file_path_str}")
                    entry = self.ledger.get_entry(file_path_str)
                    if entry:
                        self.ledger.flag_for_reingestion(file_path_str)
                        self.ledger.upsert_entry(
                            file_path_str,
                            entry["document_id"],
                            current_hash,
                            status="reingest_pending",
                        )
                        events.append(
                            FileChangeEvent(
                                file_path_str,
                                "modified",
                                current_hash,
                                entry["document_id"],
                            )
                        )

                else:
                    # File unchanged - skip
                    logger.debug(f"File unchanged, skipping: {file_path_str}")
                    entry = self.ledger.get_entry(file_path_str)
                    doc_id = entry["document_id"] if entry else None
                    events.append(
                        FileChangeEvent(
                            file_path_str, "unchanged", current_hash, doc_id
                        )
                    )

            except Exception as e:
                logger.error(f"Error processing file {file_path_str}: {e}")
                events.append(FileChangeEvent(file_path_str, "error"))

        # Check for deleted files
        deleted_paths = ledger_paths - current_paths
        for deleted_path in deleted_paths:
            logger.warning(f"File deleted from watch directory: {deleted_path}")
            events.append(FileChangeEvent(deleted_path, "deleted"))

        return events

    def start_watching(self) -> None:
        """Start real-time file system watching using watchdog."""
        if self._observer is not None:
            logger.warning("Watcher is already running")
            return

        self._file_handler = IntakeFileHandler(
            ledger=self.ledger,
            supported_extensions=self.supported_extensions,
        )

        self._observer = Observer()
        self._observer.schedule(self._file_handler, str(self.watch_dir), recursive=True)
        self._observer.start()
        logger.info(f"Started watching directory: {self.watch_dir} (recursive={True})")

        # Perform initial scan
        initial_events = self.scan()
        logger.info(f"Initial scan completed: {len(initial_events)} events detected")

    def stop_watching(self) -> None:
        """Stop file system watching."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            logger.info("Stopped watching directory")

    def get_pending_files(self) -> list[FileChangeEvent]:
        """
        Get all files pending ingestion.

        Returns:
            List of FileChangeEvent for pending files
        """
        return self.ledger.get_entries(status="pending")

    def __enter__(self) -> "FolderWatcher":
        """Context manager entry."""
        self.start_watching()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop_watching()
