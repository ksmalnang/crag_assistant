"""SQLite ledger for tracking ingestion state."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class IngestionLedger:
    """SQLite-based ledger for tracking file ingestion state."""

    def __init__(self, db_path: str | Path | None = None):
        """
        Initialize the ingestion ledger.

        Args:
            db_path: Path to SQLite database file. Defaults to data/ingestion_ledger.db
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "ingestion_ledger.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connection(self):
        """Context manager for database connections."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    document_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    ingested_at TIMESTAMP,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_path ON ingestion_state(file_path)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_id ON ingestion_state(document_id)
            """)

    def upsert_entry(
        self,
        file_path: str,
        document_id: str,
        content_hash: str,
        status: str = "pending",
    ) -> None:
        """
        Insert or update a file entry in the ledger.

        Args:
            file_path: Absolute path to the file
            document_id: Unique document identifier
            content_hash: SHA256 hash of file contents
            status: Current status (pending, ingested, skipped, error)
        """
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_state (file_path, document_id, content_hash, status, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(file_path) DO UPDATE SET
                    document_id = excluded.document_id,
                    content_hash = excluded.content_hash,
                    status = excluded.status,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (file_path, document_id, content_hash, status),
            )

    def get_entry(self, file_path: str) -> Optional[dict]:
        """
        Get ledger entry for a file.

        Args:
            file_path: Absolute path to the file

        Returns:
            Dictionary with entry data or None if not found
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM ingestion_state WHERE file_path = ?", (file_path,)
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def update_status(self, file_path: str, status: str) -> None:
        """
        Update the status of a file entry.

        Args:
            file_path: Absolute path to the file
            status: New status value
        """
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE ingestion_state
                SET status = ?, updated_at = CURRENT_TIMESTAMP, ingested_at = CASE WHEN ? = 'ingested' THEN CURRENT_TIMESTAMP ELSE ingested_at END
                WHERE file_path = ?
                """,
                (status, status, file_path),
            )

    def mark_ingested(
        self, file_path: str, document_id: str, content_hash: str
    ) -> None:
        """
        Mark a file as successfully ingested.

        Args:
            file_path: Absolute path to the file
            document_id: Unique document identifier
            content_hash: SHA256 hash of file contents
        """
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_state (file_path, document_id, content_hash, status, ingested_at, updated_at)
                VALUES (?, ?, ?, 'ingested', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(file_path) DO UPDATE SET
                    document_id = excluded.document_id,
                    content_hash = excluded.content_hash,
                    status = 'ingested',
                    ingested_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (file_path, document_id, content_hash),
            )

    def flag_for_reingestion(self, file_path: str) -> None:
        """
        Flag a file for re-ingestion due to content changes.

        Args:
            file_path: Absolute path to the file
        """
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE ingestion_state
                SET status = 'reingest_pending', updated_at = CURRENT_TIMESTAMP
                WHERE file_path = ?
                """,
                (file_path,),
            )
        logger.info(f"Flagged for re-ingestion: {file_path}")

    def get_all_entries(self, status: Optional[str] = None) -> list[dict]:
        """
        Get all ledger entries, optionally filtered by status.

        Args:
            status: Filter by status (optional)

        Returns:
            List of entry dictionaries
        """
        with self._connection() as conn:
            if status:
                cursor = conn.execute(
                    "SELECT * FROM ingestion_state WHERE status = ? ORDER BY updated_at DESC",
                    (status,),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM ingestion_state ORDER BY updated_at DESC"
                )
            return [dict(row) for row in cursor.fetchall()]

    def delete_entry(self, file_path: str) -> bool:
        """
        Delete an entry from the ledger.

        Args:
            file_path: Absolute path to the file

        Returns:
            True if entry was deleted, False if not found
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM ingestion_state WHERE file_path = ?", (file_path,)
            )
            return cursor.rowcount > 0

    def get_entries_for_hash_check(self) -> dict[str, str]:
        """
        Get all file paths and their content hashes for comparison.

        Returns:
            Dictionary mapping file_path -> content_hash
        """
        with self._connection() as conn:
            cursor = conn.execute("SELECT file_path, content_hash FROM ingestion_state")
            return {row["file_path"]: row["content_hash"] for row in cursor.fetchall()}
