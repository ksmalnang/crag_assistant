"""Document-level metadata store using SQLite."""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .resolver import DocumentStatus, Faculty

logger = logging.getLogger(__name__)


class DocumentMetadataStore:
    """
    SQLite-based document-level metadata store.

    Stores: {document_id PK, source_path, source_name, faculty, doc_type,
             semester, ingested_at, chunk_count, status}

    Supports queries by faculty, semester, and status.
    """

    def __init__(self, db_path: Optional[str | Path] = None):
        """
        Initialize the metadata store.

        Args:
            db_path: Path to SQLite database. Defaults to data/metadata.db
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "metadata.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
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
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    faculty TEXT NOT NULL,
                    doc_type TEXT NOT NULL,
                    semester TEXT NOT NULL,
                    ingested_at TIMESTAMP,
                    chunk_count INTEGER DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_faculty ON documents(faculty)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_semester ON documents(semester)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON documents(status)
            """)

    def upsert_document(
        self,
        document_id: str,
        source_path: str,
        source_name: str,
        faculty: Faculty,
        doc_type: str,
        semester: str,
        chunk_count: int = 0,
        status: DocumentStatus = DocumentStatus.PENDING,
    ) -> None:
        """
        Insert or update a document record.

        Args:
            document_id: Unique document identifier (SHA256 hash)
            source_path: File path to the document
            source_name: Display name
            faculty: Faculty enum value
            doc_type: Document type string
            semester: Semester string (e.g., '2024-S1')
            chunk_count: Number of chunks
            status: Document status
        """
        with self._connection() as conn:
            # Check if document exists
            cursor = conn.execute(
                "SELECT status FROM documents WHERE document_id = ?", (document_id,)
            )
            existing = cursor.fetchone()

            if existing:
                # Update existing record
                old_status = existing["status"]

                # If re-ingesting changed file, mark old as stale first
                if old_status == DocumentStatus.INGESTED.value and status in [
                    DocumentStatus.PENDING,
                    DocumentStatus.STALE,
                ]:
                    # Mark as stale first (valid transition: ingested -> stale)
                    conn.execute(
                        """
                        UPDATE documents
                        SET status = 'stale', updated_at = CURRENT_TIMESTAMP
                        WHERE document_id = ?
                        """,
                        (document_id,),
                    )
                    logger.info(
                        f"Document {document_id} marked as stale for re-ingestion"
                    )
                    old_status = "stale"  # Update for transition validation

                # Validate status transition
                if not self._validate_status_transition(old_status, status.value):
                    logger.warning(
                        f"Invalid status transition: {old_status} -> {status.value} "
                        f"for document {document_id}"
                    )

                conn.execute(
                    """
                    UPDATE documents
                    SET source_path = ?, source_name = ?, faculty = ?,
                        doc_type = ?, semester = ?, chunk_count = ?,
                        status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE document_id = ?
                    """,
                    (
                        source_path,
                        source_name,
                        faculty.value,
                        doc_type,
                        semester,
                        chunk_count,
                        status.value,
                        document_id,
                    ),
                )
            else:
                # Insert new record
                conn.execute(
                    """
                    INSERT INTO documents (
                        document_id, source_path, source_name, faculty,
                        doc_type, semester, chunk_count, status, ingested_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        document_id,
                        source_path,
                        source_name,
                        faculty.value,
                        doc_type,
                        semester,
                        chunk_count,
                        status.value,
                    ),
                )

    def update_status(self, document_id: str, status: DocumentStatus) -> None:
        """
        Update document status.

        Args:
            document_id: Document identifier
            status: New status
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT status FROM documents WHERE document_id = ?", (document_id,)
            )
            existing = cursor.fetchone()

            if not existing:
                raise ValueError(f"Document {document_id} not found in metadata store")

            old_status = existing["status"]

            # Validate status transition
            if not self._validate_status_transition(old_status, status.value):
                raise ValueError(
                    f"Invalid status transition: {old_status} -> {status.value}"
                )

            conn.execute(
                """
                UPDATE documents
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE document_id = ?
                """,
                (status.value, document_id),
            )

    def mark_ingested(self, document_id: str, chunk_count: int) -> None:
        """
        Mark document as successfully ingested.

        Args:
            document_id: Document identifier
            chunk_count: Number of chunks created
        """
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE documents
                SET status = 'ingested', chunk_count = ?,
                    ingested_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE document_id = ?
                """,
                (chunk_count, document_id),
            )

    def mark_failed(self, document_id: str) -> None:
        """Mark document as failed ingestion."""
        self.update_status(document_id, DocumentStatus.FAILED)

    def get_document(self, document_id: str) -> Optional[dict]:
        """
        Get document metadata by ID.

        Args:
            document_id: Document identifier

        Returns:
            Dictionary with document data or None
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def query_documents(
        self,
        faculty: Optional[Faculty] = None,
        semester: Optional[str] = None,
        status: Optional[DocumentStatus] = None,
    ) -> list[dict]:
        """
        Query documents with filters.

        Args:
            faculty: Filter by faculty
            semester: Filter by semester
            status: Filter by status

        Returns:
            List of matching documents
        """
        query = "SELECT * FROM documents WHERE 1=1"
        params = []

        if faculty:
            query += " AND faculty = ?"
            params.append(faculty.value)

        if semester:
            query += " AND semester = ?"
            params.append(semester)

        if status:
            query += " AND status = ?"
            params.append(status.value)

        query += " ORDER BY updated_at DESC"

        with self._connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_stale_documents(self) -> list[dict]:
        """Get all documents with stale status."""
        return self.query_documents(status=DocumentStatus.STALE)

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document record.

        Args:
            document_id: Document identifier

        Returns:
            True if deleted, False if not found
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM documents WHERE document_id = ?", (document_id,)
            )
            return cursor.rowcount > 0

    @staticmethod
    def _validate_status_transition(old_status: str, new_status: str) -> bool:
        """
        Validate status transition.

        Valid transitions:
        - pending -> ingested, failed
        - ingested -> stale
        - stale -> pending, ingested, failed
        - failed -> pending (retry)
        """
        valid_transitions = {
            "pending": {"ingested", "failed"},
            "ingested": {"stale"},
            "stale": {"pending", "ingested", "failed"},
            "failed": {"pending"},
        }

        return new_status in valid_transitions.get(old_status, set())


def cli() -> None:
    """CLI interface for document metadata store."""
    parser = argparse.ArgumentParser(description="Document metadata store CLI")
    parser.add_argument(
        "command",
        choices=["list", "get", "status"],
        help="Command to execute",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to metadata database",
    )
    parser.add_argument(
        "--faculty",
        default=None,
        choices=[f.value for f in Faculty],
        help="Filter by faculty",
    )
    parser.add_argument(
        "--semester",
        default=None,
        help="Filter by semester (e.g., 2024-S1)",
    )
    parser.add_argument(
        "--status",
        default=None,
        choices=[s.value for s in DocumentStatus],
        help="Filter by status",
    )
    parser.add_argument(
        "--document-id",
        default=None,
        help="Document ID for 'get' command",
    )

    args = parser.parse_args()

    # Initialize store
    store = DocumentMetadataStore(db_path=args.db_path)

    if args.command == "list":
        faculty = Faculty(args.faculty) if args.faculty else None
        status = DocumentStatus(args.status) if args.status else None

        documents = store.query_documents(
            faculty=faculty,
            semester=args.semester,
            status=status,
        )

        if not documents:
            print("No documents found.")
            return

        print(f"\nFound {len(documents)} document(s):\n")
        print(
            f"{'Document ID':<64} {'Source':<30} {'Faculty':<15} {'Semester':<10} {'Status':<10} {'Chunks':<8}"
        )
        print("-" * 140)

        for doc in documents:
            print(
                f"{doc['document_id']:<64} "
                f"{doc['source_name']:<30} "
                f"{doc['faculty']:<15} "
                f"{doc['semester']:<10} "
                f"{doc['status']:<10} "
                f"{doc['chunk_count']:<8}"
            )

    elif args.command == "get":
        if not args.document_id:
            print("Error: --document-id is required for 'get' command")
            sys.exit(1)

        doc = store.get_document(args.document_id)
        if doc:
            print(f"\nDocument Metadata:\n")
            for key, value in doc.items():
                print(f"  {key}: {value}")
        else:
            print(f"Document {args.document_id} not found")
            sys.exit(1)

    elif args.command == "status":
        if not args.document_id:
            print("Error: --document-id is required for 'status' command")
            sys.exit(1)

        doc = store.get_document(args.document_id)
        if doc:
            print(f"Document {args.document_id} status: {doc['status']}")
        else:
            print(f"Document {args.document_id} not found")
            sys.exit(1)


if __name__ == "__main__":
    cli()
