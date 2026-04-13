"""SQLite-based embedding cache to avoid re-computation."""

from __future__ import annotations

import logging
import sqlite3
import struct
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_TTL_DAYS = 30
DEFAULT_CACHE_PATH = "data/embedding_cache.db"


class EmbeddingCache:
    """SQLite-based cache for dense and sparse embeddings.

    Avoids re-embedding unchanged chunks between ingestion runs.
    Cache entries older than CACHE_TTL_DAYS are invalidated on lookup.
    """

    def __init__(
        self,
        cache_path: str = DEFAULT_CACHE_PATH,
        ttl_days: int = DEFAULT_CACHE_TTL_DAYS,
    ) -> None:
        """Initialise the embedding cache.

        Args:
            cache_path: Path to the SQLite database file.
            ttl_days: Time-to-live for cache entries in days.
        """
        self.cache_path = Path(cache_path)
        self.ttl_days = ttl_days
        self._conn: Optional[sqlite3.Connection] = None
        self._hit_count = 0
        self._miss_count = 0

    def _connect(self) -> sqlite3.Connection:
        """Get or create SQLite connection."""
        if self._conn is None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.cache_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        """Create the cache table if it doesn't exist."""
        conn = self._conn
        assert conn is not None
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                chunk_hash TEXT PRIMARY KEY,
                dense_vector BLOB NOT NULL,
                sparse_indices BLOB NOT NULL,
                sparse_values BLOB NOT NULL,
                dim INTEGER NOT NULL,
                cached_at REAL NOT NULL
            )
            """
        )
        conn.commit()

    def _serialise_vector(self, vector: list[float]) -> bytes:
        """Serialise a float list to binary BLOB."""
        return struct.pack(f"{len(vector)}f", *vector)

    def _deserialise_vector(self, blob: bytes) -> list[float]:
        """Deserialise a float list from binary BLOB."""
        count = len(blob) // struct.calcsize("f")
        return list(struct.unpack(f"{count}f", blob))

    def get(self, chunk_hash: str) -> Optional[dict]:
        """Look up a cached embedding by chunk hash.

        Args:
            chunk_hash: Hash of the chunk (consistent with Qdrant point ID).

        Returns:
            Dict with dense_vector, sparse_indices, sparse_values, dim
            or None if cache miss or expired.
        """
        conn = self._connect()
        cutoff = time.time() - (self.ttl_days * 86400)

        row = conn.execute(
            "SELECT dense_vector, sparse_indices, sparse_values, dim, cached_at "
            "FROM embedding_cache WHERE chunk_hash = ? AND cached_at > ?",
            (chunk_hash, cutoff),
        ).fetchone()

        if row is None:
            self._miss_count += 1
            return None

        # Expired entry — delete it
        if row[4] < cutoff:
            conn.execute(
                "DELETE FROM embedding_cache WHERE chunk_hash = ?", (chunk_hash,)
            )
            conn.commit()
            self._miss_count += 1
            return None

        self._hit_count += 1
        return {
            "dense_vector": self._deserialise_vector(row[0]),
            "sparse_indices": self._deserialise_vector(row[1]),
            "sparse_values": self._deserialise_vector(row[2]),
            "dim": row[3],
            "cached_at": row[4],
        }

    def put(
        self,
        chunk_hash: str,
        dense_vector: list[float],
        sparse_indices: list[float],
        sparse_values: list[float],
    ) -> None:
        """Store an embedding in the cache.

        Args:
            chunk_hash: Hash of the chunk.
            dense_vector: Dense embedding vector.
            sparse_indices: Sparse vector indices (float-encoded ints).
            sparse_values: Sparse vector values.
        """
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO embedding_cache "
            "(chunk_hash, dense_vector, sparse_indices, sparse_values, dim, cached_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                chunk_hash,
                self._serialise_vector(dense_vector),
                self._serialise_vector(sparse_indices),
                self._serialise_vector(sparse_values),
                len(dense_vector),
                time.time(),
            ),
        )
        conn.commit()

    def delete(self, chunk_hash: str) -> None:
        """Remove a cached embedding.

        Args:
            chunk_hash: Hash of the chunk to remove.
        """
        conn = self._connect()
        conn.execute("DELETE FROM embedding_cache WHERE chunk_hash = ?", (chunk_hash,))
        conn.commit()

    def invalidate_expired(self) -> int:
        """Remove all expired entries. Returns count of deleted rows."""
        conn = self._connect()
        cutoff = time.time() - (self.ttl_days * 86400)
        cursor = conn.execute(
            "DELETE FROM embedding_cache WHERE cached_at <= ?", (cutoff,)
        )
        conn.commit()
        deleted = cursor.rowcount
        if deleted:
            logger.info("Invalidated %d expired cache entries", deleted)
        return deleted

    def stats(self) -> dict:
        """Return cache hit/miss statistics."""
        conn = self._connect()
        total = conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0]
        return {
            "total_entries": total,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_ratio": (
                self._hit_count / (self._hit_count + self._miss_count)
                if (self._hit_count + self._miss_count) > 0
                else 0.0
            ),
        }

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
