"""Qdrant upsert node with full ChunkMetadata payload."""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
)

from pipeline.config import settings
from vector_store.errors import UpsertError

logger = logging.getLogger(__name__)

# Reserved prefix for internal point IDs
SCHEMA_POINT_PREFIX = "__"


class QdrantUpsertNode:
    """Upserts chunks to Qdrant with full metadata payloads.

    Point ID = SHA256(document_id + chunk_index) for stable idempotent writes.
    Supports batch upsert with retry on 429/503 errors.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        retry_max: Optional[int] = None,
        retry_backoff_base: Optional[float] = None,
    ) -> None:
        """Initialise the upsert node.

        Args:
            client: Qdrant client instance.
            collection_name: Collection name (defaults to settings).
            batch_size: Upsert batch size (defaults to settings).
            retry_max: Max retry attempts (defaults to settings).
            retry_backoff_base: Backoff base in seconds (defaults to settings).
        """
        self._client = client
        self.collection_name = collection_name or settings.qdrant_collection
        self.batch_size = batch_size or settings.qdrant_batch_size
        self.retry_max = retry_max or settings.qdrant_retry_max
        self.retry_backoff_base = (
            retry_backoff_base or settings.qdrant_retry_backoff_base
        )

    @staticmethod
    def make_point_id(document_id: str, chunk_index: int) -> str:
        """Generate stable point ID from document_id + chunk_index.

        SHA256 hash ensures idempotency across re-ingestions.

        Args:
            document_id: Document identifier.
            chunk_index: Chunk index within document.

        Returns:
            SHA256 hex digest string.
        """
        raw = f"{document_id}:{chunk_index}"
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def build_payload(
        chunk_metadata: Any,
        source_path: str,
        source_name: str,
        faculty: str,
        doc_type: str,
        semester: str,
        ingested_at: Optional[str] = None,
    ) -> dict:
        """Build Qdrant payload from ChunkMetadata and document metadata.

        Args:
            chunk_metadata: ChunkMetadata Pydantic model instance.
            source_path: Original file path.
            source_name: Human-readable document name.
            faculty: Faculty enum value string.
            doc_type: DocType enum value string.
            semester: Semester string (e.g. "2025-S1").
            ingested_at: ISO timestamp of ingestion (auto-generated if None).

        Returns:
            Payload dictionary for Qdrant point.
        """
        if ingested_at is None:
            ingested_at = datetime.now(timezone.utc).isoformat()

        return {
            "document_id": chunk_metadata.document_id,
            "source_path": source_path,
            "source_name": source_name,
            "faculty": faculty,
            "doc_type": doc_type,
            "semester": semester,
            "chunk_index": chunk_metadata.chunk_index,
            "page_start": chunk_metadata.page_start,
            "page_end": chunk_metadata.page_end,
            "heading_path": chunk_metadata.heading_path,
            "section_title": chunk_metadata.section_title,
            "char_start": chunk_metadata.char_start,
            "char_end": chunk_metadata.char_end,
            "token_count": chunk_metadata.token_count,
            "ingested_at": ingested_at,
            "unclassified": chunk_metadata.unclassified,
            "is_low_confidence": chunk_metadata.is_low_confidence,
            "table_index": chunk_metadata.table_index,
            "page_image_path": chunk_metadata.page_image_path,
            "overlap_context": chunk_metadata.overlap_context,
        }

    def upsert(
        self,
        chunks: list[tuple],
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict],
        payload_extra: dict,
        expected_count: Optional[int] = None,
    ) -> int:
        """Upsert chunks to Qdrant with retry logic.

        Args:
            chunks: List of (document_id, chunk_index, chunk_metadata) tuples.
            dense_vectors: Corresponding dense embedding vectors.
            sparse_vectors: Corresponding sparse vectors {indices, values}.
            payload_extra: Extra payload fields (source_path, source_name,
                           faculty, doc_type, semester, ingested_at).
            expected_count: Expected number of points after upsert for
                           post-upsert verification.

        Returns:
            Number of points upserted.

        Raises:
            UpsertError: If upsert fails after all retries.
        """
        if not chunks:
            return 0

        total_upserted = 0
        ingested_at = payload_extra.get("ingested_at")

        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i : i + self.batch_size]
            batch_dense = dense_vectors[i : i + self.batch_size]
            batch_sparse = sparse_vectors[i : i + self.batch_size]

            points = []
            for (doc_id, chunk_idx, meta), dense, sparse in zip(
                batch_chunks, batch_dense, batch_sparse, strict=True
            ):
                point_id = self.make_point_id(doc_id, chunk_idx)
                payload = self.build_payload(
                    chunk_metadata=meta,
                    **payload_extra,
                    ingested_at=ingested_at,
                )

                # Build sparse vector for Qdrant
                sparse_vector = None
                if sparse and sparse.get("indices"):
                    from qdrant_client.models import SparseVector

                    sparse_vector = SparseVector(
                        indices=sparse["indices"],
                        values=sparse["values"],
                    )

                points.append(
                    PointStruct(
                        id=point_id,
                        vector={
                            "dense": dense,
                            "sparse": sparse_vector,
                        },
                        payload=payload,
                    )
                )

            # Upsert with retry on 429/503
            self._upsert_with_retry(points, batch_index=i // self.batch_size)
            total_upserted += len(points)

        # Post-upsert count verification
        if expected_count is not None:
            actual_count = self.count_points()
            if actual_count < expected_count:
                shortfall = expected_count - actual_count
                logger.warning(
                    "Post-upsert count verification failed for '%s': "
                    "expected=%d, actual=%d (shortfall: %d)",
                    self.collection_name,
                    expected_count,
                    actual_count,
                    shortfall,
                )
            else:
                logger.info(
                    "Post-upsert count verified for '%s': expected=%d, actual=%d",
                    self.collection_name,
                    expected_count,
                    actual_count,
                )

        logger.info("Upserted %d points to '%s'", total_upserted, self.collection_name)
        return total_upserted

    def _upsert_with_retry(self, points: list[PointStruct], batch_index: int) -> None:
        """Upsert with exponential backoff retry on 429/503.

        Args:
            points: List of PointStruct to upsert.
            batch_index: Batch index for error reporting.

        Raises:
            UpsertError: If upsert fails after all retries.
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.retry_max):
            try:
                self._client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
                return  # Success
            except Exception as exc:
                last_error = exc
                status_code = getattr(exc, "status_code", None)
                # Retry on 429 (rate limit) or 503 (service unavailable)
                if status_code not in (429, 503):
                    raise

                backoff = self.retry_backoff_base * (2**attempt)
                logger.warning(
                    "Upsert batch %d failed (status=%s), retry %d/%d in %.1fs: %s",
                    batch_index,
                    status_code,
                    attempt + 1,
                    self.retry_max,
                    backoff,
                    exc,
                )
                time.sleep(backoff)

        raise UpsertError(
            collection_name=self.collection_name,
            reason=str(last_error),
            batch_index=batch_index,
            retry_count=self.retry_max,
        )

    def count_points(self, document_id: Optional[str] = None) -> int:
        """Count points in collection, optionally filtered by document_id.

        Args:
            document_id: Filter by document ID.

        Returns:
            Point count.
        """
        if document_id:
            filter_obj = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            )
            return self._client.count(
                collection_name=self.collection_name,
                count_filter=filter_obj,
            ).count
        return self._client.count(collection_name=self.collection_name).count
