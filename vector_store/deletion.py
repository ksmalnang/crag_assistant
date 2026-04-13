"""Stale chunk deletion on document re-ingestion."""

from __future__ import annotations

import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
)

from pipeline.config import settings
from vector_store.errors import DeletionError

logger = logging.getLogger(__name__)


class StaleDeletionNode:
    """Deletes old chunks from Qdrant when a document is re-ingested.

    Deletion happens AFTER new chunks are upserted, ensuring partial
    ingestion failures don't result in data loss.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: Optional[str] = None,
    ) -> None:
        """Initialise the deletion node.

        Args:
            client: Qdrant client instance.
            collection_name: Collection name (defaults to settings).
        """
        self._client = client
        self.collection_name = collection_name or settings.qdrant_collection

    def delete_by_document_id(self, document_id: str) -> int:
        """Delete all points matching the given document_id.

        Uses Qdrant's delete API with a filter on the document_id payload field.

        Args:
            document_id: Document identifier whose chunks should be deleted.

        Returns:
            Number of points deleted.

        Raises:
            DeletionError: If deletion fails.
        """
        try:
            deletion_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            )

            # Count before deletion for logging
            count_before = self._client.count(
                collection_name=self.collection_name,
                count_filter=deletion_filter,
            ).count

            if count_before == 0:
                logger.info("No stale chunks found for document '%s'", document_id)
                return 0

            self._client.delete(
                collection_name=self.collection_name,
                points_selector=deletion_filter,
            )

            logger.info(
                "Deleted %d stale chunks for document '%s' from '%s'",
                count_before,
                document_id,
                self.collection_name,
            )
            return count_before

        except Exception as exc:
            raise DeletionError(
                collection_name=self.collection_name,
                document_id=document_id,
                reason=str(exc),
            ) from exc

    def delete_old_after_reingest(
        self,
        document_id: str,
        new_point_ids: list[str],
        previous_chunk_count: int = 0,
    ) -> dict:
        """Report on stale chunks after new ones have been upserted.

        Since upsert is idempotent (same point IDs replace existing ones),
        this method compares current count vs expected to flag excess points.
        Actual deletion of excess should be done via
        `delete_stale_by_chunk_indices`.

        Args:
            document_id: Document identifier.
            new_point_ids: List of point IDs for the newly upserted chunks.
            previous_chunk_count: Expected chunk count from previous ingestion
                                  (for logging/comparison).

        Returns:
            Dict with comparison info for monitoring.
        """
        current_count = self._client.count(
            collection_name=self.collection_name,
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        ).count

        excess = max(0, current_count - len(new_point_ids))

        if excess > 0:
            logger.warning(
                "Document '%s' has %d points but only %d new chunks. "
                "%d stale points may remain. "
                "Call delete_stale_by_chunk_indices to clean up.",
                document_id,
                current_count,
                len(new_point_ids),
                excess,
            )

        return {
            "document_id": document_id,
            "current_count": current_count,
            "new_count": len(new_point_ids),
            "excess": excess,
            "previous_chunk_count": previous_chunk_count,
        }

    def delete_stale_by_chunk_indices(
        self,
        document_id: str,
        keep_chunk_indices: list[int],
    ) -> int:
        """Delete stale chunks for a document, keeping only specified indices.

        This is the precise deletion path: keeps only chunks with the
        specified chunk_indices and deletes all others for the document.

        Args:
            document_id: Document identifier.
            keep_chunk_indices: Chunk indices to preserve.

        Returns:
            Number of points deleted.

        Raises:
            DeletionError: If deletion fails.
        """
        try:
            # Delete all points for this document_id whose chunk_index
            # is NOT in keep_chunk_indices
            delete_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ],
                must_not=[
                    FieldCondition(
                        key="chunk_index",
                        match=MatchAny(any=list(keep_chunk_indices)),
                    )
                ],
            )

            count_before = self._client.count(
                collection_name=self.collection_name,
                count_filter=delete_filter,
            ).count

            if count_before == 0:
                return 0

            self._client.delete(
                collection_name=self.collection_name,
                points_selector=delete_filter,
            )

            logger.info(
                "Deleted %d stale chunks for document '%s' (kept %d indices)",
                count_before,
                document_id,
                len(keep_chunk_indices),
            )
            return count_before

        except Exception as exc:
            raise DeletionError(
                collection_name=self.collection_name,
                document_id=document_id,
                reason=str(exc),
            ) from exc
