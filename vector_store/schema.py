"""Qdrant collection schema definition and provisioning."""

from __future__ import annotations

import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from pipeline.config import settings
from vector_store.errors import SchemaError

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"
SCHEMA_VERSION_KEY = "schema_version"

# Payload fields that must have keyword indexes
KEYWORD_INDEX_FIELDS = ["faculty", "doc_type", "semester", "document_id"]
INTEGER_INDEX_FIELDS = ["chunk_index"]


def _parse_sparse_distance(distance_str: str) -> str:
    """Parse sparse distance string to Qdrant modifier value.

    Qdrant sparse vectors use modifier='idf' for BM25-style weighting
    or modifier='none' for raw dot product.

    Args:
        distance_str: Distance string (e.g. "Dot", "IDF").

    Returns:
        Modifier string value ('idf' or 'none').
    """
    distance_map = {
        "dot": "none",
        "cosine": "none",
        "idf": "idf",
    }
    return distance_map.get(distance_str.lower(), "none")


class CollectionSchemaManager:
    """Defines and provisions Qdrant collection schema.

    Handles collection creation with dual dense+sparse vectors,
    payload index creation, and schema versioning for auditability.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: Optional[str] = None,
        dense_dim: Optional[int] = None,
        sparse_distance: Optional[str] = None,
        schema_version: Optional[str] = None,
    ) -> None:
        """Initialise the schema manager.

        Args:
            client: Qdrant client instance.
            collection_name: Collection name (defaults to settings).
            dense_dim: Dense vector dimension (defaults to settings).
            sparse_distance: Sparse vector distance (defaults to settings).
            schema_version: Schema version string (defaults to settings).
        """
        self._client = client
        self.collection_name = collection_name or settings.qdrant_collection
        self.dense_dim = dense_dim or settings.qdrant_dense_dim
        self.sparse_distance = sparse_distance or settings.qdrant_sparse_distance
        self.schema_version = schema_version or settings.qdrant_schema_version

    def ensure_collection(self) -> bool:
        """Ensure collection exists with correct schema.

        Creates the collection if it doesn't exist, or verifies schema
        if it does. Raises SchemaError on mismatch.

        Returns:
            True if collection is ready.
        """
        collections = self._client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            return self._create_collection()

        return self._verify_schema()

    def _create_collection(self) -> bool:
        """Create the collection with dual vectors and payload indexes."""
        logger.info(
            "Creating collection '%s' (dense_dim=%d, sparse_distance=%s, schema_version=%s)",
            self.collection_name,
            self.dense_dim,
            self.sparse_distance,
            self.schema_version,
        )

        sparse_modifier = _parse_sparse_distance(self.sparse_distance)

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=self.dense_dim,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                    modifier=sparse_modifier,
                )
            },
        )

        self._create_payload_indexes()
        self._store_schema_version()

        logger.info(
            "Collection '%s' created with schema version %s",
            self.collection_name,
            self.schema_version,
        )
        return True

    def _verify_schema(self) -> bool:
        """Verify existing collection schema matches expected config."""
        collection_info = self._client.get_collection(self.collection_name)

        # Check dense vector config
        vectors_config = collection_info.config.params.vectors
        if "dense" not in vectors_config:
            raise SchemaError(
                reason="missing_dense_vector",
                collection_name=self.collection_name,
                expected={"dense": {"size": self.dense_dim, "distance": "Cosine"}},
                actual={
                    "vectors": list(vectors_config.keys()) if vectors_config else "none"
                },
            )

        dense_config = vectors_config["dense"]
        actual_size = getattr(dense_config, "size", None)
        actual_distance = getattr(dense_config, "distance", None)

        if actual_size != self.dense_dim:
            raise SchemaError(
                reason="dense_dim_mismatch",
                collection_name=self.collection_name,
                expected={"dense_dim": self.dense_dim},
                actual={"dense_dim": actual_size},
            )

        if actual_distance != Distance.COSINE:
            raise SchemaError(
                reason="distance_mismatch",
                collection_name=self.collection_name,
                expected={"distance": "Cosine"},
                actual={"distance": str(actual_distance)},
            )

        # Check sparse vector config
        sparse_config = collection_info.config.params.sparse_vectors
        if "sparse" not in sparse_config:
            raise SchemaError(
                collection_name=self.collection_name,
                expected={"sparse": "configured"},
                actual={
                    "sparse_vectors": list(sparse_config.keys())
                    if sparse_config
                    else "none"
                },
            )

        # Verify sparse distance
        expected_sparse_modifier = _parse_sparse_distance(self.sparse_distance)
        actual_sparse_modifier = getattr(sparse_config["sparse"], "modifier", None)
        if actual_sparse_modifier != expected_sparse_modifier:
            raise SchemaError(
                reason="sparse_distance_mismatch",
                collection_name=self.collection_name,
                expected={"sparse_distance": expected_sparse_modifier},
                actual={"sparse_distance": actual_sparse_modifier},
            )

        # Check payload indexes exist
        payload_indexes = collection_info.payload_index_schema or {}
        required_fields = KEYWORD_INDEX_FIELDS + INTEGER_INDEX_FIELDS
        missing = [f for f in required_fields if f not in payload_indexes]

        if missing:
            # Try to create missing indexes
            logger.warning(
                "Missing payload indexes for fields %s in '%s', creating now",
                missing,
                self.collection_name,
            )
            self._create_payload_indexes(fields=missing)

        # Verify schema version
        stored_version = self._get_stored_schema_version()
        if stored_version and stored_version != self.schema_version:
            logger.warning(
                "Schema version mismatch: stored=%s, expected=%s",
                stored_version,
                self.schema_version,
            )

        logger.info(
            "Collection '%s' schema verified (version=%s)",
            self.collection_name,
            stored_version or self.schema_version,
        )
        return True

    def _create_payload_indexes(
        self,
        fields: Optional[list[str]] = None,
    ) -> None:
        """Create payload indexes for filtering.

        Args:
            fields: Specific fields to index. If None, creates all required indexes.
        """
        keyword_fields = fields if fields else KEYWORD_INDEX_FIELDS
        integer_fields = (
            [f for f in fields if f in INTEGER_INDEX_FIELDS]
            if fields
            else INTEGER_INDEX_FIELDS
        )

        for field in keyword_fields:
            if field in INTEGER_INDEX_FIELDS:
                continue
            try:
                self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                logger.debug("Created keyword index for '%s'", field)
            except Exception:
                # Index may already exist — ignore
                logger.debug("Index for '%s' may already exist, skipping", field)

        for field in integer_fields:
            try:
                self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.INTEGER,
                )
                logger.debug("Created integer index for '%s'", field)
            except Exception:
                logger.debug("Index for '%s' may already exist, skipping", field)

    def _store_schema_version(self) -> None:
        """Store schema version as collection metadata."""
        # Qdrant doesn't support comments on collections natively.
        # We store the schema version in the collection's optimizer config
        # via a payload index on a reserved field.
        try:
            self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name=SCHEMA_VERSION_KEY,
                field_schema=PayloadSchemaType.KEYWORD,
            )
            # Set the version value via a dummy upsert
            from qdrant_client.models import PointStruct, SparseVector

            self._client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id="__schema_version__",
                        vector={
                            "dense": [0.0] * self.dense_dim,
                            "sparse": SparseVector(indices=[], values=[]),
                        },
                        payload={SCHEMA_VERSION_KEY: self.schema_version},
                    )
                ],
            )
            logger.info(
                "Schema version %s stored for '%s'",
                self.schema_version,
                self.collection_name,
            )
        except Exception:
            logger.warning(
                "Failed to store schema version for '%s'",
                self.collection_name,
            )

    def _get_stored_schema_version(self) -> Optional[str]:
        """Retrieve stored schema version from collection."""
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            results = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key=SCHEMA_VERSION_KEY,
                            match=MatchValue(value=self.schema_version),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            points, _ = results
            if points:
                return points[0].payload.get(SCHEMA_VERSION_KEY)
        except Exception:
            pass
        return None
