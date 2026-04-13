"""Qdrant vector store connection and management."""

from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from pipeline.config import settings


class QdrantConnectionManager:
    """Manages Qdrant client connections and collections."""

    def __init__(self):
        self._client: Optional[QdrantClient] = None

    def get_client(self) -> QdrantClient:
        """Get or create Qdrant client connection."""
        if self._client is None:
            self._client = QdrantClient(url=settings.qdrant_url)
        return self._client

    def health_check(self) -> bool:
        """Check if Qdrant server is healthy."""
        try:
            client = self.get_client()
            response = client.http_client.get("/healthz")
            return response.status_code == 200
        except Exception:
            return False

    def create_collection(
        self,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
    ) -> bool:
        """Create a collection with dense and sparse vectors.

        Args:
            collection_name: Name of the collection (defaults to settings)
            vector_size: Size of dense vectors (defaults to settings)

        Returns:
            True if collection was created successfully
        """
        client = self.get_client()
        collection_name = collection_name or settings.qdrant_collection
        vector_size = vector_size or settings.qdrant_vector_size

        # Check if collection already exists
        collections = client.get_collections().collections
        if any(c.name == collection_name for c in collections):
            return True

        # Create collection with named vectors (dense + sparse)
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(size=vector_size, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False,
                    )
                )
            },
        )

        return True

    def close(self):
        """Close the Qdrant client connection."""
        if self._client:
            self._client = None


# Global instance
qdrant_manager = QdrantConnectionManager()
