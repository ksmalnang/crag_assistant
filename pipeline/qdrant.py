"""Qdrant vector store connection and management."""

from __future__ import annotations

from typing import Optional

from qdrant_client import QdrantClient

from pipeline.config import settings


class QdrantConnectionManager:
    """Manages Qdrant client connections."""

    def __init__(self):
        self._client: Optional[QdrantClient] = None

    def get_client(self) -> QdrantClient:
        """Get or create Qdrant client connection."""
        if self._client is None:
            kwargs: dict = {"url": settings.qdrant_url}
            if settings.qdrant_api_key:
                kwargs["api_key"] = settings.qdrant_api_key
            self._client = QdrantClient(**kwargs)
        return self._client

    def health_check(self) -> bool:
        """Check if Qdrant server is healthy."""
        try:
            client = self.get_client()
            response = client.http_client.get("/healthz")
            return response.status_code == 200
        except Exception:
            return False

    def close(self):
        """Close the Qdrant client connection."""
        if self._client:
            self._client.close()
            self._client = None


# Global instance
qdrant_manager = QdrantConnectionManager()
