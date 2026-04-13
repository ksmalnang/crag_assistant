"""Tests for Qdrant connection manager."""

from unittest.mock import Mock, patch

from qdrant_client.models import CollectionDescription, CollectionsResponse

from pipeline.qdrant import QdrantConnectionManager


class TestQdrantConnectionManager:
    """Test cases for QdrantConnectionManager."""

    def test_health_check_success(self):
        """Test health check returns True when server is healthy."""
        with patch.object(QdrantConnectionManager, "get_client") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value.http_client.get.return_value = mock_response

            manager = QdrantConnectionManager()
            result = manager.health_check()

            assert result is True

    def test_health_check_failure(self):
        """Test health check returns False when server is unreachable."""
        with patch.object(QdrantConnectionManager, "get_client") as mock_get:
            mock_get.side_effect = Exception("Connection refused")

            manager = QdrantConnectionManager()
            result = manager.health_check()

            assert result is False

    @patch("pipeline.qdrant.settings")
    def test_create_collection_new(self, mock_settings):
        """Test creating a new collection."""
        mock_settings.qdrant_collection = "test_collection"
        mock_settings.qdrant_vector_size = 3072

        with patch.object(QdrantConnectionManager, "get_client") as mock_get:
            mock_client = Mock()
            mock_client.get_collections.return_value = CollectionsResponse(
                collections=[]
            )
            mock_get.return_value = mock_client

            manager = QdrantConnectionManager()
            result = manager.create_collection()

            assert result is True
            mock_client.create_collection.assert_called_once()

    @patch("pipeline.qdrant.settings")
    def test_create_collection_existing(self, mock_settings):
        """Test creating collection that already exists."""
        mock_settings.qdrant_collection = "test_collection"
        mock_settings.qdrant_vector_size = 3072

        with patch.object(QdrantConnectionManager, "get_client") as mock_get:
            mock_client = Mock()
            mock_client.get_collections.return_value = CollectionsResponse(
                collections=[CollectionDescription(name="test_collection")]
            )
            mock_get.return_value = mock_client

            manager = QdrantConnectionManager()
            result = manager.create_collection()

            assert result is True
            mock_client.create_collection.assert_not_called()

    def test_close_connection(self):
        """Test closing connection sets client to None."""
        manager = QdrantConnectionManager()
        manager._client = Mock()
        manager.close()

        assert manager._client is None
