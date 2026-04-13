"""Tests for LangSmith tracing configuration."""

import os
from unittest.mock import patch

from pipeline.tracing import (
    disable_langsmith_tracing,
    get_tracing_status,
    setup_langsmith_tracing,
)


class TestLangSmithTracing:
    """Test cases for LangSmith tracing."""

    def test_tracing_disabled_when_not_configured(self):
        """Test tracing returns False when not properly configured."""
        with patch("pipeline.tracing.settings") as mock_settings:
            mock_settings.langsmith_enabled = False

            result = setup_langsmith_tracing()

            assert result is False

    @patch.dict(os.environ, {}, clear=True)
    def test_tracing_sets_environment_variables(self):
        """Test that tracing sets correct environment variables."""
        with patch("pipeline.tracing.settings") as mock_settings:
            mock_settings.langsmith_enabled = True
            mock_settings.langchain_project = "test-project"
            mock_settings.langchain_api_key = "test-key"

            result = setup_langsmith_tracing()

            assert result is True
            assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
            assert os.environ["LANGCHAIN_PROJECT"] == "test-project"
            assert os.environ["LANGCHAIN_API_KEY"] == "test-key"

    @patch.dict(os.environ, {}, clear=True)
    def test_disable_tracing_clears_variables(self):
        """Test that disabling tracing removes environment variables."""
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = "test-key"

        disable_langsmith_tracing()

        assert "LANGCHAIN_TRACING_V2" not in os.environ
        assert "LANGCHAIN_API_KEY" not in os.environ

    def test_get_tracing_status(self):
        """Test getting tracing status returns correct structure."""
        with patch("pipeline.tracing.settings") as mock_settings:
            mock_settings.langsmith_enabled = True
            mock_settings.langchain_project = "test-project"
            mock_settings.langchain_api_key = "test-key"

            status = get_tracing_status()

            assert "enabled" in status
            assert "project" in status
            assert "api_key_configured" in status
            assert "tracing_v2_env" in status
            assert status["enabled"] is True
            assert status["project"] == "test-project"
            assert status["api_key_configured"] is True
