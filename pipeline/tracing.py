"""LangSmith tracing configuration and utilities."""

import os

from pipeline.config import settings


def setup_langsmith_tracing() -> bool:
    """Initialize LangSmith tracing for development.

    Returns:
        True if tracing was successfully enabled, False otherwise.
    """
    if not settings.langsmith_enabled:
        return False

    # Set environment variables for LangSmith
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key

    return True


def disable_langsmith_tracing():
    """Gracefully disable LangSmith tracing."""
    os.environ.pop("LANGCHAIN_TRACING_V2", None)
    os.environ.pop("LANGCHAIN_API_KEY", None)
    os.environ.pop("LANGCHAIN_PROJECT", None)


def get_tracing_status() -> dict:
    """Get current LangSmith tracing status.

    Returns:
        Dictionary with tracing configuration status.
    """
    return {
        "enabled": settings.langsmith_enabled,
        "project": settings.langchain_project,
        "api_key_configured": bool(settings.langchain_api_key),
        "tracing_v2_env": os.environ.get("LANGCHAIN_TRACING_V2", "false"),
    }
