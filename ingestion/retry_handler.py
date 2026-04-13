"""Per-node retry with exponential backoff for retryable ingestion nodes.

Retryable nodes: embedding_node, upsert_node (transient failures)
Non-retryable nodes: intake_node, parser_node (deterministic failures)
"""

from __future__ import annotations

import asyncio
import logging
import time
from functools import wraps
from typing import Any, Callable

from ingestion.errors_base import IngestionError

logger = logging.getLogger(__name__)

# Retryable node names
RETRYABLE_NODES = {
    "embedding_node",
    "upsert_node",
}

DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 1.0  # seconds


def is_retryable_node(node_name: str) -> bool:
    """Check if a node is retryable (transient failure possible)."""
    return node_name in RETRYABLE_NODES


async def retry_with_backoff(
    func: Callable[..., Any],
    node_name: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    **kwargs: Any,
) -> Any:
    """Execute a function with exponential backoff retry."""
    if not is_retryable_node(node_name):
        result = func(**kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            result = func(**kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                backoff = backoff_base * (2**attempt)
                logger.warning(
                    f"Node '{node_name}' failed (attempt {attempt + 1}/{max_retries + 1}), "
                    f"retrying in {backoff:.1f}s: {e}"
                )
                await asyncio.sleep(backoff)

    if isinstance(last_error, IngestionError):
        raise last_error
    raise IngestionError(
        node=node_name,
        reason=f"retry_exhausted:{type(last_error).__name__}",
        message=str(last_error),
        details={"attempts": max_retries + 1, "last_error": str(last_error)},
    )


def retryable_node(  # noqa: C901
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
):
    """Decorator to add retry logic to an ingestion node function."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:  # noqa: C901
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            node_name = func.__name__
            if not is_retryable_node(node_name):
                return await func(*args, **kwargs)

            last_error: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        backoff = backoff_base * (2**attempt)
                        logger.warning(
                            f"Node '{node_name}' failed (attempt {attempt + 1}/{max_retries + 1}), "
                            f"retrying in {backoff:.1f}s: {e}"
                        )
                        await asyncio.sleep(backoff)

            if isinstance(last_error, IngestionError):
                raise last_error
            raise IngestionError(
                node=node_name,
                reason=f"retry_exhausted:{type(last_error).__name__}",
                message=str(last_error),
                details={"attempts": max_retries + 1, "last_error": str(last_error)},
            )

        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            node_name = func.__name__
            if not is_retryable_node(node_name):
                return func(*args, **kwargs)

            last_error: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        backoff = backoff_base * (2**attempt)
                        logger.warning(
                            f"Node '{node_name}' failed (attempt {attempt + 1}/{max_retries + 1}), "
                            f"retrying in {backoff:.1f}s: {e}"
                        )
                        time.sleep(backoff)

            if isinstance(last_error, IngestionError):
                raise last_error
            raise IngestionError(
                node=node_name,
                reason=f"retry_exhausted:{type(last_error).__name__}",
                message=str(last_error),
                details={"attempts": max_retries + 1, "last_error": str(last_error)},
            )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
