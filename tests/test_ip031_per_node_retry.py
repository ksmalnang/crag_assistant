"""Tests for IP-031: Per-node retry with exponential backoff."""

import pytest

from ingestion.errors_base import IngestionError
from ingestion.retry_handler import (
    is_retryable_node,
    retry_with_backoff,
    retryable_node,
)

# ─── is_retryable_node Tests ─────────────────────────────────────────────────


class TestIsRetryableNode:
    """Tests for node retryability classification."""

    def test_embedding_node_is_retryable(self):
        """embedding_node should be retryable."""
        assert is_retryable_node("embedding_node") is True

    def test_upsert_node_is_retryable(self):
        """upsert_node should be retryable."""
        assert is_retryable_node("upsert_node") is True

    def test_intake_node_is_not_retryable(self):
        """intake_node should not be retryable."""
        assert is_retryable_node("intake_node") is False

    def test_parser_node_is_not_retryable(self):
        """parser_node should not be retryable."""
        assert is_retryable_node("parser_node") is False

    def test_chunker_node_is_not_retryable(self):
        """chunker_node should not be retryable."""
        assert is_retryable_node("chunker_node") is False

    def test_metadata_resolver_node_is_not_retryable(self):
        """metadata_resolver_node should not be retryable."""
        assert is_retryable_node("metadata_resolver_node") is False


# ─── retry_with_backoff Tests ────────────────────────────────────────────────


class TestRetryWithBackoff:
    """Tests for the retry_with_backoff function."""

    @pytest.mark.asyncio
    async def test_retryable_node_succeeds_on_first_try(self):
        """Should succeed immediately if function doesn't fail."""
        call_count = 0

        def success_func(**kwargs):
            nonlocal call_count
            call_count += 1
            return "success"

        result = await retry_with_backoff(
            success_func,
            node_name="embedding_node",
            max_retries=3,
            backoff_base=0.01,
        )

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retryable_node_retries_on_failure(self):
        """Should retry the specified number of times before failing."""
        call_count = 0

        def failing_func(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient failure")
            return "success"

        result = await retry_with_backoff(
            failing_func,
            node_name="embedding_node",
            max_retries=3,
            backoff_base=0.01,
        )

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_node_fails_immediately(self):
        """Non-retryable nodes should not retry."""
        call_count = 0

        def failing_func(**kwargs):
            nonlocal call_count
            call_count += 1
            raise ValueError("Deterministic failure")

        with pytest.raises(ValueError, match="Deterministic failure"):
            await retry_with_backoff(
                failing_func,
                node_name="intake_node",
                max_retries=3,
                backoff_base=0.01,
            )

        assert call_count == 1  # Only called once, no retries

    @pytest.mark.asyncio
    async def test_max_retries_exhausted_raises_ingestion_error(self):
        """After max retries, should raise IngestionError."""

        def always_fails(**kwargs):
            raise ConnectionError("Always fails")

        with pytest.raises(IngestionError) as exc_info:
            await retry_with_backoff(
                always_fails,
                node_name="upsert_node",
                max_retries=2,
                backoff_base=0.01,
            )

        assert "retry_exhausted" in exc_info.value.reason
        assert exc_info.value.details["attempts"] == 3

    @pytest.mark.asyncio
    async def test_backoff_increases_exponentially(self):
        """Backoff should increase exponentially: 1s → 2s → 4s."""
        call_times = []

        def failing_func(**kwargs):
            import time

            call_times.append(time.time())
            raise ConnectionError("Failing")

        with pytest.raises(IngestionError):
            await retry_with_backoff(
                failing_func,
                node_name="embedding_node",
                max_retries=3,
                backoff_base=0.05,  # 50ms base for faster tests
            )

        # Verify increasing gaps between calls
        assert len(call_times) == 4  # Initial + 3 retries
        gaps = [call_times[i + 1] - call_times[i] for i in range(len(call_times) - 1)]
        assert gaps[0] < gaps[1] < gaps[2]  # Exponential increase


# ─── retryable_node Decorator Tests ──────────────────────────────────────────


class TestRetryableNodeDecorator:
    """Tests for the @retryable_node decorator."""

    @pytest.mark.asyncio
    async def test_decorator_retries_on_retryable_node(self):
        """Decorator should retry functions on retryable nodes."""
        call_count = 0

        @retryable_node(max_retries=2, backoff_base=0.01)
        async def embedding_node():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient")
            return "done"

        result = await embedding_node()
        assert result == "done"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_decorator_no_retry_on_non_retryable_node(self):
        """Decorator should not retry on non-retryable nodes."""
        call_count = 0

        @retryable_node(max_retries=3, backoff_base=0.01)
        def failing_sync_node():
            nonlocal call_count
            call_count += 1
            raise ValueError("Deterministic")

        with pytest.raises(ValueError, match="Deterministic"):
            failing_sync_node()

        assert call_count == 1  # No retries
