"""Post-upsert index health check."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

from pipeline.config import settings

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Result of post-upsert health check."""

    collection_name: str
    status: str  # "green", "degraded", "unhealthy"
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        """Check if health check passed completely."""
        return self.status == "green"

    @property
    def is_degraded(self) -> bool:
        """Check if health check passed with warnings."""
        return self.status == "degraded"


class UpsertHealthChecker:
    """Verifies Qdrant collection health after upsert.

    Runs test queries and checks collection status.
    Failures mark ingestion as 'degraded' (not failed — data was written).
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: Optional[str] = None,
        dense_dim: Optional[int] = None,
    ) -> None:
        """Initialise the health checker.

        Args:
            client: Qdrant client instance.
            collection_name: Collection name (defaults to settings).
            dense_dim: Dense vector dimension for test queries
                       (defaults to settings).
        """
        self._client = client
        self.collection_name = collection_name or settings.qdrant_collection
        self.dense_dim = dense_dim or settings.qdrant_dense_dim

    def check(
        self,
        expected_count: int,
        test_document_id: Optional[str] = None,
    ) -> HealthCheckResult:
        """Run all post-upsert health checks.

        Args:
            expected_count: Expected point count after upsert.
            test_document_id: Document ID to test query with.
                              If None, tests against all points.

        Returns:
            HealthCheckResult with pass/fail status and details.
        """
        result = HealthCheckResult(
            collection_name=self.collection_name,
            status="green",
        )

        # Check 1: Collection status is accessible
        self._check_collection_status(result)

        # Check 2: Vector count matches expected
        self._check_vector_count(result, expected_count)

        # Check 3: Test hybrid query returns results
        if test_document_id:
            self._check_test_query(result, test_document_id)

        # Determine overall status
        if result.checks_failed:
            result.status = "degraded"
            logger.warning(
                "Health check degraded for '%s': failed=%s",
                self.collection_name,
                result.checks_failed,
            )

        return result

    def _check_collection_status(self, result: HealthCheckResult) -> None:
        """Check collection is accessible and get status info."""
        try:
            info = self._client.get_collection(self.collection_name)
            status = info.status
            result.details["collection_status"] = status

            if str(status).lower() in ("green", "yellow", "active"):
                result.checks_passed.append("collection_accessible")
                result.details["vectors_count"] = info.vectors_count
                result.details["indexed_vectors_count"] = info.indexed_vectors_count
                result.details["points_count"] = info.points_count
            else:
                result.checks_failed.append("collection_not_green")
                result.issues.append(f"Collection status: {status}")

        except Exception as exc:
            result.checks_failed.append("collection_inaccessible")
            result.issues.append(str(exc))

    def _check_vector_count(self, result: HealthCheckResult, expected: int) -> None:
        """Check indexed vectors count matches expected count."""
        try:
            info = self._client.get_collection(self.collection_name)
            actual = info.points_count
            result.details["expected_count"] = expected
            result.details["actual_count"] = actual

            if actual >= expected:
                result.checks_passed.append("count_verified")
            else:
                shortfall = expected - actual
                result.checks_failed.append("count_mismatch")
                result.issues.append(
                    f"Expected {expected} points, got {actual} (shortfall: {shortfall})"
                )

        except Exception as exc:
            result.checks_failed.append("count_check_failed")
            result.issues.append(str(exc))

    def _check_test_query(self, result: HealthCheckResult, document_id: str) -> None:
        """Issue a test hybrid query and assert results > 0."""
        try:
            # Build a test hybrid query filtering by document_id
            dummy_vector = [0.0] * self.dense_dim

            filter_obj = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            )

            # Simple search to verify points are queryable
            search_results = self._client.query_points(
                collection_name=self.collection_name,
                query=dummy_vector,
                using="dense",
                query_filter=filter_obj,
                limit=1,
            )

            hits = (
                len(search_results.points) if hasattr(search_results, "points") else 0
            )
            result.details["test_query_hits"] = hits

            if hits > 0:
                result.checks_passed.append("test_query_ok")
            else:
                result.checks_failed.append("test_query_no_hits")
                result.issues.append(
                    f"Test query for document '{document_id}' returned 0 hits"
                )

        except Exception:
            # Try fallback with simple search
            try:
                dummy_vector = [0.0] * self.dense_dim
                search_results = self._client.search(
                    collection_name=self.collection_name,
                    query_vector=("dense", dummy_vector),
                    query_filter=Filter(
                        must=[
                            FieldCondition(
                                key="document_id",
                                match=MatchValue(value=document_id),
                            )
                        ]
                    ),
                    limit=1,
                )
                hits = len(search_results) if search_results else 0
                result.details["test_query_hits"] = hits

                if hits > 0:
                    result.checks_passed.append("test_query_ok")
                else:
                    result.checks_failed.append("test_query_no_hits")
                    result.issues.append(
                        f"Test query for document '{document_id}' returned 0 hits"
                    )
            except Exception as exc2:
                result.checks_failed.append("test_query_failed")
                result.issues.append(f"Query error: {exc2}")

    def record_in_manifest(self, result: HealthCheckResult, manifest: dict) -> dict:
        """Record health check result in run manifest.

        Args:
            result: HealthCheckResult to record.
            manifest: Run manifest dictionary (modified in place).

        Returns:
            Updated manifest dictionary.
        """
        manifest["health_check"] = {
            "status": result.status,
            "collection": result.collection_name,
            "passed": result.checks_passed,
            "failed": result.checks_failed,
            "issues": result.issues,
            "details": result.details,
        }
        return manifest
