"""Embedding quality sanity checks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from embedding.errors import (
    DegenerateVectorError,
    DenseVectorNaNError,
    DenseVectorNormError,
    SparseVectorEmptyError,
)

logger = logging.getLogger(__name__)

# Thresholds
DENSE_NORM_MIN = 0.01
DEGENERATE_ALERT_THRESHOLD = 0.01  # 1% of batch


@dataclass
class QualityCheckResult:
    """Result of embedding quality validation."""

    chunk_id: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


class EmbeddingQualityChecker:
    """Validates embedding quality before upsert to Qdrant.

    Checks for degenerate vectors (zero, NaN, Inf) and logs failure rates.
    """

    def __init__(
        self,
        dense_norm_min: float = DENSE_NORM_MIN,
        alert_threshold: float = DEGENERATE_ALERT_THRESHOLD,
    ) -> None:
        """Initialise the quality checker.

        Args:
            dense_norm_min: Minimum acceptable dense vector norm.
            alert_threshold: Fraction of failed chunks that triggers alert.
        """
        self.dense_norm_min = dense_norm_min
        self.alert_threshold = alert_threshold
        self._total_checked = 0
        self._total_failed = 0

    def check_dense(self, vector: list[float], chunk_id: str) -> QualityCheckResult:
        """Run quality checks on a dense embedding vector.

        Check 1: dense vector norm > dense_norm_min (zero vectors rejected)
        Check 2: no NaN or Inf values in dense vector

        Args:
            vector: Dense embedding vector to validate.
            chunk_id: Chunk identifier for error reporting.

        Returns:
            QualityCheckResult with pass/fail status and failure details.
        """
        failures: list[str] = []
        details: dict[str, Any] = {}

        self._total_checked += 1

        arr = np.array(vector, dtype=np.float32)

        # Check 2: NaN/Inf first (so norm check is meaningful)
        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())
        details["nan_count"] = nan_count
        details["inf_count"] = inf_count

        if nan_count > 0 or inf_count > 0:
            failures.append(f"NaN:{nan_count} Inf:{inf_count}")

        # Check 1: norm check
        norm = float(np.linalg.norm(arr))
        details["norm"] = norm
        if norm < self.dense_norm_min:
            failures.append(f"norm_too_low:{norm:.6f}")

        passed = len(failures) == 0
        if not passed:
            self._total_failed += 1

        result = QualityCheckResult(
            chunk_id=chunk_id, passed=passed, failures=failures, details=details
        )

        if not passed:
            logger.warning("Dense vector quality failed for %s: %s", chunk_id, failures)

        return result

    def check_sparse(self, sparse_vector: dict, chunk_id: str) -> QualityCheckResult:
        """Run quality checks on a sparse embedding vector.

        Check 3: sparse vector has at least 1 non-zero entry

        Args:
            sparse_vector: Sparse vector as {indices: list[int], values: list[float]}.
            chunk_id: Chunk identifier for error reporting.

        Returns:
            QualityCheckResult with pass/fail status and failure details.
        """
        failures: list[str] = []
        details: dict[str, Any] = {}

        self._total_checked += 1

        values = sparse_vector.get("values", [])
        indices = sparse_vector.get("indices", [])

        details["index_count"] = len(indices)
        details["value_count"] = len(values)

        non_zero = [v for v in values if abs(v) > 1e-9]
        details["non_zero_count"] = len(non_zero)

        if len(non_zero) < 1:
            failures.append("no_non_zero_entries")

        passed = len(failures) == 0
        if not passed:
            self._total_failed += 1

        result = QualityCheckResult(
            chunk_id=chunk_id, passed=passed, failures=failures, details=details
        )

        if not passed:
            logger.warning(
                "Sparse vector quality failed for %s: %s", chunk_id, failures
            )

        return result

    def check(
        self,
        dense_vector: list[float],
        sparse_vector: dict,
        chunk_id: str,
    ) -> QualityCheckResult:
        """Run all quality checks on both dense and sparse vectors.

        Args:
            dense_vector: Dense embedding vector.
            sparse_vector: Sparse vector as {indices, values}.
            chunk_id: Chunk identifier for error reporting.

        Returns:
            Combined QualityCheckResult.

        Raises:
            DegenerateVectorError: If any check fails (caller should catch and skip).
        """
        # Note: _total_checked is incremented by check_dense and check_sparse

        dense_result = self.check_dense(dense_vector, chunk_id)
        sparse_result = self.check_sparse(sparse_vector, chunk_id)

        all_failures = dense_result.failures + sparse_result.failures
        passed = len(all_failures) == 0

        result = QualityCheckResult(
            chunk_id=chunk_id,
            passed=passed,
            failures=all_failures,
            details={
                "dense": dense_result.details,
                "sparse": sparse_result.details,
            },
        )

        if not passed:
            # Raise typed error so caller can skip this chunk
            error = DegenerateVectorError(
                chunk_id=chunk_id,
                reason="; ".join(all_failures),
            )
            raise error

        # Check alert threshold
        if self._total_checked > 0:
            fail_rate = self._total_failed / self._total_checked
            if fail_rate > self.alert_threshold:
                logger.error(
                    "ALERT: Degenerate vector rate %.2f%% exceeds threshold %.2f%% "
                    "(%d/%d chunks failed)",
                    fail_rate * 100,
                    self.alert_threshold * 100,
                    self._total_failed,
                    self._total_checked,
                )

        return result

    def stats(self) -> dict:
        """Return quality check statistics."""
        fail_rate = (
            self._total_failed / self._total_checked if self._total_checked > 0 else 0.0
        )
        return {
            "total_checked": self._total_checked,
            "total_failed": self._total_failed,
            "failure_rate": fail_rate,
            "alert_threshold": self.alert_threshold,
            "alert_triggered": fail_rate > self.alert_threshold,
        }

    def raise_on_failure(self, dense_result: QualityCheckResult, chunk_id: str) -> None:
        """Raise typed error if dense check failed.

        Args:
            dense_result: Result from check_dense().
            chunk_id: Chunk identifier.

        Raises:
            DenseVectorNormError: If norm is too low.
            DenseVectorNaNError: If NaN or Inf present.
        """
        if dense_result.passed:
            return

        norm = dense_result.details.get("norm", 0.0)
        nan_count = dense_result.details.get("nan_count", 0)
        inf_count = dense_result.details.get("inf_count", 0)

        if norm < self.dense_norm_min:
            raise DenseVectorNormError(chunk_id=chunk_id, norm_value=norm)

        if nan_count > 0 or inf_count > 0:
            raise DenseVectorNaNError(
                chunk_id=chunk_id, nan_count=nan_count, inf_count=inf_count
            )

    def raise_on_sparse_failure(
        self, sparse_result: QualityCheckResult, chunk_id: str
    ) -> None:
        """Raise typed error if sparse check failed.

        Args:
            sparse_result: Result from check_sparse().
            chunk_id: Chunk identifier.

        Raises:
            SparseVectorEmptyError: If sparse vector has no non-zero entries.
        """
        if sparse_result.passed:
            return

        raise SparseVectorEmptyError(chunk_id=chunk_id)
