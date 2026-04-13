"""Typed error hierarchy for embedding validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EmbeddingError(Exception):
    """Base error for embedding-related failures."""

    chunk_id: str
    reason: str
    message: Optional[str] = None

    def __post_init__(self) -> None:
        if self.message is None:
            self.message = f"Embedding failed for chunk {self.chunk_id}: {self.reason}"
        super().__init__(self.message)


@dataclass
class DegenerateVectorError(EmbeddingError):
    """Error raised when embedding produces a degenerate vector (zero, NaN, Inf)."""

    def __init__(
        self,
        chunk_id: str,
        reason: str = "degenerate_vector",
        message: Optional[str] = None,
    ) -> None:
        super().__init__(
            chunk_id=chunk_id,
            reason=reason,
            message=message,
        )


@dataclass
class DenseVectorNormError(DegenerateVectorError):
    """Error raised when dense vector norm is too small (near-zero vector)."""

    def __init__(
        self,
        chunk_id: str,
        norm_value: float,
        message: Optional[str] = None,
    ) -> None:
        super().__init__(
            chunk_id=chunk_id,
            reason=f"dense_vector_norm_too_low:{norm_value}",
            message=message
            or f"Dense vector norm {norm_value:.6f} below threshold for chunk {chunk_id}",
        )


@dataclass
class DenseVectorNaNError(DegenerateVectorError):
    """Error raised when dense vector contains NaN or Inf values."""

    def __init__(
        self,
        chunk_id: str,
        nan_count: int = 0,
        inf_count: int = 0,
        message: Optional[str] = None,
    ) -> None:
        issues = []
        if nan_count:
            issues.append(f"{nan_count} NaN")
        if inf_count:
            issues.append(f"{inf_count} Inf")
        super().__init__(
            chunk_id=chunk_id,
            reason=f"dense_vector_invalid:{','.join(issues)}",
            message=message
            or f"Dense vector contains {' and '.join(issues)} for chunk {chunk_id}",
        )


@dataclass
class SparseVectorEmptyError(DegenerateVectorError):
    """Error raised when sparse vector has no non-zero entries."""

    def __init__(
        self,
        chunk_id: str,
        message: Optional[str] = None,
    ) -> None:
        super().__init__(
            chunk_id=chunk_id,
            reason="sparse_vector_empty",
            message=message
            or f"Sparse vector has no non-zero entries for chunk {chunk_id}",
        )
