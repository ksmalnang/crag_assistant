"""Embedding package for document vector generation."""

from .cache import EmbeddingCache
from .dense_node import DenseEmbeddingNode
from .errors import (
    DegenerateVectorError,
    DenseVectorNaNError,
    DenseVectorNormError,
    EmbeddingError,
    SparseVectorEmptyError,
)
from .quality import EmbeddingQualityChecker, QualityCheckResult
from .sparse_node import SparseEmbeddingNode

__all__ = [
    # Dense embedding
    "DenseEmbeddingNode",
    # Sparse embedding
    "SparseEmbeddingNode",
    # Cache
    "EmbeddingCache",
    # Quality
    "EmbeddingQualityChecker",
    "QualityCheckResult",
    # Errors
    "EmbeddingError",
    "DegenerateVectorError",
    "DenseVectorNormError",
    "DenseVectorNaNError",
    "SparseVectorEmptyError",
]
