"""Dense embedding node using qwen/qwen3-embedding-8b via OpenRouter."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from openai import OpenAI

from embedding.errors import DenseVectorNaNError, DenseVectorNormError

logger = logging.getLogger(__name__)

DENSE_DIM = 4096
DEFAULT_BATCH_SIZE = 32
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3-embedding-8b"


class DenseEmbeddingNode:
    """Generates dense embeddings using qwen/qwen3-embedding-8b via OpenRouter.

    Uses the OpenAI-compatible API endpoint. Model is initialised once and
    reused across all chunks. No fallback.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        openrouter_api_key: Optional[str] = None,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
    ) -> None:
        """Initialise the dense embedding node.

        Args:
            model_name: OpenRouter model identifier.
            batch_size: Number of chunks to embed per batch.
            openrouter_api_key: OpenRouter API key. Defaults to OPENROUTER_API_KEY env var.
            base_url: OpenRouter API base URL.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.base_url = base_url
        self._client: Optional[OpenAI] = None
        # Store explicitly so _get_client can pick it up from env if not passed
        self._explicit_api_key = openrouter_api_key

    def _get_client(self) -> OpenAI:
        """Get or create OpenAI client pointing to OpenRouter."""
        if self._client is None:
            logger.info(
                "Initialising OpenRouter client (model=%s, base_url=%s)",
                self.model_name,
                self.base_url,
            )
            kwargs: dict = {"base_url": self.base_url}
            if self._explicit_api_key:
                kwargs["api_key"] = self._explicit_api_key
            self._client = OpenAI(**kwargs)
            logger.info("OpenRouter client initialised")
        return self._client

    def embed(
        self,
        texts: list[str],
        chunk_ids: Optional[list[str]] = None,
    ) -> list[list[float]]:
        """Generate dense embeddings for a list of texts via OpenRouter.

        Input: chunk texts only — metadata fields never included in embedding input.
        Output: validated 4096-dimensional vectors.

        Args:
            texts: List of chunk texts to embed.
            chunk_ids: Optional list of chunk IDs for error reporting.
                       Auto-generated if not provided.

        Returns:
            List of dense vectors (each 4096-dimensional).

        Raises:
            DenseVectorNormError: If any vector norm is below threshold.
            DenseVectorNaNError: If any vector contains NaN or Inf.
        """
        if not texts:
            return []

        if chunk_ids is None:
            chunk_ids = [f"chunk-{i}" for i in range(len(texts))]

        if len(texts) != len(chunk_ids):
            raise ValueError("texts and chunk_ids must have the same length")

        client = self._get_client()
        results: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_chunk_ids = chunk_ids[i : i + self.batch_size]

            response = client.embeddings.create(
                input=batch_texts,
                model=self.model_name,
            )

            for data in response.data:
                vec = data.embedding
                vec_np = np.array(vec, dtype=np.float32)

                # NaN/Inf check
                nan_count = int(np.isnan(vec_np).sum())
                inf_count = int(np.isinf(vec_np).sum())
                if nan_count > 0 or inf_count > 0:
                    raise DenseVectorNaNError(
                        chunk_id=batch_chunk_ids[response.data.index(data)],
                        nan_count=nan_count,
                        inf_count=inf_count,
                    )

                # Norm check
                norm = float(np.linalg.norm(vec_np))
                if norm < 0.01:
                    raise DenseVectorNormError(
                        chunk_id=batch_chunk_ids[response.data.index(data)],
                        norm_value=norm,
                    )

                results.append(vec)

        logger.info(
            "Embedded %d chunks via OpenRouter (%s)", len(results), self.model_name
        )
        return results

    def embed_single(self, text: str, chunk_id: str = "single") -> list[float]:
        """Generate a dense embedding for a single chunk.

        Args:
            text: Chunk text to embed.
            chunk_id: Chunk ID for error reporting.

        Returns:
            4096-dimensional dense vector.
        """
        results = self.embed([text], [chunk_id])
        return results[0]

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity in [-1, 1].
        """
        a_arr = np.array(a, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)
        dot = float(np.dot(a_arr, b_arr))
        norm_a = float(np.linalg.norm(a_arr))
        norm_b = float(np.linalg.norm(b_arr))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
