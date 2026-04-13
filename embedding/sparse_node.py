"""Sparse embedding node using BM25 via fastembed."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

from fastembed.sparse.bm25 import Bm25

logger = logging.getLogger(__name__)

DEFAULT_VOCAB_DIR = "data/bm25_vocab"


class SparseEmbeddingNode:
    """Generates sparse embeddings using BM25 via fastembed.

    The BM25 model is initialised at ingestion start. Corpus vocabulary
    (document-level IDF statistics) is persisted for reuse on incremental
    ingestions to avoid re-initialising the model.
    """

    def __init__(
        self,
        model_name: str = "Qdrant/bm25",
        vocab_dir: str = DEFAULT_VOCAB_DIR,
        collection_name: Optional[str] = None,
    ) -> None:
        """Initialise the sparse embedding node.

        Args:
            model_name: Fastembed BM25 model identifier.
            vocab_dir: Directory to store persisted vocabular.
            collection_name: Name of the collection for vocab persistence.
        """
        self.model_name = model_name
        self.vocab_dir = Path(vocab_dir)
        self.collection_name = collection_name
        self._model: Optional[Bm25] = None
        self._is_fitted = False

    def _vocab_path(self) -> Path:
        """Get the path where the vocabulary is persisted."""
        if not self.collection_name:
            raise ValueError("collection_name required for vocab persistence")
        self.vocab_dir.mkdir(parents=True, exist_ok=True)
        return self.vocab_dir / f"{self.collection_name}.pkl"

    def _load_model(self) -> Bm25:
        """Load the BM25 model."""
        if self._model is not None:
            return self._model
        logger.info("Loading sparse embedding model: %s", self.model_name)
        self._model = Bm25(self.model_name)
        logger.info("Sparse embedding model loaded: %s", self.model_name)
        return self._model

    def fit(self, corpus_texts: list[str], force: bool = False) -> None:
        """Prepare the BM25 model, using persisted corpus vocabulary if available.

        The BM25 model in fastembed is pre-trained. We persist the corpus texts
        used for IDF warmup so the model can be re-initialised identically across
        ingestion runs.

        Args:
            corpus_texts: All chunk texts in the current ingestion batch.
                          Used to warm up IDF statistics.
            force: Force re-initialisation even if persisted corpus exists.
        """
        vocab_path = self._vocab_path()

        if not force and vocab_path.exists():
            logger.info("Loading persisted BM25 corpus from %s", vocab_path)
            with open(vocab_path, "rb") as f:
                corpus_data = pickle.load(f)
            saved_texts = corpus_data.get("texts", [])
            if saved_texts:
                model = self._load_model()
                model.embed(saved_texts)
                self._model = model
                self._is_fitted = True
                logger.info(
                    "BM25 model re-initialised from %d persisted texts",
                    len(saved_texts),
                )
                return

        logger.info("Initialising BM25 model on %d corpus documents", len(corpus_texts))
        model = self._load_model()

        # Warm up the model by embedding the corpus so IDF stats are populated
        model.embed(corpus_texts)
        self._model = model
        self._is_fitted = True

        # Persist corpus texts for re-initialisation
        with open(vocab_path, "wb") as f:
            pickle.dump({"texts": corpus_texts}, f)
        logger.info("BM25 corpus saved to %s", vocab_path)

    def embed(self, texts: list[str]) -> list[dict]:
        """Generate sparse embeddings for a list of texts.

        Args:
            texts: List of chunk texts to embed.

        Returns:
            List of sparse vectors as {indices: list[int], values: list[float]}.
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError(
                "BM25 model not fitted. Call fit() with corpus_texts first."
            )

        results: list[dict] = []
        encoded = self._model.embed(texts)

        for i, sparse_vec in enumerate(encoded):
            # Convert sparse vector to {indices, values} format
            indices = sparse_vec.indices.tolist()
            values = sparse_vec.values.tolist()

            # Convert to native Python types (from numpy)
            indices = [int(idx) for idx in indices]
            values = [float(v) for v in values]

            # Sanity: at least 1 non-zero entry
            non_zero = [v for v in values if abs(v) > 1e-9]
            if not non_zero:
                logger.warning(
                    "Sparse vector for text %d has no significant entries", i
                )

            results.append({"indices": indices, "values": values})

        return results

    def embed_single(self, text: str) -> dict:
        """Generate a sparse embedding for a single chunk.

        Args:
            text: Chunk text to embed.

        Returns:
            Sparse vector as {indices: list[int], values: list[float]}.
        """
        results = self.embed([text])
        return results[0]
