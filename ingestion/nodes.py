"""LangGraph nodes for the ingestion subgraph."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from qdrant_client import QdrantClient

from chunking.chunkers import (
    ChunkingConfig,
    TableAwareChunker,
)
from embedding.cache import EmbeddingCache
from embedding.dense_node import DenseEmbeddingNode
from embedding.sparse_node import SparseEmbeddingNode
from ingestion.errors import (
    EmptyFileError,
    FileTooLargeError,
    IntakeError,
    UnsupportedFormatError,
)
from ingestion.formats import is_supported_extension
from ingestion.preflight import PreflightValidator
from ingestion.state import ErrorEntry, IngestionState
from metadata.resolver import FilenameMetadata
from parsing.docling_parser import DoclingParser
from parsing.errors import ParseError as DocParseError
from pipeline.config import settings
from vector_store.deletion import StaleDeletionNode
from vector_store.health_check import UpsertHealthChecker
from vector_store.upsert import QdrantUpsertNode

logger = logging.getLogger(__name__)


def _make_error(node: str, reason: str, message: str, file_path: str) -> ErrorEntry:
    """Helper to create an ErrorEntry dict for the LangGraph state."""
    return ErrorEntry(
        node=node,
        reason=reason,
        message=message,
        file_path=file_path,
    )


# ─── Intake Node ───────────────────────────────────────────────────────────────


def intake_node(state: IngestionState) -> dict:
    """
    Intake node: validates file and prepares for parsing.

    Checks:
    - File exists and is readable
    - Supported format
    - File size within limits
    - File is not empty
    - File is not corrupted/encrypted

    Returns updated state or routes to error handler.
    """
    file_path = state["file_path"]
    run_id = state["run_id"]
    errors = []

    logger.info(f"[run:{run_id}] Intake node processing: {file_path}")

    try:
        path = Path(file_path)

        # Check file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check extension
        if not is_supported_extension(file_path):
            raise UnsupportedFormatError(file_path=file_path)

        # Read file bytes for preflight and document_id
        file_bytes = path.read_bytes()

        # Check empty
        if len(file_bytes) == 0:
            raise EmptyFileError(file_path=file_path)

        # Check size
        if len(file_bytes) > settings.max_file_size_bytes:
            raise FileTooLargeError(file_path=file_path)

        # Run preflight validation
        validator = PreflightValidator()
        validator.validate(file_path)

        # Compute document_id from file bytes (SHA256)
        document_id = hashlib.sha256(file_bytes).hexdigest()

        logger.info(
            f"[run:{run_id}] Intake passed for {file_path}, document_id={document_id[:12]}..."
        )

        return {
            "document_id": document_id,
            "status": "processing",
            "errors": errors,
        }

    except IntakeError as e:
        errors.append(
            _make_error(
                node="intake_node",
                reason=e.reason,
                message=e.message or str(e),
                file_path=file_path,
            )
        )
        logger.error(f"[run:{run_id}] Intake error: {e}")
        return {"errors": errors, "status": "failed"}

    except Exception as e:
        errors.append(
            _make_error(
                node="intake_node",
                reason="unexpected_error",
                message=str(e),
                file_path=file_path,
            )
        )
        logger.error(f"[run:{run_id}] Unexpected intake error: {e}")
        return {"errors": errors, "status": "failed"}


# ─── Parser Node ───────────────────────────────────────────────────────────────


async def parser_node(state: IngestionState) -> dict:
    """
    Parser node: parses the document using Docling.

    Output: ParsedDocument with structure tree, tables, page images, OCR metadata.
    """
    file_path = state["file_path"]
    document_id = state["document_id"]
    run_id = state["run_id"]
    errors = list(state.get("errors", []))

    logger.info(f"[run:{run_id}] Parser node processing: {file_path}")

    try:
        parser = DoclingParser()
        parsed_doc = await parser.parse_file(
            file_path=file_path,
            document_id=document_id,
        )

        logger.info(
            f"[run:{run_id}] Parser completed: {parsed_doc.page_count} pages, "
            f"{len(parsed_doc.structure_tree)} headings, "
            f"{len(parsed_doc.tables)} tables"
        )

        return {
            "docling_doc": parsed_doc,
            "structure_tree": [
                {
                    "level": n.level,
                    "title": n.title,
                    "page_start": n.page_start,
                    "char_start": n.char_start,
                }
                for n in parsed_doc.structure_tree
            ],
            "errors": errors,
        }

    except DocParseError as e:
        errors.append(
            _make_error(
                node="parser_node",
                reason=e.reason,
                message=e.message or str(e),
                file_path=file_path,
            )
        )
        logger.error(f"[run:{run_id}] Parse error: {e}")
        return {"errors": errors, "status": "failed"}

    except Exception as e:
        errors.append(
            _make_error(
                node="parser_node",
                reason="unexpected_error",
                message=str(e),
                file_path=file_path,
            )
        )
        logger.error(f"[run:{run_id}] Unexpected parse error: {e}")
        return {"errors": errors, "status": "failed"}


# ─── Metadata Resolver Node ────────────────────────────────────────────────────


def metadata_resolver_node(state: IngestionState) -> dict:
    """
    Metadata resolver node: extracts metadata from filename convention.

    Resolves: faculty, semester, doc_type, display_name, source_name.
    If filename is unclassified, continues with warning (not abort).
    """
    file_path = state["file_path"]
    run_id = state["run_id"]
    errors = list(state.get("errors", []))

    logger.info(f"[run:{run_id}] Metadata resolver processing: {file_path}")

    try:
        # Read file bytes for document_id computation consistency
        file_bytes = Path(file_path).read_bytes()
        filename_meta = FilenameMetadata.from_filename(
            file_path=file_path,
            file_bytes=file_bytes,
        )

        metadata = {
            "source_path": str(file_path),
            "source_name": filename_meta.source_name,
            "faculty": filename_meta.faculty.value,
            "doc_type": filename_meta.doc_type.value,
            "semester": filename_meta.semester,
            "document_id": filename_meta.document_id,
            "unclassified": filename_meta.unclassified,
        }

        if filename_meta.unclassified:
            logger.warning(
                f"[run:{run_id}] File '{file_path}' has unclassified filename. "
                f"Using fallback metadata."
            )
            # Add warning but continue — don't abort
            errors.append(
                _make_error(
                    node="metadata_resolver_node",
                    reason="unclassified_filename",
                    message=f"Filename '{Path(file_path).stem}' is malformed, using fallback values",
                    file_path=file_path,
                )
            )

        logger.info(
            f"[run:{run_id}] Metadata resolved: faculty={metadata['faculty']}, "
            f"doc_type={metadata['doc_type']}, semester={metadata['semester']}"
        )

        return {
            "metadata": metadata,
            "errors": errors,
        }

    except Exception as e:
        errors.append(
            _make_error(
                node="metadata_resolver_node",
                reason="metadata_resolution_failed",
                message=str(e),
                file_path=file_path,
            )
        )
        logger.error(f"[run:{run_id}] Metadata resolution error: {e}")
        return {"errors": errors, "status": "failed"}


# ─── Chunker Node ──────────────────────────────────────────────────────────────


def chunker_node(state: IngestionState) -> dict:
    """
    Chunker node: splits parsed document into chunks.

    Uses HeadingHierarchyChunker if document has headings,
    falls back to SlidingWindowChunker otherwise.
    Uses TableAwareChunker for table-aware splitting.
    """
    run_id = state["run_id"]
    file_path = state["file_path"]
    parsed_doc = state.get("docling_doc")
    errors = list(state.get("errors", []))

    logger.info(f"[run:{run_id}] Chunker node processing")

    if parsed_doc is None:
        errors.append(
            _make_error(
                node="chunker_node",
                reason="no_parsed_document",
                message="Cannot chunk: no parsed document available",
                file_path=file_path,
            )
        )
        return {"errors": errors, "status": "failed"}

    try:
        config = ChunkingConfig()

        # Use table-aware chunker which delegates to heading or sliding window
        chunker = TableAwareChunker(config)
        chunks = chunker.chunk(parsed_doc)

        logger.info(f"[run:{run_id}] Chunker created {len(chunks)} chunks")

        return {
            "chunks": chunks,
            "errors": errors,
        }

    except Exception as e:
        errors.append(
            _make_error(
                node="chunker_node",
                reason="chunking_failed",
                message=str(e),
                file_path=file_path,
            )
        )
        logger.error(f"[run:{run_id}] Chunking error: {e}")
        return {"errors": errors, "status": "failed"}


# ─── Embedding Node ────────────────────────────────────────────────────────────


def _extract_texts_from_chunks(
    chunks: list,
    parsed_doc,
) -> tuple[list[str], list[str]]:
    """Extract text content from chunks using parsed document char offsets."""
    full_text = parsed_doc.text_content
    texts = []
    chunk_ids = []

    for chunk in chunks:
        if chunk.char_start >= 0 and chunk.char_end > chunk.char_start:
            text = full_text[chunk.char_start : chunk.char_end]
        else:
            text = " > ".join(chunk.heading_path) if chunk.heading_path else ""

        if text.strip():
            texts.append(text)
            chunk_ids.append(f"{chunk.document_id}:{chunk.chunk_index}")

    return texts, chunk_ids


def _generate_dense_embeddings(
    texts: list[str],
) -> tuple[list[list[float]], int]:
    """Generate dense embeddings with cache support."""
    cache = EmbeddingCache()
    dense_vectors: list[list[float] | None] = []
    cache_hits = 0

    import struct

    import numpy as np

    dense_node = DenseEmbeddingNode()
    uncached_texts = []
    uncached_indices = []

    for i, text in enumerate(texts):
        cached = cache.get(text)
        if cached and cached["dense_vector"]:
            raw = cached["dense_vector"]
            vec = list(struct.unpack(f"{len(raw) // 4}f", raw))
            dense_vectors.append(vec)
            cache_hits += 1
        else:
            uncached_texts.append(text)
            uncached_indices.append(i)
            dense_vectors.append(None)

    if uncached_texts:
        new_dense = dense_node.embed(
            uncached_texts, [f"chunk-{i}" for i in uncached_indices]
        )
        for idx, vec in zip(uncached_indices, new_dense, strict=False):
            dense_vectors[idx] = vec  # type: ignore[arg-type]
            vec_arr = np.array(vec, dtype=np.float32)
            cache.put(text, vec_arr, None, None)

    return dense_vectors, cache_hits  # type: ignore[return-value]


def _generate_sparse_embeddings(texts: list[str]) -> list[dict]:
    """Generate sparse embeddings using BM25."""
    sparse_node = SparseEmbeddingNode(
        collection_name=settings.qdrant_collection,
    )
    sparse_node.fit(texts, force=False)
    return sparse_node.embed(texts)


def embedding_node(state: IngestionState) -> dict:
    """
    Embedding node: generates dense and sparse embeddings for all chunks.

    Uses embedding cache to skip re-computing unchanged chunks.
    """
    run_id = state["run_id"]
    file_path = state["file_path"]
    chunks = state.get("chunks", [])
    errors = list(state.get("errors", []))

    logger.info(f"[run:{run_id}] Embedding node processing {len(chunks)} chunks")

    if not chunks:
        errors.append(
            _make_error(
                node="embedding_node",
                reason="no_chunks_to_embed",
                message="No chunks available for embedding",
                file_path=file_path,
            )
        )
        return {"errors": errors, "status": "failed"}

    try:
        parsed_doc = state.get("docling_doc")
        if parsed_doc is None:
            errors.append(
                _make_error(
                    node="embedding_node",
                    reason="no_parsed_document_for_text",
                    message="Cannot embed: no parsed document for text extraction",
                    file_path=file_path,
                )
            )
            return {"errors": errors, "status": "failed"}

        texts, chunk_ids = _extract_texts_from_chunks(chunks, parsed_doc)

        if not texts:
            errors.append(
                _make_error(
                    node="embedding_node",
                    reason="no_text_content",
                    message="No text content extracted for embedding",
                    file_path=file_path,
                )
            )
            return {"errors": errors, "status": "failed"}

        dense_vectors, cache_hits = _generate_dense_embeddings(texts)
        sparse_vectors = _generate_sparse_embeddings(texts)

        logger.info(
            f"[run:{run_id}] Embedding completed: {len(dense_vectors)} dense, "
            f"{len(sparse_vectors)} sparse, {cache_hits} cache hits"
        )

        return {
            "dense_vectors": dense_vectors,
            "sparse_vectors": sparse_vectors,
            "errors": errors,
        }

    except Exception as e:
        errors.append(
            _make_error(
                node="embedding_node",
                reason="embedding_failed",
                message=str(e),
                file_path=file_path,
            )
        )
        logger.error(f"[run:{run_id}] Embedding error: {e}")
        return {"errors": errors, "status": "failed"}


# ─── Upsert Node ───────────────────────────────────────────────────────────────


def upsert_node(state: IngestionState) -> dict:
    """
    Upsert node: writes embedded chunks to Qdrant vector store.

    Handles stale document deletion before upsert.
    """
    run_id = state["run_id"]
    file_path = state["file_path"]
    document_id = state["document_id"]
    chunks = state.get("chunks", [])
    dense_vectors = state.get("dense_vectors", [])
    sparse_vectors = state.get("sparse_vectors", [])
    metadata = state.get("metadata", {})
    errors = list(state.get("errors", []))

    logger.info(f"[run:{run_id}] Upsert node processing {len(chunks)} chunks")

    if not chunks or not dense_vectors:
        errors.append(
            _make_error(
                node="upsert_node",
                reason="no_data_to_upsert",
                message="No chunks or vectors to upsert",
                file_path=file_path,
            )
        )
        return {"errors": errors, "status": "failed"}

    try:
        # Connect to Qdrant
        client = QdrantClient(url=settings.qdrant_url)

        # Stale deletion: remove old chunks for this document_id
        deletion_node = StaleDeletionNode(client=client)
        deleted = deletion_node.delete_by_document_id(document_id=document_id)
        if deleted > 0:
            logger.info(
                f"[run:{run_id}] Deleted {deleted} stale points for {document_id}"
            )

        # Build upsert tuples: (document_id, chunk_index, chunk_metadata)
        upsert_tuples = [(document_id, chunk.chunk_index, chunk) for chunk in chunks]

        # Upsert
        upsert_node_impl = QdrantUpsertNode(client=client)
        count = upsert_node_impl.upsert(
            chunks=upsert_tuples,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
            payload_extra={
                "source_path": metadata.get("source_path", file_path),
                "source_name": metadata.get("source_name", ""),
                "faculty": metadata.get("faculty", "other"),
                "doc_type": metadata.get("doc_type", "other"),
                "semester": metadata.get("semester", "unknown"),
            },
            expected_count=len(chunks),
        )

        logger.info(f"[run:{run_id}] Upserted {count} points")

        return {
            "upsert_count": count,
            "errors": errors,
        }

    except Exception as e:
        errors.append(
            _make_error(
                node="upsert_node",
                reason="upsert_failed",
                message=str(e),
                file_path=file_path,
            )
        )
        logger.error(f"[run:{run_id}] Upsert error: {e}")
        return {"errors": errors, "status": "failed"}


# ─── Health Check Node ─────────────────────────────────────────────────────────


def health_check_node(state: IngestionState) -> dict:
    """
    Health check node: verifies Qdrant collection health after upsert.
    """
    run_id = state["run_id"]
    file_path = state["file_path"]
    document_id = state["document_id"]
    upsert_count = state.get("upsert_count", 0)
    errors = list(state.get("errors", []))

    logger.info(f"[run:{run_id}] Health check node running")

    try:
        client = QdrantClient(url=settings.qdrant_url)
        checker = UpsertHealthChecker(client=client)

        result = checker.check(
            expected_count=upsert_count,
            test_document_id=document_id,
        )

        if not result.is_healthy:
            logger.warning(
                f"[run:{run_id}] Health check degraded: {result.checks_failed}"
            )
            for issue in result.issues:
                errors.append(
                    _make_error(
                        node="health_check_node",
                        reason=f"health_check_{issue}",
                        message=issue,
                        file_path=file_path,
                    )
                )

        return {
            "errors": errors,
            "status": "completed"
            if result.is_healthy
            else "completed",  # still completed even if degraded
        }

    except Exception as e:
        errors.append(
            _make_error(
                node="health_check_node",
                reason="health_check_failed",
                message=str(e),
                file_path=file_path,
            )
        )
        logger.error(f"[run:{run_id}] Health check error: {e}")
        return {
            "errors": errors,
            "status": "completed",
        }  # mark completed since data was written
