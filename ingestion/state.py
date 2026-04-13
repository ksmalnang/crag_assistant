"""LangGraph state definition for the ingestion subgraph."""

from __future__ import annotations

import operator
from typing import Annotated, Any, List, Optional, TypedDict

from metadata.chunk_metadata import ChunkMetadata
from parsing.docling_parser import ParsedDocument


class ErrorEntry(TypedDict):
    """Represents an error that occurred during ingestion."""

    node: str
    reason: str
    message: str
    file_path: str


class IngestionState(TypedDict):
    """
    LangGraph state for the ingestion subgraph.

    Flows through: intake -> parser -> metadata_resolver -> chunker -> embedding -> upsert -> health_check
    """

    # Run context
    run_id: str
    file_path: str
    document_id: str

    # Parsed document (set by parser_node)
    docling_doc: Optional[ParsedDocument]

    # Structure tree (extracted from docling_doc)
    structure_tree: list

    # Metadata (set by metadata_resolver_node)
    metadata: dict[str, Any]

    # Chunks (set by chunker_node)
    chunks: List[ChunkMetadata]

    # Embeddings (set by embedding_node)
    dense_vectors: list[list[float]]
    sparse_vectors: list[dict]

    # Upsert result (set by upsert_node)
    upsert_count: int

    # Errors (accumulated via operator.add)
    errors: Annotated[List[ErrorEntry], operator.add]

    # Status tracking
    status: str  # 'pending', 'processing', 'completed', 'failed', 'skipped'
