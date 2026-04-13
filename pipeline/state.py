"""LangGraph base state schema for the CRAG pipeline."""

import operator
from typing import Annotated, Any, Dict, List, Optional

from typing_extensions import TypedDict


class RAGState(TypedDict):
    """Base state schema for the RAG pipeline.

    Fields:
        query: Original user query
        rewritten_query: Rewritten/expanded query (if applicable)
        intent: Detected intent classification
        retrieved_docs: List of retrieved documents
        relevance_scores: Relevance scores for retrieved documents
        generation: Generated response from LLM
        retry_count: Number of retry attempts
        citations: Citations/sources for the response
    """

    # Query and intent
    query: str
    rewritten_query: Optional[str]
    intent: Optional[str]

    # Retrieval
    retrieved_docs: Annotated[List[Dict[str, Any]], operator.add]
    relevance_scores: Annotated[List[float], operator.add]

    # Generation
    generation: Optional[str]

    # Control flow
    retry_count: int

    # Output
    citations: Annotated[List[Dict[str, Any]], operator.add]


# State schema version for tracking changes
STATE_SCHEMA_VERSION: str = "1.0.0"


def create_initial_state(query: str) -> RAGState:
    """Create a new RAG state with default values.

    Args:
        query: The user's query string

    Returns:
        Initialized RAGState with default values
    """
    return RAGState(
        query=query,
        rewritten_query=None,
        intent=None,
        retrieved_docs=[],
        relevance_scores=[],
        generation=None,
        retry_count=0,
        citations=[],
    )
