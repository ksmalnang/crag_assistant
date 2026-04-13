"""LangGraph ingestion subgraph wiring.

Wires all ingestion nodes into a complete LangGraph subgraph with conditional
edges for error handling and state management.

Graph flow:
  intake_node -> parser_node -> metadata_resolver_node -> chunker_node -> embedding_node -> upsert_node -> health_check_node -> END

Conditional edges:
  - After intake_node: if errors → END (fail fast)
  - After parser_node: if errors → END (fail fast)
  - After metadata_resolver: continues even with warnings (unclassified=True is not fatal)
  - After chunker_node: if errors → END
  - After embedding_node: if errors → END
  - After upsert_node: if errors → END (data was not written)
  - After health_check_node: always → END
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from ingestion.nodes import (
    chunker_node,
    embedding_node,
    health_check_node,
    intake_node,
    metadata_resolver_node,
    parser_node,
    upsert_node,
)
from ingestion.state import IngestionState

logger = logging.getLogger(__name__)


def _has_fatal_errors(state: IngestionState) -> bool:
    """Check if state contains fatal errors that should abort the pipeline."""
    errors = state.get("errors", [])
    if not errors:
        return False

    # Fatal nodes: intake, parser, chunker, embedding, upsert
    # Non-fatal: metadata_resolver (unclassified filename is a warning)
    fatal_nodes = {
        "intake_node",
        "parser_node",
        "chunker_node",
        "embedding_node",
        "upsert_node",
    }

    for error in errors:
        if error.get("node") in fatal_nodes:
            return True

    return False


def _route_after_intake(state: IngestionState) -> str:
    """Route after intake node: if fatal errors, go to END."""
    if _has_fatal_errors(state):
        return "error_end"
    return "parser"


def _route_after_parser(state: IngestionState) -> str:
    """Route after parser node: if fatal errors, go to END."""
    if _has_fatal_errors(state):
        return "error_end"
    return "metadata_resolver"


def _route_after_metadata(state: IngestionState) -> str:
    """Route after metadata resolver: always continue (warnings are non-fatal)."""
    return "chunker"


def _route_after_chunker(state: IngestionState) -> str:
    """Route after chunker node: if fatal errors, go to END."""
    if _has_fatal_errors(state):
        return "error_end"
    return "embedding"


def _route_after_embedding(state: IngestionState) -> str:
    """Route after embedding node: if fatal errors, go to END."""
    if _has_fatal_errors(state):
        return "error_end"
    return "upsert"


def _route_after_upsert(state: IngestionState) -> str:
    """Route after upsert node: if fatal errors, go to END."""
    if _has_fatal_errors(state):
        return "error_end"
    return "health_check"


def build_ingestion_graph() -> StateGraph:
    """
    Build the full LangGraph ingestion subgraph.

    Returns:
        Compiled StateGraph for ingestion pipeline.
    """
    graph = StateGraph(IngestionState)

    # Add nodes
    graph.add_node("intake", intake_node)
    graph.add_node("parser", parser_node)
    graph.add_node("metadata_resolver", metadata_resolver_node)
    graph.add_node("chunker", chunker_node)
    graph.add_node("embedding", embedding_node)
    graph.add_node("upsert", upsert_node)
    graph.add_node("health_check", health_check_node)

    # Edges from START
    graph.add_edge(START, "intake")

    # Conditional edges
    graph.add_conditional_edges(
        "intake",
        _route_after_intake,
        {"parser": "parser", "error_end": END},
    )
    graph.add_conditional_edges(
        "parser",
        _route_after_parser,
        {"metadata_resolver": "metadata_resolver", "error_end": END},
    )
    graph.add_conditional_edges(
        "metadata_resolver",
        _route_after_metadata,
        {"chunker": "chunker"},
    )
    graph.add_conditional_edges(
        "chunker",
        _route_after_chunker,
        {"embedding": "embedding", "error_end": END},
    )
    graph.add_conditional_edges(
        "embedding",
        _route_after_embedding,
        {"upsert": "upsert", "error_end": END},
    )
    graph.add_conditional_edges(
        "upsert",
        _route_after_upsert,
        {"health_check": "health_check", "error_end": END},
    )

    # Final node always goes to END
    graph.add_edge("health_check", END)

    return graph


def compile_ingestion_graph():
    """
    Compile the ingestion graph into a runnable LangGraph instance.

    Returns:
        Compiled LangGraph ready for invocation.
    """
    graph_builder = build_ingestion_graph()
    compiled = graph_builder.compile()

    logger.info("Ingestion graph compiled successfully")
    logger.info(
        f"Graph nodes: {list(compiled.get_graph().nodes.keys()) if hasattr(compiled, 'get_graph') else 'N/A'}"
    )

    return compiled


# Convenience for direct import
ingestion_graph = compile_ingestion_graph()
