"""Tests for LangGraph state schema."""

from typing import get_type_hints

import pytest

from pipeline.state import STATE_SCHEMA_VERSION, RAGState, create_initial_state


class TestRAGState:
    """Test cases for RAGState TypedDict."""

    def test_state_fields_exist(self):
        """Test that all required fields are defined."""
        hints = get_type_hints(RAGState)

        expected_fields = [
            "query",
            "rewritten_query",
            "intent",
            "retrieved_docs",
            "relevance_scores",
            "generation",
            "retry_count",
            "citations",
        ]

        for field in expected_fields:
            assert field in hints, f"Field '{field}' missing from RAGState"

    def test_create_initial_state(self):
        """Test creating initial state with default values."""
        query = "What is machine learning?"
        state = create_initial_state(query)

        assert state["query"] == query
        assert state["rewritten_query"] is None
        assert state["intent"] is None
        assert state["retrieved_docs"] == []
        assert state["relevance_scores"] == []
        assert state["generation"] is None
        assert state["retry_count"] == 0
        assert state["citations"] == []

    def test_state_field_types(self):
        """Test that state fields have correct types."""
        state = create_initial_state("test query")

        # String fields
        assert isinstance(state["query"], str)
        assert state["rewritten_query"] is None or isinstance(
            state["rewritten_query"], str
        )
        assert state["intent"] is None or isinstance(state["intent"], str)
        assert state["generation"] is None or isinstance(state["generation"], str)

        # List fields
        assert isinstance(state["retrieved_docs"], list)
        assert isinstance(state["relevance_scores"], list)
        assert isinstance(state["citations"], list)

        # Integer field
        assert isinstance(state["retry_count"], int)

    def test_state_schema_version(self):
        """Test that state schema version is defined."""
        assert isinstance(STATE_SCHEMA_VERSION, str)
        assert len(STATE_SCHEMA_VERSION) > 0

    def test_state_mutability_with_annotated_fields(self):
        """Test that annotated fields work correctly with operator.add."""
        state = create_initial_state("test query")

        # Simulate adding docs
        state["retrieved_docs"].append({"content": "doc1"})
        state["retrieved_docs"].append({"content": "doc2"})

        assert len(state["retrieved_docs"]) == 2

        # Simulate adding scores
        state["relevance_scores"].append(0.95)
        state["relevance_scores"].append(0.87)

        assert len(state["relevance_scores"]) == 2
