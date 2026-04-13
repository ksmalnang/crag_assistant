# CRAG Pipeline - Corrective Retrieval Augmented Generation

A production-ready RAG pipeline using LangGraph, Qdrant vector store, and LangSmith tracing.

## Setup

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Environment Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys and configuration
```

Required environment variables:
- `OPENAI_API_KEY` - OpenAI API key for embeddings and generation
- `QDRANT_URL` - Qdrant vector store URL (default: http://localhost:6333)
- `LANGCHAIN_API_KEY` - LangSmith API key (optional, for tracing)
- `LANGCHAIN_TRACING_V2` - Enable LangSmith tracing (true/false)
- `LANGCHAIN_PROJECT` - LangSmith project name

### Running Services

```bash
# Start Qdrant vector store
docker-compose up -d

# Verify Qdrant health
curl http://localhost:6333/healthz
```

### Running Tests

```bash
uv run pytest
```

### Pre-commit Hooks

```bash
uv run pre-commit install
```

## Project Structure

```
crag_pipeline/
├── ingestion/      # Document ingestion and processing
├── pipeline/       # LangGraph RAG pipeline
├── api/            # FastAPI REST API
├── frontend/       # User interface
├── evals/          # Evaluation and benchmarking
├── infra/          # Infrastructure and deployment configs
└── tests/          # Unit and integration tests
```

## CI/CD Environment Variables

The following variables should be configured in your CI/CD system:

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `QDRANT_URL` | Qdrant URL | Yes |
| `LANGCHAIN_API_KEY` | LangSmith API key | No (dev only) |
| `LANGCHAIN_TRACING_V2` | Enable tracing | No (dev only) |
| `LANGCHAIN_PROJECT` | LangSmith project | No (dev only) |
