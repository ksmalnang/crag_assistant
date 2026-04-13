# Infrastructure Configuration

This directory contains infrastructure-related configuration files.

## Qdrant Vector Store

### Collections Schema

The CRAG pipeline uses Qdrant with the following collection configuration:

**Collection Name**: `documents` (configurable via `QDRANT_COLLECTION`)

**Vectors Configuration**:
- **Dense Vector**: 
  - Name: `dense`
  - Size: 3072 (configurable via `QDRANT_VECTOR_SIZE`)
  - Distance: Cosine similarity
  
- **Sparse Vector**: 
  - Name: `sparse`
  - Index: In-memory (configurable)

### Starting Qdrant

```bash
docker-compose up -d
```

### Health Check

```bash
curl http://localhost:6333/healthz
```

Expected response: `{"title":"qdrant","version":"1.12.4","status":"green"}`
