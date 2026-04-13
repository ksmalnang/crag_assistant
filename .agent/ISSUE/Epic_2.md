# Epic E2: Document Ingestion – Load, Chunk, Embed, Upsert

**Epic:** E2  
**Labels:** `ingestion`, `document-processing`, `epic-E2`  
**Priority:** High (US-006, US-007, US-008, US-009) / Medium (US-010, US-011)  
**Total Story Points:** 5+4+3+3+3+2 = **20**

---

## Description

This epic implements the complete document ingestion pipeline. It covers loading academic documents (PDF, DOCX, PPTX) using Docling, semantically chunking by heading hierarchy, generating dense and sparse embeddings, upserting into Qdrant with deduplication, and orchestrating the whole process via a LangGraph subgraph. Additionally, metadata extraction from filenames is added for downstream filtering.

All subtasks must be completed before query/retrieval features can rely on ingested documents.

---

## Tasks

### [ ] US-006: Build Docling document loader node
**Role:** Developer  
**Points:** 5  
**Priority:** High  

- [ ] Supports PDF, DOCX, PPTX input formats  
- [ ] Tables extracted and preserved as structured data  
- [ ] Heading hierarchy retained in output metadata  
- [ ] Figures detected and page references stored  
- [ ] Unit tested with at least 3 real academic document samples  

---

### [ ] US-007: Implement semantic section chunker
**Role:** Developer  
**Points:** 4  
**Priority:** High  

- [ ] Chunks split at H1/H2/H3 boundaries from Docling structure  
- [ ] Fallback to 512-token sliding window with 64-token overlap if no headings  
- [ ] Each chunk carries metadata: source, page_number, section_title, doc_type, faculty, semester  
- [ ] No chunk exceeds 1024 tokens  
- [ ] Unit tests cover heading-based and fallback splitting  

---

### [ ] US-008: Implement dense + sparse embedding generation
**Role:** Developer  
**Points:** 3  
**Priority:** High  

- [ ] Dense embeddings via bge-m3 or text-embedding-3-small  
- [ ] Sparse embeddings via fastembed BM25  
- [ ] Both vectors stored as named vectors in Qdrant  
- [ ] Embedding step is batched (configurable batch size)  
- [ ] Integration test confirms both vectors exist in Qdrant payload  

---

### [ ] US-009: Build Qdrant upsert node with deduplication
**Role:** Developer  
**Points:** 3  
**Priority:** High  

- [ ] Point ID = SHA256(source_path + chunk_index)  
- [ ] Upsert used (not insert) to handle reruns safely  
- [ ] Metadata payload stored alongside vectors  
- [ ] Batch upsert with configurable size (default: 100)  
- [ ] Upsert errors logged and retried up to 3 times  

---

### [ ] US-010: Build ingestion orchestration script (LangGraph subgraph)
**Role:** Developer  
**Points:** 3  
**Priority:** Medium  

- [ ] LangGraph ingestion subgraph with 4 nodes wired in sequence  
- [ ] Accepts a folder path as input  
- [ ] Progress logged per document  
- [ ] Final summary report: X docs ingested, Y chunks created, Z errors  
- [ ] CLI entry point: `python -m ingestion.run --folder ./docs`  

---

### [ ] US-011: Add metadata filter extractor for faculty/semester/type
**Role:** Developer  
**Points:** 2  
**Priority:** Medium  

- [ ] Filename convention documented: `FACULTY_SEMESTER_DOCTYPE_name.pdf`  
- [ ] Parser extracts all three fields reliably  
- [ ] Fallback to 'unknown' for missing fields (never null)  
- [ ] Unit tested against 10 sample filenames  

---

## Definition of Done

- All checkboxes above are ticked.  
- Each task’s acceptance criteria are satisfied.  
- Ingestion pipeline can process a folder of mixed academic documents.  
- Unit and integration tests pass.  
- Qdrant contains both dense and sparse vectors for ingested chunks.  
- Code reviewed and merged into `main`.
