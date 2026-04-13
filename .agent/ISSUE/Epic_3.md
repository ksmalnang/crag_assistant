# Epic E3: RAG Query Pipeline – Intent, Retrieval, Grading, Rewriting, Generation, Hallucination Checking

**Epic:** E3  
**Labels:** `rag`, `query-pipeline`, `langgraph`, `epic-E3`  
**Priority:** High (US-012 through US-018) / Medium (US-019, US-020, US-021)  
**Total Story Points:** 3+4+3+3+4+3+4+3+3+2 = **32**

---

## Description

This epic builds the complete Corrective RAG (CRAG) query pipeline. It starts with classifying the user’s intent, then performs hybrid retrieval (dense + sparse) from Qdrant, grades retrieved documents for relevance, optionally rewrites the query if retrieval fails, generates a grounded answer with citations, and finally checks for hallucinations before returning the answer. Additional nodes include multi‑query expansion, conversation memory, and automatic metadata filter extraction.

All subtasks must be completed before the assistant can answer academic queries reliably.

---

## Tasks

### [ ] US-012: Build intent classifier node
**Role:** Developer  
**Points:** 3  
**Priority:** High  

- [ ] Few‑shot prompt with at least 10 examples per class  
- [ ] Output is always one of three enum values: `rag`, `chitchat`, `out_of_scope`  
- [ ] Unit tested with 20 queries (min 85% accuracy on test set)  
- [ ] Fallback: if classification fails, default to `rag`  

---

### [ ] US-013: Build hybrid retrieval node (Qdrant dense + sparse)
**Role:** Developer  
**Points:** 4  
**Priority:** High  

- [ ] Qdrant `query_points` with fusion=RRF (Reciprocal Rank Fusion)  
- [ ] Top‑K configurable (default: 6)  
- [ ] Payload filter injected from state metadata filters  
- [ ] Retrieved docs include: `content`, `source`, `page_number`, `section_title`, `score`  
- [ ] Integration test against populated Qdrant collection  

---

### [ ] US-014: Build retrieval grader node
**Role:** Developer  
**Points:** 3  
**Priority:** High  

- [ ] Binary score: `yes` or `no` per document  
- [ ] Structured output using Pydantic model  
- [ ] If 0 relevant docs found → route to query rewriter  
- [ ] If ≥1 relevant doc → route to generator  
- [ ] Grading prompt tested against 15 query‑doc pairs  

---

### [ ] US-015: Build query rewriter node
**Role:** Developer  
**Points:** 3  
**Priority:** High  

- [ ] Rewrites by decomposing, expanding, and clarifying the original query  
- [ ] `retry_count` incremented on each rewrite  
- [ ] Hard stop after 3 retries → fallback response  
- [ ] Rewritten query stored in state as `rewritten_query`  
- [ ] Unit tested with 10 vague/ambiguous student queries  

---

### [ ] US-016: Build answer generator node with citations
**Role:** Developer  
**Points:** 4  
**Priority:** High  

- [ ] Prompt instructs model to answer only from retrieved context  
- [ ] Citations formatted as `[Source: filename, p.X]`  
- [ ] Answer includes “I don’t know” if context is insufficient  
- [ ] Structured output: `{answer: str, citations: list[Citation]}`  
- [ ] Tested against 20 question‑context pairs for groundedness  

---

### [ ] US-017: Build hallucination checker node
**Role:** Developer  
**Points:** 3  
**Priority:** High  

- [ ] LLM‑as‑judge prompt: “Is every claim in the answer supported by the provided docs?”  
- [ ] Binary output: `grounded` | `not_grounded`  
- [ ] If `not_grounded` → regenerate (max 2 attempts)  
- [ ] If still `not_grounded` after 2 → return safe fallback message  
- [ ] Unit tested with 10 grounded and 10 hallucinated answer pairs  

---

### [ ] US-018: Wire full CRAG LangGraph with conditional edges
**Role:** Developer  
**Points:** 4  
**Priority:** High  

- [ ] StateGraph compiled without errors  
- [ ] Conditional edges defined for: intent, relevance, hallucination  
- [ ] Graph visualized and saved as `graph.png` using `draw_mermaid_png()`  
- [ ] End‑to‑end integration test passes for: `rag`, `chitchat`, `out_of_scope` query types  
- [ ] `retry_count` loop correctly terminates at max retries  

---

### [ ] US-019: Implement multi‑query expansion node
**Role:** Developer  
**Points:** 3  
**Priority:** Medium  

- [ ] Generates 3 semantically varied versions of the original query  
- [ ] Retrieval run for each variant  
- [ ] Results deduplicated by doc ID before grading  
- [ ] Togglable via config flag (`multi_query: true/false`)  

---

### [ ] US-020: Add conversation memory with LangGraph MemorySaver
**Role:** Student  
**Points:** 3  
**Priority:** Medium  

- [ ] `MemorySaver` checkpointer configured on the graph  
- [ ] `thread_id` passed per user session  
- [ ] Last N=5 turns included in generation context  
- [ ] Memory cleared on explicit user reset or new session  
- [ ] Integration test validates follow‑up question resolution  

---

### [ ] US-021: Implement metadata filter extractor node
**Role:** Developer  
**Points:** 2  
**Priority:** Medium  

- [ ] Extracts: `faculty`, `semester`, `doc_type` from query using LLM or regex  
- [ ] Filters injected into retrieval node state  
- [ ] No filter applied if extraction confidence is low  
- [ ] Unit tested with 15 scoped queries  

---

## Definition of Done

- All checkboxes above are ticked.  
- Each task’s acceptance criteria are satisfied.  
- The full CRAG graph runs end‑to‑end and correctly routes based on intent, relevance, and hallucination checks.  
- Unit and integration tests pass.  
- Graph visualisation is committed.  
- Code reviewed and merged into `main`.
