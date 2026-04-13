# Epic E1: Foundation Setup – Monorepo, Environment, Qdrant, LangSmith & LangGraph State

**Epic:** E1  
**Labels:** `foundation`, `setup`, `epic-E1`  
**Priority:** High (US-001, US-002, US-003) / Medium (US-004, US-005)  
**Total Story Points:** 3+2+3+2+2 = **12**

---

## Description

This issue covers the initial foundation tasks required to establish the monorepo structure, environment & secrets management, Qdrant vector store, LangSmith tracing, and the base LangGraph state schema. All subtasks must be completed before any feature work begins.

---

## Tasks

### [ ] US-001: Initialize monorepo project structure
**Role:** Project Lead  
**Points:** 3  
**Priority:** High  

- [ ] Monorepo initialized (e.g. using Poetry workspaces or uv)  
- [ ] Folders created: `/ingestion`, `/pipeline`, `/api`, `/frontend`, `/evals`, `/infra`  
- [ ] `README.md` written with setup instructions  
- [ ] Pre-commit hooks configured (ruff, black, isort)  

---

### [ ] US-002: Configure environment & secrets management
**Role:** Developer  
**Points:** 2  
**Priority:** High  

- [ ] `pydantic-settings` `BaseSettings` class defined  
- [ ] `.env.example` committed with all required keys documented  
- [ ] Secrets never committed to git (`.gitignore` verified)  
- [ ] CI environment variables documented in `README.md`  

---

### [ ] US-003: Provision Qdrant vector store
**Role:** Developer  
**Points:** 3  
**Priority:** High  

- [ ] `docker-compose.yml` with Qdrant service defined  
- [ ] Qdrant health check endpoint verified  
- [ ] Collections schema documented (named vectors: dense + sparse)  
- [ ] Connection utility class written and unit tested  

---

### [ ] US-004: Set up LangSmith tracing for development
**Role:** Developer  
**Points:** 2  
**Priority:** Medium  

- [ ] `LANGCHAIN_TRACING_V2=true` set in `.env`  
- [ ] LangSmith project created and linked  
- [ ] Sample trace confirmed visible in LangSmith dashboard  
- [ ] Tracing disabled gracefully when key is absent  

---

### [ ] US-005: Define LangGraph base state schema
**Role:** Developer  
**Points:** 2  
**Priority:** Medium  

- [ ] `RAGState` TypedDict defined with: `query`, `rewritten_query`, `intent`, `retrieved_docs`, `relevance_scores`, `generation`, `retry_count`, `citations`  
- [ ] State schema versioned and documented  
- [ ] Unit test asserting state fields exist and are typed correctly  

---

## Definition of Done

- All checkboxes above are ticked.  
- Each task’s acceptance criteria are met.  
- CI passes (once available).  
- Code reviewed and merged into `main`.
