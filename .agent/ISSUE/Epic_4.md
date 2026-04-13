# Epic E4: API & Backend – Chat Endpoint, Streaming, Sessions, Rate Limiting, Ingestion Trigger

**Epic:** E4  
**Labels:** `api`, `backend`, `fastapi`, `epic-E4`  
**Priority:** High (US-022, US-023) / Medium (US-024, US-025, US-026)  
**Total Story Points:** 4+3+2+3+2 = **14**

---

## Description

This epic builds the public API and backend services that expose the RAG assistant to clients. It includes a chat endpoint (with streaming support), session management, rate limiting and optional API key authentication, plus an admin endpoint to trigger document ingestion. All endpoints must be documented via OpenAPI.

---

## Tasks

### [ ] US-022: Build FastAPI chat endpoint
**Role:** Developer  
**Points:** 4  
**Priority:** High  

- [ ] `POST /chat` accepts: `{query: str, session_id: str, filters?: object}`  
- [ ] Returns: `{answer: str, citations: list, intent: str, sources: list}`  
- [ ] Input validated with Pydantic  
- [ ] Errors return standard RFC 7807 problem JSON  
- [ ] OpenAPI docs auto‑generated at `/docs`  

---

### [ ] US-023: Add streaming support to chat endpoint
**Role:** Student  
**Points:** 3  
**Priority:** High  

- [ ] `GET /chat/stream` endpoint with Server‑Sent Events (SSE)  
- [ ] LangGraph `astream_events` used for streaming  
- [ ] Tokens streamed as they are generated  
- [ ] Final event includes citations and metadata  
- [ ] Frontend tested with EventSource API  

---

### [ ] US-024: Add session management endpoint
**Role:** Developer  
**Points:** 2  
**Priority:** Medium  

- [ ] `POST /session/new` returns a new UUID `session_id`  
- [ ] `DELETE /session/{id}` clears MemorySaver state for that thread  
- [ ] Sessions expire after 30 minutes of inactivity  

---

### [ ] US-025: Implement rate limiting and basic auth
**Role:** Developer  
**Points:** 3  
**Priority:** Medium  

- [ ] Rate limit: 30 requests/minute per `session_id` (using `slowapi`)  
- [ ] Optional `X-API-Key` header auth configurable via env  
- [ ] `429 Too Many Requests` returned with `Retry-After` header  
- [ ] Auth bypass available for internal/admin routes  

---

### [ ] US-026: Build ingestion trigger endpoint
**Role:** Admin  
**Points:** 2  
**Priority:** Medium  

- [ ] `POST /ingest` accepts: `{folder_path: str, faculty: str, semester: str}`  
- [ ] Ingestion runs as background task (FastAPI `BackgroundTasks`)  
- [ ] Returns `job_id` immediately; `GET /ingest/{job_id}` returns status  
- [ ] Admin‑only route protected by API key  

---

## Definition of Done

- All checkboxes above are ticked.  
- Each task’s acceptance criteria are satisfied.  
- API is fully documented (Swagger UI at `/docs`).  
- All endpoints are tested (unit + integration).  
- Rate limiting and auth work as specified.  
- Streaming works with a test frontend client.  
- Code reviewed and merged into `main`.
