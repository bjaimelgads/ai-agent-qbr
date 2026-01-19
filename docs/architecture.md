# Target Architecture

## Goals
- Clean architecture separation: domain and application layers do not depend on SQLite or SQLAlchemy.
- Preserve AG-UI and legacy WebSocket streaming.
- Enable multiple storage backends (SQLite now, others later).
- Minimize churn by wrapping existing data access behind ports.

## Layered Design

### Domain Layer
Entities:
- Document
- Slide
- Chunk
- Embedding
- RetrievalResult
- Answer

Value Objects:
- DocumentId
- ChunkId
- EmbeddingVector
- Score

Responsibilities:
- Pure data + invariants.
- No persistence or transport concerns.

### Application Layer
Use cases:
- IngestPpt
- SearchKnowledge
- AnswerQuestion
- StreamAnswer

Ports (interfaces):
- KnowledgeRepository
- VectorIndex
- EmbeddingsProvider

Responsibilities:
- Orchestrate domain logic.
- Apply policies (e.g., top-k, score thresholds, context formatting).
- Call ports only.

### Infrastructure Layer
Implementations:
- SQLiteKnowledgeRepository (SQLAlchemy against `qbr_intelligence.db`).
- VectorIndex implementation backed by existing embeddings (in-process cosine search or FAISS index).
- Adapters for existing QBR extraction data (chunk metadata, slide references).

Responsibilities:
- Translate between persistence models and domain entities.
- Connect to SQLite and any vector backend.

### Interface Layer
- WebSocket server (legacy + AG-UI output).
- HTTP endpoints if required (future).
- Input adapters to map inbound WebSocket payloads to application use cases.

Responsibilities:
- Accept requests, validate, and invoke use cases.
- Stream AG-UI events or legacy DTOs.

## Configuration and Dependency Injection
- Select storage and vector backends via env vars:
  - `STORAGE_BACKEND=sqlite`
  - `VECTOR_BACKEND=sqlite_embeddings|faiss|pgvector`
- Resolve implementations in a composition root (e.g., `src/ai_agent_qbr/app_factory.py`).
- Keep configuration in `Config` dataclass with validation for required fields.

## Migration Strategy (Minimize Churn)
1. Wrap existing QBR SQLAlchemy queries behind a `KnowledgeRepository` interface.
2. Introduce `VectorIndex` port with a default implementation that reads from `chunks.embedding` and performs cosine similarity in-process.
3. Update the orchestrator/use case to call `SearchKnowledge` and pass retrieved chunks into the LLM context.
4. Swap WebSocket handling to reuse the base agent AG-UI adapter while preserving legacy output.
5. Add optional FAISS/pgvector adapters later without touching domain or application layers.
   - Build FAISS index in `FAISS_DIR` and set `VECTOR_BACKEND=faiss`.

## Clean Architecture Rules
- Domain does not import from application, infrastructure, or interface.
- Application depends only on domain and ports.
- Infrastructure implements ports but never imported by domain.
- Interface uses application use cases and DTOs only.

## Testing Strategy Alignment
- Domain: pure unit tests for value object validation.
- Application: use-case tests with fake ports.
- Infrastructure: integration tests against SQLite fixtures.
- Interface: end-to-end WebSocket tests that validate AG-UI streaming.
