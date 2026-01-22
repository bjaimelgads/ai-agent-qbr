# QBR Agent Architecture and Workflows

## How This System Works

- Ingest a QBR deck, extract and chunk content, and store it in a SQL database with embeddings and FTS5 text index.
- At runtime, retrieve the most relevant chunks and use them as context to answer questions.
- Answers are grounded in retrieved QBR content.

## Architecture (High Level)

```mermaid
flowchart LR
  %% Styles
  classDef source fill:#F4F1E8,stroke:#7A6E5A,color:#2E2A23
  classDef process fill:#E7F0FA,stroke:#3B6EA8,color:#1F2A3A
  classDef store fill:#FFF5E6,stroke:#C77D2D,color:#3B2A12
  classDef runtime fill:#EAF7EF,stroke:#3D8B5C,color:#1C2E23
  classDef edge fill:#FDECEC,stroke:#C94A4A,color:#3A1A1A

  subgraph A["Extraction Pipeline (qbr_extraction)"]
    A1["PPTX files"]:::source --> A2["Extraction pipeline<br/>text + images + slide structure"]:::process
    A2 --> A3["Chunking + embeddings"]:::process
    A2 --> A4["Heuristics<br/>metrics + charts + keywords"]:::process
    A4 --> A5["Post-processing<br/>classification + summaries"]:::process
    A5 --> A6["Post-embeddings"]:::process
    A6 --> A7["SQLAlchemy writer"]:::process
    A7 --> A8[(SQL database: qbr_intelligence + FTS5)]:::store
  end

  subgraph B["Indexing & Retrieval Infrastructure"]
    B1[("SQL database: documents + chunks + metadata + FTS5")]:::store
    B2["Vector index<br/>SQL embeddings or FAISS"]:::process
    B3["Embeddings provider<br/>SentenceTransformers or hash"]:::process
    B4["BM25 text search<br/>FTS5 chunks_fts"]:::process
  end

  subgraph C["Runtime Agent (ai_agent_qbr)"]
    C1["Client interface layer"]:::runtime
    C2["Session + Orchestrator"]:::runtime
    C3["Planner (PenguiFlow)"]:::runtime
    C4["Retrieval use cases<br/>HybridSearch + AnswerQuestion"]:::runtime
    C5["Memory + Telemetry"]:::runtime
  end

  subgraph D["Users & Clients"]
  D1["Web app / client"]:::edge
  end

  A8 --> B1
  B1 --> B2
  B3 --> B2
  B1 --> B4

  D1 --> C1
  C1 --> C2 --> C4 --> B2
  C4 --> B4
  C4 --> C3
  C3 --> C1 --> D1
  C2 --> C5
```

## Workflow 1: Extraction-Only (Ingestion to Database)

```mermaid
flowchart TD
  classDef source fill:#F4F1E8,stroke:#7A6E5A,color:#2E2A23
  classDef process fill:#E7F0FA,stroke:#3B6EA8,color:#1F2A3A
  classDef store fill:#FFF5E6,stroke:#C77D2D,color:#3B2A12

  E1["PPTX file uploaded"]:::source --> E2["Kreuzberg extraction<br/>text + images + slide structure"]:::process
  E2 --> E3["Chunking + embedding generation"]:::process
  E2 --> E4["Detect metrics + charts + keywords"]:::process
  E4 --> E5["LLM enhancement<br/>classify + summarize + normalize"]:::process
  E5 --> E6["Post-embeddings"]:::process
  E6 --> E7["Write to SQL database + FTS5 index<br/>documents, slides, chunks"]:::process
  E7 --> E8[("qbr_intelligence ready for retrieval")]:::store
```

### What Gets Stored
- Documents + slides
- Chunks (text segments with embeddings)
- Metrics, charts, keywords, entities, images
- LLM-generated summaries and classifications (when enabled)

## Workflow 2: End-to-End (Extraction to Answer)

```mermaid
sequenceDiagram
  participant U as User
  participant C as Client App
  participant IF as Client Interface
  participant OR as Orchestrator
  participant UC as Retrieval Use Cases
  participant DB as SQL database (qbr_intelligence + FTS5)
  participant VI as Vector Index (SQL/FAISS)
  participant TS as Text Search (FTS5 BM25)
  participant PL as Planner/LLM

  Note over DB: Filled earlier by extraction pipeline

  U->>C: Ask a QBR question
  C->>IF: User request
  IF->>OR: SendMessageUseCase
  OR->>UC: AnswerQuestion (hybrid)
  UC->>VI: Vector search on embeddings
  UC->>TS: BM25 text search on chunks
  UC->>DB: Fetch chunks by ID
  UC->>UC: Rerank top candidates
  UC-->>OR: Ranked chunks + context
  OR->>PL: Run planner with qbr_context
  PL-->>OR: Final answer + metadata
  OR-->>IF: Response payload
  IF-->>C: Response delivered
  C-->>U: Answer shown
```

### Plain-English Summary
- Extraction builds a searchable knowledge base (SQL + FTS5) from PowerPoint slides.
- When a user asks a question, the agent searches both BM25 text and embeddings, reranks candidates, and then gives the LLM the most relevant slides as context.
- The client interface returns the response back to the user.

## Where This Lives in the Repo
- Extraction pipeline: `qbr_extraction/qbr_intelligence/pipeline/processor.py`
- Runtime agent orchestration: `src/ai_agent_qbr/orchestrator.py`
- Retrieval use cases: `src/qbr_agent/application/use_cases.py`
- Storage & vector index: `src/qbr_agent/infrastructure/`
- Client interface layer: `src/ai_agent_qbr/transport/`
