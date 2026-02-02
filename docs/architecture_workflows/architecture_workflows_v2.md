# QBR Agent Architecture and Workflows (v2)

## How This System Works (v2)

- Ingest a QBR deck, extract text/structure, and build two knowledge stores:
  - Unstructured chunk store (embeddings + FTS5 BM25) for semantic retrieval.
  - Structured metric store (metric catalog + metric facts) for deterministic metric QA.
- At runtime, the planner routes metric/KPI questions to Metric QA, and uses hybrid retrieval
  (embeddings + BM25) for narrative, insight, or slide-context questions.
- Optional LLM enrichment exists for metric refinement and context labeling, but is still WIP
  and gated by configuration.

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
    A1["PPTX files"]:::source --> A2["Kreuzberg extraction<br/>text + images + slide structure"]:::process
    A2 --> A3["Chunking + embeddings"]:::process
    A2 --> A4["Metric scanning<br/>raw_context + values + units"]:::process
    A4 --> A5["Metric catalog mapping<br/>aliases + disambiguation"]:::process
    A5 --> A6["Metric context strategies<br/>period/brand/baseline"]:::process
    A6 --> A7["Confidence + dedupe filters<br/>notes removal + best-candidate"]:::process
    A7 --> A8["Optional LLM refinement<br/>context label + normalization (WIP)"]:::process
    A8 --> A9["SQLAlchemy writer"]:::process
    A9 --> A10[("qbr_intelligence database")]:::store
  end

  subgraph B["Structured Metric Store"]
    B1[("metric_catalog + metric_aliases")]:::store
    B2[("metrics + periods + clients + regions")]:::store
    B3["metric_fact view<br/>joins + provenance"]:::process
  end

  subgraph C["Indexing & Retrieval Infrastructure"]
    C1[("SQL DB: documents + chunks + metadata + FTS5")]:::store
    C2["Vector index<br/>SQL embeddings or FAISS"]:::process
    C3["Embeddings provider"]:::process
    C4["BM25 text search<br/>FTS5 chunks_fts"]:::process
  end

  subgraph D["Runtime Agent (ai_agent_qbr)"]
    D1["Client interface layer"]:::runtime
    D2["Session + Orchestrator"]:::runtime
    D3["Planner (PenguiFlow)"]:::runtime
    D4["Metric Router + Metric QA tool"]:::runtime
    D5["Hybrid Retrieval tools"]:::runtime
    D6["Memory + Telemetry"]:::runtime
  end

  subgraph E["Users & Clients"]
    E1["Web app / client"]:::edge
  end

  A10 --> C1
  A10 --> B1
  A10 --> B2
  B1 --> B3
  B2 --> B3
  C1 --> C2
  C3 --> C2
  C1 --> C4

  E1 --> D1
  D1 --> D2 --> D3
  D3 --> D4 --> B3
  D3 --> D5 --> C2
  D5 --> C4
  D2 --> D6
  D3 --> D1 --> E1
```

## Workflow 1: Extraction + Knowledge Build (Ingestion to Database)

```mermaid
flowchart TD
  classDef source fill:#F4F1E8,stroke:#7A6E5A,color:#2E2A23
  classDef process fill:#E7F0FA,stroke:#3B6EA8,color:#1F2A3A
  classDef store fill:#FFF5E6,stroke:#C77D2D,color:#3B2A12

  E1["PPTX file uploaded"]:::source --> E2["Kreuzberg extraction<br/>text + images + slide structure"]:::process
  E2 --> E3["Chunking + embedding generation"]:::process
  E2 --> E4["Metric scan<br/>raw_context + values + units"]:::process
  E4 --> E5["Metric catalog mapping<br/>aliases + disambiguation"]:::process
  E5 --> E6["Context strategies<br/>period/brand/baseline"]:::process
  E6 --> E7["Confidence + dedupe filters<br/>speaker-notes removal"]:::process
  E7 --> E8["Optional LLM refine<br/>context label + normalize (WIP)"]:::process
  E8 --> E9["Write to SQL + FTS5<br/>documents, slides, chunks, metrics"]:::process
  E9 --> E10[("qbr_intelligence ready for retrieval + Metric QA")]:::store
```

### Confidence & Selection Strategies (Metric Pipeline)
- Use metric catalog and aliases to resolve metric names and disambiguation tokens.
- Prefer candidates with higher `extraction_confidence`; tie-break by richer `raw_context`.
- Drop metrics sourced only from speaker notes or empty/duplicate contexts.
- Apply rule-based period/brand/baseline extraction (with optional LLM refinement).

## Workflow 2: Runtime Answering (Planner Routes to the Right Retrieval)

```mermaid
sequenceDiagram
  participant U as User
  participant C as Client App
  participant OR as Orchestrator
  participant PL as Planner/LLM
  participant MR as Metric Router
  participant MQ as Metric QA Tool
  participant HR as Hybrid Retrieval
  participant DB as SQL DB (qbr_intelligence)

  U->>C: Ask a QBR question
  C->>OR: Request
  OR->>PL: Run planner with qbr_context + tools
  PL->>MR: Evaluate metric/KPI intent
  alt Metric/KPI question
    MR-->>PL: Route to Metric QA
    PL->>MQ: query_metrics(question)
    MQ->>DB: Query metric_fact view
    MQ-->>PL: MetricAnswer + citations
  else Narrative / context question
    MR-->>PL: Use hybrid search
    PL->>HR: search_documents / answer_question
    HR->>DB: Vector + BM25 search
    HR-->>PL: Ranked chunks + context
  end
  PL-->>OR: Final answer + metadata
  OR-->>C: Response
  C-->>U: Answer
```

### Metric QA: How It Works
- A deterministic engine builds a `metric_fact` view by joining metrics with catalog, periods,
  clients, and regions.
- The intent parser resolves metric IDs, client/region, and period filters.
- The query planner applies filters, ordering, and aggregation (latest, trend, compare, avg, sum).
- The answer composer returns a structured answer plus citations (document_id, slide_id).
- Optional LLM phrasing is gated and currently WIP.

### Planner Routing: Metric QA vs Semantic Search
- Metric/KPI questions (values, trends, period comparisons) are routed to `query_metrics` first.
- Open-ended narrative or qualitative questions use hybrid retrieval (embeddings + BM25).
- If structured metric intent fails or is ambiguous, the planner can fall back to hybrid search
  and ask for clarifications.

## Workflow 3: Metric QA (Structured Retrieval)

```mermaid
flowchart TD
  classDef process fill:#E7F0FA,stroke:#3B6EA8,color:#1F2A3A
  classDef store fill:#FFF5E6,stroke:#C77D2D,color:#3B2A12

  M1["User asks metric question"]:::process --> M2["Intent parsing<br/>metric + client + region + period"]:::process
  M2 --> M3["Metric catalog resolution<br/>aliases + disambiguation"]:::process
  M3 --> M4["SQL plan builder<br/>filters + ordering"]:::process
  M4 --> M5[("metric_fact view")]:::store
  M5 --> M6["Answer composer<br/>math + citations"]:::process
  M6 --> M7["Optional LLM phrasing (WIP)"]:::process
```

## WIP: Document Path and Slide Links
- Current retrieval returns document metadata only; it does not resolve Google Slides URLs.
- Planned: connect to Google Slides API, map document -> slide IDs, and return a concrete
  slide link per citation.
- The agent currently includes `document_url` only if already stored in the DB.

## Where This Lives in the Repo
- Extraction pipeline: `qbr_extraction/qbr_intelligence/pipeline/processor.py`
- Metric context strategies: `qbr_extraction/qbr_intelligence/pipeline/metric_context.py`
- Metric QA engine: `qbr_extraction/qbr_intelligence/metric_qa/`
- Runtime orchestrator: `src/ai_agent_qbr/orchestrator.py`
- Metric router + tool: `src/ai_agent_qbr/infrastructure/metric_router.py`, `src/ai_agent_qbr/tools/query_metrics.py`
- Hybrid retrieval use cases: `src/qbr_agent/application/use_cases.py`
