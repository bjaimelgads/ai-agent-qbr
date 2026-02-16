# QBR Progress

## Step 1: Broad Search Proof of Concept

```mermaid
flowchart LR
  Q[User Question] --> A[Search all docs]
  Q --> B[Search all slides]
  A --> R[Answer]
  B --> R
```

- First, we searched across all documents and all slides.
- This proved the concept and showed end-to-end value quickly.
- However, comparison use cases exposed limits in precision.
- Because context was too broad, noisy matches could reduce answer quality.

## Step 2: Intent-Based Narrowing + Metric Catalog

```mermaid
flowchart LR
  Q[User Question] --> I[Detect intent]
  I --> F[Filter by region/client/period]
  I --> M[Use metric_catalog]
  F --> R[Focused retrieval]
  M --> R
```

- Then, we narrowed context using user intent.
- We added filters for region, brand/client, and time period.
- In parallel, we added `metric_catalog` for business metric values.
- As a result, retrieval became more focused and reliable.

## Step 3: Multi-Deck Search + Source Links

```mermaid
flowchart LR
  Q[Comparison question] --> D[Search deck A + B + C]
  D --> C[Compare metrics/content]
  C --> S[Answer with clickable deck/slide sources]
```

- After that, we introduced multi-deck search.
- This enabled controlled comparisons across periods, regions, and clients/brands.
- We also added clickable deck and slide source links.
- Therefore, answers became easier to validate and trust.

## Step 4: Next Improvements

```mermaid
flowchart LR
  N[Next phase] --> C[Improve metric catalog quality]
  N --> T[Improve metric retrieval tables]
  C --> A[More accurate reasoning]
  T --> B[Better answers for trends<br/>e.g., CPA by recent quarters]
```

- Next, we will improve metric catalog quality.
- This should improve reasoning accuracy over business metric patterns.
- We will also improve metric retrieval output format.
- For example, we want table-style answers for trends like CPA across recent quarters.

## Step 5: Adding New Clients/Brands

```mermaid
flowchart LR
  N[New client/brand] --> O[Onboard deck and metadata]
  O --> M[Map metrics to metric_catalog]
  M --> V[Validate retrieval and comparisons]
  V --> S[Go live for questions and analysis]
```

- After that, we will scale by adding new clients and brands.
- We will onboard their decks and connect their metrics to `metric_catalog`.
- We will validate that comparisons and sources stay accurate.
- This will expand coverage while keeping the same product experience.
