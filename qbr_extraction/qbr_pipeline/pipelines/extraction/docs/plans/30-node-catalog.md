# Node Catalog Draft (ReactPlanner)

## Node: `rag_retrieve`

Purpose:
- Retrieve top-k chunks relevant to the user question using embeddings.

Input:
- `question: str`
- `document_id: int | None`
- `filters: dict | None`

Output:
- `chunks: list[{id, content, document_id, score, metadata}]`

Notes:
- Use embeddings from `chunks.embedding` when available.
- Fallback to text search for missing embeddings.

## Node: `rag_summarize`

Purpose:
- Summarize retrieved chunks into compact context.

Input:
- `chunks`

Output:
- `summary: str`
- `citations: list`

## Node: `answer_final`

Purpose:
- Compose final user response grounded in summary + citations.

Input:
- `question`
- `summary`
- `citations`

Output:
- `answer: str`
- `citations: list`

## Node: `list_documents`

Purpose:
- List available documents with filters.

Input:
- `client_name | status | date_from | date_to | limit | offset`

Output:
- `documents: list`

## Node: `get_document`

Purpose:
- Fetch detailed document metadata.

Input:
- `document_id`

Output:
- `document`

## Node: `set_filters`

Purpose:
- Store session filters (document_id, client_name, period).

Input:
- `filters`

Output:
- `filters_applied`

## Node: `explain_sources`

Purpose:
- Provide "how I got this" with chunk citations.

Input:
- `citations`

Output:
- `explanation`
