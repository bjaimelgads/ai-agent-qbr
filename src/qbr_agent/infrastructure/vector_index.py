"""Vector index implementations for QBR retrieval."""

from __future__ import annotations

import math
from dataclasses import dataclass

from qbr_agent.application.ports import KnowledgeRepository, VectorIndex, VectorMatch
from qbr_agent.domain.value_objects import DocumentId, EmbeddingVector


@dataclass
class SqliteEmbeddingVectorIndex(VectorIndex):
    repository: KnowledgeRepository

    async def search(
        self,
        *,
        vector: EmbeddingVector,
        top_k: int,
        document_id: DocumentId | None = None,
        embedding_model: str | None = None,
        min_score: float | None = None,
    ) -> list[VectorMatch]:
        if top_k <= 0:
            return []
        query_vector = vector.values
        if not query_vector:
            return []

        candidates = await self.repository.fetch_chunk_embeddings(
            document_id=document_id,
            embedding_model=embedding_model,
        )
        scored: list[VectorMatch] = []
        for chunk_id, candidate_vector in candidates:
            score = _cosine_similarity(query_vector, candidate_vector.values)
            if min_score is not None and score < min_score:
                continue
            scored.append(VectorMatch(chunk_id=chunk_id, score=score))

        scored.sort(key=lambda match: match.score, reverse=True)
        return scored[:top_k]


def _cosine_similarity(vec_a: tuple[float, ...], vec_b: tuple[float, ...]) -> float:
    if len(vec_a) != len(vec_b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b, strict=False):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)
