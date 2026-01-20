"""Application use cases for the QBR agent."""

from __future__ import annotations

from dataclasses import dataclass

from qbr_agent.application.ports import (
    EmbeddingsProvider,
    KnowledgeRepository,
    TextMatch,
    VectorIndex,
    VectorMatch,
)
from qbr_agent.domain.entities import Answer, RetrievalResult
from qbr_agent.domain.value_objects import ChunkId, DocumentId, Score


@dataclass
class SearchKnowledge:
    repository: KnowledgeRepository
    vector_index: VectorIndex
    embeddings: EmbeddingsProvider

    async def execute(
        self,
        *,
        query: str,
        document_id: int | None = None,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:
        embedding_result = await self.embeddings.embed_query(query)
        matches = await self.vector_index.search(
            vector=embedding_result.vector,
            top_k=top_k,
            document_id=DocumentId(document_id) if document_id is not None else None,
            embedding_model=embedding_result.model,
            min_score=min_score,
        )
        if not matches:
            return []

        chunk_map = {
            chunk.chunk_id.value: chunk
            for chunk in await self.repository.fetch_chunks_by_ids(
                [match.chunk_id for match in matches]
            )
        }
        results: list[RetrievalResult] = []
        for match in matches:
            chunk = chunk_map.get(match.chunk_id.value)
            if chunk is None:
                continue
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=Score(match.score),
                )
            )
        return results


@dataclass
class HybridSearchKnowledge:
    repository: KnowledgeRepository
    vector_index: VectorIndex
    embeddings: EmbeddingsProvider
    text_weight: float = 0.6
    vector_weight: float = 0.4
    candidate_multiplier: int = 4

    async def execute(
        self,
        *,
        query: str,
        document_id: int | None = None,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:
        embedding_result = await self.embeddings.embed_query(query)
        candidate_limit = max(top_k * self.candidate_multiplier, top_k)
        vector_matches = await self.vector_index.search(
            vector=embedding_result.vector,
            top_k=candidate_limit,
            document_id=DocumentId(document_id) if document_id is not None else None,
            embedding_model=embedding_result.model,
            min_score=None,
        )
        text_matches = await self.repository.search_chunks_text(
            query=query,
            limit=candidate_limit,
            document_id=DocumentId(document_id) if document_id is not None else None,
        )
        combined = _combine_hybrid_scores(
            vector_matches=vector_matches,
            text_matches=text_matches,
            text_weight=self.text_weight,
            vector_weight=self.vector_weight,
        )
        if min_score is not None:
            combined = [item for item in combined if item.score >= min_score]
        combined = combined[:top_k]
        if not combined:
            return []

        chunk_map = {
            chunk.chunk_id.value: chunk
            for chunk in await self.repository.fetch_chunks_by_ids(
                [match.chunk_id for match in combined]
            )
        }
        results: list[RetrievalResult] = []
        for match in combined:
            chunk = chunk_map.get(match.chunk_id.value)
            if chunk is None:
                continue
            results.append(RetrievalResult(chunk=chunk, score=Score(match.score)))
        return results


@dataclass
class AnswerQuestion:
    repository: KnowledgeRepository
    vector_index: VectorIndex
    embeddings: EmbeddingsProvider
    use_hybrid: bool = True
    text_weight: float = 0.6
    vector_weight: float = 0.4
    candidate_multiplier: int = 4

    async def execute(
        self,
        *,
        query: str,
        document_id: int | None = None,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> Answer:
        if self.use_hybrid:
            search = HybridSearchKnowledge(
                repository=self.repository,
                vector_index=self.vector_index,
                embeddings=self.embeddings,
                text_weight=self.text_weight,
                vector_weight=self.vector_weight,
                candidate_multiplier=self.candidate_multiplier,
            )
            results = await search.execute(
                query=query,
                document_id=document_id,
                top_k=top_k,
                min_score=min_score,
            )
        else:
            search = SearchKnowledge(
                repository=self.repository,
                vector_index=self.vector_index,
                embeddings=self.embeddings,
            )
            results = await search.execute(
                query=query,
                document_id=document_id,
                top_k=top_k,
                min_score=min_score,
            )
        context_lines = []
        for idx, result in enumerate(results, start=1):
            chunk = result.chunk
            slide_range = _format_slide_range(chunk.start_slide, chunk.end_slide)
            label = f"[{idx}] Doc {chunk.document_id.value}{slide_range}"
            meta = chunk.metadata or {}
            extras = []
            title = meta.get("slide_title")
            if title:
                extras.append(f"Title: {title}")
            key_message = meta.get("slide_key_message")
            if key_message:
                extras.append(f"Key message: {key_message}")
            if extras:
                context_lines.append(f"{label}: {' | '.join(extras)}\n{chunk.content}")
            else:
                context_lines.append(f"{label}: {chunk.content}")
        context_text = "\n\n".join(context_lines)

        return Answer(
            question=query,
            text=None,
            context=context_text,
            citations=results,
        )


@dataclass
class IngestPpt:
    """Placeholder for future ingestion wiring."""

    async def execute(self, *, file_path: str) -> None:
        raise NotImplementedError("IngestPpt is not wired yet")


@dataclass
class StreamAnswer:
    """Placeholder for future streaming use case."""

    async def execute(self, *, query: str) -> None:
        raise NotImplementedError("StreamAnswer is not wired yet")


def _format_slide_range(start: int | None, end: int | None) -> str:
    if start is None and end is None:
        return ""
    if start is None:
        return f" (slides ?-{end})"
    if end is None or end == start:
        return f" (slide {start})"
    return f" (slides {start}-{end})"


def _combine_hybrid_scores(
    *,
    vector_matches: list[VectorMatch],
    text_matches: list[TextMatch],
    text_weight: float,
    vector_weight: float,
) -> list[VectorMatch]:
    vector_by_id = {match.chunk_id.value: match.score for match in vector_matches}
    text_by_id = {match.chunk_id.value: match.score for match in text_matches}
    candidate_ids = set(vector_by_id) | set(text_by_id)
    if not candidate_ids:
        return []
    max_vector = max(vector_by_id.values(), default=0.0)
    max_text = max(text_by_id.values(), default=0.0)
    combined: list[VectorMatch] = []
    for chunk_id in candidate_ids:
        vector_score = vector_by_id.get(chunk_id, 0.0)
        text_score = text_by_id.get(chunk_id, 0.0)
        vector_norm = vector_score / max_vector if max_vector else 0.0
        text_norm = text_score / max_text if max_text else 0.0
        hybrid_score = (text_norm * text_weight) + (vector_norm * vector_weight)
        combined.append(VectorMatch(chunk_id=ChunkId(chunk_id), score=hybrid_score))
    combined.sort(key=lambda match: match.score, reverse=True)
    return combined
