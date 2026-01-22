"""Application use cases for the QBR agent."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math

from qbr_agent.application.ports import (
    EmbeddingsProvider,
    KnowledgeRepository,
    Reranker,
    TextMatch,
    VectorIndex,
    VectorMatch,
)
from qbr_agent.domain.entities import Answer, Chunk, RetrievalResult
from qbr_agent.domain.value_objects import ChunkId, DocumentId, EmbeddingVector, Score


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
    reranker: Reranker | None = None
    text_weight: float = 0.6
    vector_weight: float = 0.4
    candidate_multiplier: int = 4
    rerank_top_n: int = 20
    mmr_lambda: float = 0.5
    max_chunks_per_doc: int = 3
    last_debug: dict | None = None

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
        logger = logging.getLogger(__name__)
        if logger.isEnabledFor(logging.DEBUG):
            vector_top = vector_matches[0].score if vector_matches else None
            text_top = text_matches[0].score if text_matches else None
            logger.debug(
                "Hybrid retrieval: vector_matches=%d (top=%s) text_matches=%d (top=%s)",
                len(vector_matches),
                vector_top,
                len(text_matches),
                text_top,
            )
        combined = _combine_hybrid_scores(
            vector_matches=vector_matches,
            text_matches=text_matches,
            text_weight=self.text_weight,
            vector_weight=self.vector_weight,
        )
        if min_score is not None:
            combined = [item for item in combined if item.score >= min_score]
        if not combined:
            return []

        combined = combined[:candidate_limit]
        chunk_map = {
            chunk.chunk_id.value: chunk
            for chunk in await self.repository.fetch_chunks_by_ids(
                [match.chunk_id for match in combined]
            )
        }
        ordered_chunks = [
            chunk_map[match.chunk_id.value]
            for match in combined
            if match.chunk_id.value in chunk_map
        ]
        if not ordered_chunks:
            return []

        combined_scores = {match.chunk_id.value: match.score for match in combined}
        reranked_chunks, rerank_scores = await _apply_rerank(
            query=query,
            chunks=ordered_chunks,
            reranker=self.reranker,
            rerank_top_n=self.rerank_top_n,
            scores=combined_scores,
        )
        selected_chunks, mmr_selected = _apply_mmr(
            query_vector=embedding_result.vector,
            chunks=reranked_chunks,
            top_k=top_k,
            mmr_lambda=self.mmr_lambda,
        )
        final_chunks = _apply_doc_cap(
            selected_chunks=selected_chunks,
            candidate_chunks=reranked_chunks,
            max_chunks_per_doc=self.max_chunks_per_doc,
            top_k=top_k,
        )
        self.last_debug = _build_retrieval_debug(
            query=query,
            vector_matches=vector_matches,
            text_matches=text_matches,
            combined=combined,
            reranked_chunks=reranked_chunks,
            rerank_scores=rerank_scores,
            mmr_selected=mmr_selected,
            final_chunks=final_chunks,
        )

        results: list[RetrievalResult] = []
        for chunk in final_chunks:
            score_value = combined_scores.get(chunk.chunk_id.value, 0.0)
            results.append(RetrievalResult(chunk=chunk, score=Score(score_value)))
        return results


@dataclass
class AnswerQuestion:
    repository: KnowledgeRepository
    vector_index: VectorIndex
    embeddings: EmbeddingsProvider
    reranker: Reranker | None = None
    use_hybrid: bool = True
    text_weight: float = 0.6
    vector_weight: float = 0.4
    candidate_multiplier: int = 4
    include_document_path: bool = False
    rerank_top_n: int = 20
    mmr_lambda: float = 0.5
    max_chunks_per_doc: int = 3
    last_debug: dict | None = None

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
                reranker=self.reranker,
                text_weight=self.text_weight,
                vector_weight=self.vector_weight,
                candidate_multiplier=self.candidate_multiplier,
                rerank_top_n=self.rerank_top_n,
                mmr_lambda=self.mmr_lambda,
                max_chunks_per_doc=self.max_chunks_per_doc,
            )
            results = await search.execute(
                query=query,
                document_id=document_id,
                top_k=top_k,
                min_score=min_score,
            )
            if isinstance(search, HybridSearchKnowledge):
                self.last_debug = search.last_debug
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
        document_ids = {result.chunk.document_id.value for result in results}
        documents = await self.repository.fetch_documents_by_ids(
            [DocumentId(doc_id) for doc_id in document_ids]
        )
        documents_by_id = {doc.document_id.value: doc for doc in documents}
        context_lines = []
        for idx, result in enumerate(results, start=1):
            chunk = result.chunk
            slide_range = _format_slide_range(chunk.start_slide, chunk.end_slide)
            label = f"[{idx}] Doc {chunk.document_id.value}{slide_range}"
            meta = chunk.metadata or {}
            extras = []
            doc = documents_by_id.get(chunk.document_id.value)
            if doc and doc.filename:
                extras.append(f"File: {doc.filename}")
            if self.include_document_path and doc and doc.file_path:
                extras.append(f"Path: {doc.file_path}")
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


async def _apply_rerank(
    *,
    query: str,
    chunks: list[Chunk],
    reranker: Reranker | None,
    rerank_top_n: int,
    scores: dict[int, float],
) -> tuple[list[Chunk], dict[int, float]]:
    if reranker is None or rerank_top_n <= 0:
        return chunks, {}
    top_n = min(rerank_top_n, len(chunks))
    top_chunks = chunks[:top_n]
    rerank_scores = await reranker.score(query=query, chunks=top_chunks)
    if len(rerank_scores) != top_n:
        return chunks, {}
    if rerank_scores:
        logger = logging.getLogger(__name__)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Reranker applied to %d chunks; top score=%s", top_n, rerank_scores[0])
    reranked = sorted(
        zip(top_chunks, rerank_scores, strict=False),
        key=lambda item: item[1],
        reverse=True,
    )
    rerank_map: dict[int, float] = {}
    for chunk, score in reranked:
        scores[chunk.chunk_id.value] = score
        rerank_map[chunk.chunk_id.value] = float(score)
    return [chunk for chunk, _score in reranked] + chunks[top_n:], rerank_map


def _apply_mmr(
    *,
    query_vector: EmbeddingVector,
    chunks: list[Chunk],
    top_k: int,
    mmr_lambda: float,
) -> tuple[list[Chunk], list[int]]:
    if top_k <= 0 or len(chunks) <= 1:
        return chunks[:top_k], list(range(min(top_k, len(chunks))))
    if not query_vector.values:
        return chunks[:top_k], list(range(min(top_k, len(chunks))))
    embeddings: list[EmbeddingVector] = []
    for chunk in chunks:
        embedding = chunk.embedding.vector if chunk.embedding else None
        if embedding is None or not embedding.values:
            return chunks[:top_k], list(range(min(top_k, len(chunks))))
        embeddings.append(embedding)

    selected_indices: list[int] = []
    candidate_indices = list(range(len(chunks)))
    while candidate_indices and len(selected_indices) < top_k:
        best_idx = None
        best_score = None
        for idx in candidate_indices:
            query_sim = _cosine_similarity(query_vector, embeddings[idx])
            if selected_indices:
                max_sim = max(
                    _cosine_similarity(embeddings[idx], embeddings[sel_idx])
                    for sel_idx in selected_indices
                )
            else:
                max_sim = 0.0
            score = (mmr_lambda * query_sim) - ((1 - mmr_lambda) * max_sim)
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected_indices.append(best_idx)
        candidate_indices.remove(best_idx)
    return [chunks[idx] for idx in selected_indices], selected_indices


def _apply_doc_cap(
    *,
    selected_chunks: list[Chunk],
    candidate_chunks: list[Chunk],
    max_chunks_per_doc: int,
    top_k: int,
) -> list[Chunk]:
    if max_chunks_per_doc <= 0:
        return selected_chunks[:top_k]
    counts: dict[int, int] = {}
    selected_ids = {chunk.chunk_id.value for chunk in selected_chunks}
    ordered_candidates = selected_chunks + [
        chunk for chunk in candidate_chunks if chunk.chunk_id.value not in selected_ids
    ]
    final: list[Chunk] = []
    for chunk in ordered_candidates:
        if len(final) >= top_k:
            break
        doc_id = chunk.document_id.value
        if counts.get(doc_id, 0) >= max_chunks_per_doc:
            continue
        counts[doc_id] = counts.get(doc_id, 0) + 1
        final.append(chunk)
    return final


def _cosine_similarity(vec_a: EmbeddingVector, vec_b: EmbeddingVector) -> float:
    if len(vec_a.values) != len(vec_b.values):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a.values, vec_b.values, strict=False):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def _build_retrieval_debug(
    *,
    query: str,
    vector_matches: list[VectorMatch],
    text_matches: list[TextMatch],
    combined: list[VectorMatch],
    reranked_chunks: list[Chunk],
    rerank_scores: dict[int, float],
    mmr_selected: list[int],
    final_chunks: list[Chunk],
) -> dict:
    return {
        "query": query,
        "vector_matches": [
            {"chunk_id": match.chunk_id.value, "score": match.score}
            for match in vector_matches
        ],
        "text_matches": [
            {"chunk_id": match.chunk_id.value, "score": match.score}
            for match in text_matches
        ],
        "hybrid_combined": [
            {"chunk_id": match.chunk_id.value, "score": match.score}
            for match in combined
        ],
        "rerank_scores": rerank_scores,
        "mmr_selected_chunk_ids": [
            reranked_chunks[idx].chunk_id.value
            for idx in mmr_selected
            if idx < len(reranked_chunks)
        ],
        "final_chunk_ids": [chunk.chunk_id.value for chunk in final_chunks],
    }
