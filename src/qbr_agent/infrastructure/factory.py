"""Factories for QBR infrastructure components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qbr_agent.application.ports import EmbeddingsProvider, KnowledgeRepository, VectorIndex
from qbr_agent.infrastructure.embeddings import HashEmbeddingsProvider, SentenceTransformersEmbeddingsProvider
from qbr_agent.infrastructure.sqlalchemy_gateway import DatabaseGateway
from qbr_agent.infrastructure.sqlalchemy_repository import SqlAlchemyKnowledgeRepository
from qbr_agent.infrastructure.vector_index import SqliteEmbeddingVectorIndex


@dataclass
class InfrastructureBundle:
    repository: KnowledgeRepository
    vector_index: VectorIndex
    embeddings: EmbeddingsProvider


async def build_infrastructure(
    *,
    database_url: str,
    storage_backend: str,
    vector_backend: str,
    embeddings_backend: str,
    embeddings_model: str,
    embeddings_normalize: bool,
    faiss_dir: str,
    faiss_normalize: bool,
) -> InfrastructureBundle:
    storage_backend = storage_backend.lower()
    vector_backend = vector_backend.lower()

    if storage_backend != "sqlite":
        raise ValueError(f"Unsupported STORAGE_BACKEND: {storage_backend}")
    gateway = DatabaseGateway(database_url=database_url)
    repository = SqlAlchemyKnowledgeRepository(sessionmaker=gateway.sessionmaker())

    if vector_backend == "sqlite_embeddings":
        vector_index = SqliteEmbeddingVectorIndex(repository=repository)
    elif vector_backend == "faiss":
        from qbr_agent.infrastructure.faiss_index import FaissVectorIndex  # noqa: PLC0415

        vector_index = FaissVectorIndex(base_dir=Path(faiss_dir), normalize=faiss_normalize)
    else:
        raise ValueError(f"Unsupported VECTOR_BACKEND: {vector_backend}")

    embeddings_backend = embeddings_backend.lower()
    if embeddings_backend == "sentence_transformers":
        embeddings = SentenceTransformersEmbeddingsProvider(
            model_name=embeddings_model,
            normalize=embeddings_normalize,
        )
    else:
        embeddings = HashEmbeddingsProvider()

    return InfrastructureBundle(
        repository=repository,
        vector_index=vector_index,
        embeddings=embeddings,
    )
