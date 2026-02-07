"""Build metric fact embeddings and FAISS index."""

from __future__ import annotations

import asyncio
import hashlib
import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from qbr_intelligence.db.models import Base, MetricFactEmbedding
from qbr_intelligence.metric_qa.dao import MetricFactStore


def _build_metric_fact_text(row: dict) -> str:
    parts: list[str] = []
    metric = row.get("metric_name") or row.get("label_text")
    if metric:
        parts.append(f"Metric: {metric}")
    if row.get("llm_context_label"):
        parts.append(f"Context: {row['llm_context_label']}")
    if row.get("snippet"):
        parts.append(f"Snippet: {row['snippet']}")
    if row.get("slide_title"):
        parts.append(f"Slide: {row['slide_title']}")
    if row.get("period_label"):
        parts.append(f"Period: {row['period_label']}")
    if row.get("client_name"):
        parts.append(f"Client: {row['client_name']}")
    if row.get("region"):
        parts.append(f"Region: {row['region']}")
    if row.get("document_name"):
        parts.append(f"Document: {row['document_name']}")
    return " | ".join(parts)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def _ensure_tables(engine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def _build_embeddings(
    *,
    database_url: str,
    model_name: str,
    normalize: bool,
    batch_size: int,
    force_rebuild: bool,
) -> int:
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError("sentence-transformers is required to build embeddings") from exc

    engine = create_async_engine(database_url)
    await _ensure_tables(engine)

    store = MetricFactStore(database_url)
    await store.ensure_view()

    model = SentenceTransformer(model_name)
    model_label = f"sentence-transformers:{model_name}"

    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    total_updated = 0
    offset = 0
    while True:
        rows = await store.fetch_metric_fact_embedding_inputs(limit=batch_size, offset=offset)
        if not rows:
            break
        fact_ids = [int(row["fact_id"]) for row in rows]
        existing: dict[int, tuple[str | None, str | None]] = {}
        async with sessionmaker() as session:
            result = await session.execute(
                select(
                    MetricFactEmbedding.metric_id,
                    MetricFactEmbedding.text_hash,
                    MetricFactEmbedding.embedding_model,
                ).where(MetricFactEmbedding.metric_id.in_(fact_ids))
            )
            for metric_id, text_hash, embedding_model in result.all():
                existing[int(metric_id)] = (text_hash, embedding_model)

        pending_ids: list[int] = []
        pending_hashes: list[str] = []
        pending_texts: list[str] = []
        for row in rows:
            fact_id = int(row["fact_id"])
            text = _build_metric_fact_text(row)
            if not text:
                continue
            text_hash = _hash_text(text)
            if not force_rebuild:
                cached = existing.get(fact_id)
                if cached and cached[0] == text_hash and cached[1] == model_label:
                    continue
            pending_ids.append(fact_id)
            pending_hashes.append(text_hash)
            pending_texts.append(text)

        if pending_texts:
            vectors = model.encode(
                pending_texts,
                normalize_embeddings=normalize,
            )
            if hasattr(vectors, "tolist"):
                vectors = vectors.tolist()
            payloads = []
            for fact_id, text_hash, vector in zip(
                pending_ids,
                pending_hashes,
                vectors,
                strict=False,
            ):
                payloads.append(
                    {
                        "metric_id": fact_id,
                        "embedding": [float(value) for value in vector],
                        "embedding_model": model_label,
                        "text_hash": text_hash,
                        "updated_at": datetime.utcnow(),
                    }
                )
            async with sessionmaker() as session:
                await session.execute(
                    delete(MetricFactEmbedding).where(MetricFactEmbedding.metric_id.in_(pending_ids))
                )
                await session.execute(MetricFactEmbedding.__table__.insert(), payloads)
                await session.commit()
            total_updated += len(payloads)

        offset += len(rows)

    await engine.dispose()
    return total_updated


async def _build_faiss_index(
    *,
    database_url: str,
    base_dir: Path,
    normalize: bool,
    metric: str,
    embedding_model_filter: str | None,
    force: bool,
) -> bool:
    try:  # pragma: no cover - optional dependency
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("numpy is required to build FAISS indices") from exc
    try:  # pragma: no cover - optional dependency
        import faiss  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("faiss is required to build FAISS indices") from exc

    index_path = base_dir / "index.faiss"
    ids_path = base_dir / "index_ids.json"
    if not force and index_path.exists() and ids_path.exists():
        return False

    engine = create_async_engine(database_url)
    await _ensure_tables(engine)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)

    async with sessionmaker() as session:
        query = select(
            MetricFactEmbedding.metric_id,
            MetricFactEmbedding.embedding,
            MetricFactEmbedding.embedding_model,
        ).where(MetricFactEmbedding.embedding.is_not(None))
        if embedding_model_filter:
            query = query.where(MetricFactEmbedding.embedding_model == embedding_model_filter)
        result = await session.execute(query)
        rows = result.all()

    if not rows:
        await engine.dispose()
        return False

    ids = [int(row.metric_id) for row in rows]
    vectors = [list(row.embedding) for row in rows]

    matrix = np.asarray(vectors, dtype="float32")
    if normalize:
        faiss.normalize_L2(matrix)

    dimension = matrix.shape[1]
    metric = metric.lower()
    if metric == "ip":
        index = faiss.IndexFlatIP(dimension)
    elif metric == "l2":
        index = faiss.IndexFlatL2(dimension)
    else:
        raise ValueError(f"Unsupported METRIC_FAISS_METRIC: {metric}")

    index.add(matrix)

    base_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    ids_payload = {
        "ids": ids,
        "embedding_model": embedding_model_filter,
        "metric": metric,
        "normalize": normalize,
    }
    ids_path.write_text(
        __import__("json").dumps(ids_payload),
        encoding="utf-8",
    )
    await engine.dispose()
    return True


async def main() -> None:
    database_url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///qbr_intelligence.db")
    model_name = os.getenv("METRIC_EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
    normalize = os.getenv("METRIC_EMBEDDINGS_NORMALIZE", "true").lower() in {"1", "true", "yes"}
    batch_size = int(os.getenv("METRIC_EMBEDDINGS_BATCH_SIZE", "64"))
    force_rebuild = os.getenv("METRIC_EMBEDDINGS_FORCE_REBUILD", "").lower() in {"1", "true", "yes"}

    updated = await _build_embeddings(
        database_url=database_url,
        model_name=model_name,
        normalize=normalize,
        batch_size=batch_size,
        force_rebuild=force_rebuild,
    )
    print(f"[metric-embeddings] Updated {updated} metric fact embeddings.")

    faiss_dir = Path(os.getenv("METRIC_FAISS_DIR", "./data/faiss_metric"))
    faiss_normalize = os.getenv("METRIC_FAISS_NORMALIZE", "true").lower() in {"1", "true", "yes"}
    faiss_metric = os.getenv("METRIC_FAISS_METRIC", "ip")
    faiss_force = os.getenv("METRIC_FAISS_FORCE_REBUILD", "").lower() in {"1", "true", "yes"}
    embedding_model_filter = os.getenv("METRIC_FAISS_EMBEDDING_MODEL_FILTER", "").strip() or None
    if embedding_model_filter is None:
        embedding_model_filter = f"sentence-transformers:{model_name}"

    built = await _build_faiss_index(
        database_url=database_url,
        base_dir=faiss_dir,
        normalize=faiss_normalize,
        metric=faiss_metric,
        embedding_model_filter=embedding_model_filter,
        force=faiss_force,
    )
    if built:
        print(f"[metric-faiss] Index built at {faiss_dir}")
    else:
        print("[metric-faiss] Index already exists or no embeddings found.")


if __name__ == "__main__":
    asyncio.run(main())
