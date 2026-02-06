"""Post-extraction embeddings using SentenceTransformers."""

from __future__ import annotations

import os
from typing import Mapping

from pydantic import BaseModel, Field


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


class PostEmbeddingSettings(BaseModel):
    """Environment-driven settings for post-extraction embeddings."""

    enabled: bool = Field(default=True)
    model: str = Field(default="all-MiniLM-L6-v2")
    device: str | None = Field(default="cpu")
    batch_size: int = Field(default=32)
    normalize: bool = Field(default=True)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "PostEmbeddingSettings":
        source = environ or os.environ
        return cls(
            enabled=_parse_bool(source.get("QBR_POST_EMBEDDINGS_ENABLED"), True),
            model=source.get("QBR_POST_EMBEDDINGS_MODEL", "all-MiniLM-L6-v2").strip(),
            device=(source.get("QBR_POST_EMBEDDINGS_DEVICE") or "cpu").strip(),
            batch_size=_parse_int(source.get("QBR_POST_EMBEDDINGS_BATCH_SIZE"), 32),
            normalize=_parse_bool(source.get("QBR_POST_EMBEDDINGS_NORMALIZE"), True),
        )

    def model_label(self) -> str:
        return f"sentence-transformers:{self.model}"


def apply_post_embeddings(chunks: list[dict], settings: PostEmbeddingSettings) -> int:
    """Populate missing embeddings on chunks in-place."""

    if not settings.enabled:
        return 0

    missing_indices: list[int] = []
    texts: list[str] = []
    for idx, chunk in enumerate(chunks):
        if chunk.get("embedding") is None:
            missing_indices.append(idx)
            texts.append(chunk.get("content", ""))

    if not missing_indices:
        return 0

    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        msg = (
            "sentence-transformers is not installed. "
            "Install it or disable QBR_POST_EMBEDDINGS_ENABLED."
        )
        raise RuntimeError(msg) from exc

    model = SentenceTransformer(settings.model, device=settings.device or "cpu")
    vectors = model.encode(
        texts,
        batch_size=settings.batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=settings.normalize,
    )
    if hasattr(vectors, "tolist"):
        vectors_list = vectors.tolist()
    else:
        vectors_list = list(vectors)

    if len(vectors_list) != len(texts):
        raise RuntimeError(
            f"Embedding client returned {len(vectors_list)} vectors for {len(texts)} texts."
        )

    model_label = settings.model_label()
    for idx, vector in zip(missing_indices, vectors_list, strict=False):
        chunks[idx]["embedding"] = [float(value) for value in vector]
        chunks[idx]["embedding_model"] = model_label
    return len(missing_indices)
