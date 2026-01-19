"""Embedding settings and helpers for Kreuzberg integration."""

from __future__ import annotations

import os
from typing import Mapping

from pydantic import BaseModel, Field

from kreuzberg import EmbeddingConfig, EmbeddingModelType, list_embedding_presets


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


class EmbeddingSettings(BaseModel):
    """Environment-driven embedding settings for Kreuzberg."""

    enabled: bool = Field(default=True)
    preset: str = Field(default="fast")
    normalize: bool = Field(default=True)
    batch_size: int = Field(default=32)
    show_download_progress: bool = Field(default=False)
    cache_dir: str | None = Field(default=None)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "EmbeddingSettings":
        source = environ or os.environ
        return cls(
            enabled=_parse_bool(source.get("KREUZBERG_EMBEDDINGS_ENABLED"), True),
            preset=source.get("KREUZBERG_EMBEDDINGS_PRESET", "fast").strip(),
            normalize=_parse_bool(source.get("KREUZBERG_EMBEDDINGS_NORMALIZE"), True),
            batch_size=_parse_int(source.get("KREUZBERG_EMBEDDINGS_BATCH_SIZE"), 32),
            show_download_progress=_parse_bool(
                source.get("KREUZBERG_EMBEDDINGS_SHOW_DOWNLOAD_PROGRESS"), False
            ),
            cache_dir=source.get("KREUZBERG_EMBEDDINGS_CACHE_DIR") or None,
        )

    def to_embedding_config(self) -> EmbeddingConfig | None:
        if not self.enabled:
            return None
        presets = set(list_embedding_presets())
        if self.preset not in presets:
            raise ValueError(
                f"Unknown embedding preset '{self.preset}'. Available: {sorted(presets)}"
            )
        return EmbeddingConfig(
            model=EmbeddingModelType.preset(self.preset),
            normalize=self.normalize,
            batch_size=self.batch_size,
            show_download_progress=self.show_download_progress,
            cache_dir=self.cache_dir,
        )

    def model_label(self) -> str | None:
        if not self.enabled:
            return None
        return f"kreuzberg:{self.preset}"
