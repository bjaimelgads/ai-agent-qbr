import json

import pytest

kreuzberg = pytest.importorskip("kreuzberg")
config_to_json = kreuzberg.config_to_json

from qbr_intelligence.pipeline.embeddings import EmbeddingSettings
from qbr_intelligence.pipeline.processor import QBRProcessor


def test_embedding_settings_defaults():
    settings = EmbeddingSettings.from_env({})
    assert settings.enabled is True
    assert settings.preset == "fast"


def test_embedding_settings_disabled():
    settings = EmbeddingSettings.from_env({"KREUZBERG_EMBEDDINGS_ENABLED": "false"})
    assert settings.enabled is False
    assert settings.to_embedding_config() is None


def test_extraction_config_includes_embedding():
    env = {
        "KREUZBERG_EMBEDDINGS_ENABLED": "true",
        "KREUZBERG_EMBEDDINGS_PRESET": "fast",
    }
    settings = EmbeddingSettings.from_env(env)
    processor = QBRProcessor(embedding_settings=settings)
    config = processor.create_extraction_config()
    payload = json.loads(config_to_json(config))
    embedding = payload["chunking"]["embedding"]
    assert embedding["model"]["type"] == "preset"
    assert embedding["model"]["name"] == "fast"
