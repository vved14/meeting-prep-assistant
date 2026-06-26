"""Turn text into meaning-vectors with a local sentence-transformers model."""
from __future__ import annotations

import numpy as np

import config


def get_embedder():
    """Return the configured local embedder (an object exposing `.embed(texts)`)."""
    return LocalEmbedder(config.LOCAL_EMBEDDING_MODEL)

class LocalEmbedder:
    """Embeds text with a local sentence-transformers model (no API key)."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

