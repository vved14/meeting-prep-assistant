"""Cache embeddings to disk so repeat runs skip re-embedding text we've seen."""
from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path

import numpy as np

import config


def _key(text: str) -> str:
    """Stable hex hash for `text` — same text always maps to the same cache key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_cache(path) -> dict[str, np.ndarray]:
    """Return the {hash: vector} dict saved at `path`, or {} if there is no file yet."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def save_cache(cache: dict[str, np.ndarray], path) -> None:
    """Persist the {hash: vector} dict to `path` atomically (temp file + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)  # atomic on the same filesystem: a crash can't half-write the cache


class CachingEmbedder:
    """Drop-in embedder that remembers vectors it has already computed."""

    def __init__(self, embedder, path=config.EMBED_CACHE_PATH):
        self.embedder = embedder
        self.path = path
        self.cache = load_cache(path)

    def embed(self, texts: list[str]) -> np.ndarray:
        keys = [_key(t) for t in texts]

        misses, seen = [], set()
        for text, key in zip(texts, keys):
            if key not in self.cache and key not in seen:
                seen.add(key)
                misses.append(text)

        if misses:
            new_vecs = np.asarray(self.embedder.embed(misses), dtype=np.float32)
            for text, vec in zip(misses, new_vecs):
                self.cache[_key(text)] = vec
            save_cache(self.cache, self.path)

        return np.asarray([self.cache[k] for k in keys], dtype=np.float32)
