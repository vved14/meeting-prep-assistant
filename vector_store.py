"""Build a FAISS index of chunk vectors and search it for the most relevant chunks."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import faiss
import numpy as np

import config


@dataclass
class Chunk:
    """A chunk of transcript plus where it came from, so results can cite a source."""
    text: str
    meeting_id: str
    meeting_title: str
    meeting_date: str


def _parse_date(s: str):
    """Best-effort YYYY-MM-DD parse of a chunk's meeting_date; None if unparseable."""
    try:
        return date.fromisoformat((s or "")[:10])
    except (ValueError, TypeError):
        return None


def _recency_weight(meeting_date: str, as_of) -> float:
    """
    Exponential time-decay multiplier for a chunk.

    Returns 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS): a chunk one half-life older
    than the upcoming meeting is worth half as much. Returns 1.0 (no decay) when
    decay is disabled, `as_of` is missing, or the date can't be parsed.
    """
    if as_of is None or config.RECENCY_HALF_LIFE_DAYS <= 0:
        return 1.0
    d = _parse_date(meeting_date)
    if d is None:
        return 1.0
    ref = as_of.date() if hasattr(as_of, "date") else as_of
    age_days = max(0, (ref - d).days)  # future-dated chunks get no boost, no penalty
    return 0.5 ** (age_days / config.RECENCY_HALF_LIFE_DAYS)


class VectorStore:
    def __init__(self, embedder):
        """Store the embedder; the FAISS index and parallel chunk list start empty."""
        self.embedder = embedder
        self.index = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        """
        Embed every chunk's text, L2-normalise, build a FAISS inner-product index of
        the right dimension, add the vectors, and keep `chunks` parallel to the index
        so result positions map back to their source metadata.
        """
        if not chunks:
            raise ValueError("Cannot build a vector store from an empty list of chunks")

        texts = [chunk.text for chunk in chunks]
        vectors = self.embedder.embed(texts)
        vectors = np.asarray(vectors, dtype="float32")

        faiss.normalize_L2(vectors)

        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

        self.chunks = list(chunks)

    def search(self, query: str, k: int, as_of=None) -> list[tuple[Chunk, float]]:
        """
        Embed the query and return the top `k` (Chunk, score) pairs, best first.

        The query is prefixed for bge, and instead of taking FAISS's raw top-k we
        pull a wider candidate pool and re-rank by cosine * recency decay (relative
        to `as_of`, the upcoming meeting's time). `as_of=None` keeps pure cosine
        ranking.
        """
        if self.index is None:
            raise RuntimeError("Index not built yet. Call build() before search().")
        if not self.chunks:
            return []

        # bge wants the QUERY prefixed (documents are embedded as-is).
        prefixed_query = config.BGE_QUERY_PREFIX + query
        query_vec = np.asarray(self.embedder.embed([prefixed_query]), dtype="float32")
        faiss.normalize_L2(query_vec)

        # Widen the pool so recency re-ranking can promote recent chunks that aren't
        # the very top cosine matches.
        pool = min(max(k * config.CANDIDATE_MULTIPLIER, k), len(self.chunks))
        scores, indices = self.index.search(query_vec, pool)

        reranked: list[tuple[Chunk, float]] = []
        for position, score in zip(indices[0], scores[0]):
            if position < 0:  # FAISS pads with -1 when fewer than `pool` exist
                continue
            chunk = self.chunks[position]
            final = float(score) * _recency_weight(chunk.meeting_date, as_of)
            reranked.append((chunk, final))

        reranked.sort(key=lambda pair: pair[1], reverse=True)
        return reranked[:k]
