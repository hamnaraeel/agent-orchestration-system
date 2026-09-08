"""Deterministic, offline embedding function for tests: a hashed bag-of-words
vector. Avoids network access / model downloads that Chroma's real default
embedding function would need.
"""
from __future__ import annotations

import hashlib
import math

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

_DIM = 32


class FakeEmbeddingFunction(EmbeddingFunction):
    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            vector = [0.0] * _DIM
            for word in text.lower().split():
                bucket = int(hashlib.sha256(word.encode()).hexdigest(), 16) % _DIM
                vector[bucket] += 1.0
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / norm for v in vector])
        return vectors

    @staticmethod
    def name() -> str:
        return "fake-hashed-bow"
