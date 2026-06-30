"""Embedding provider protocol.

Providers split passage vs query embedding because asymmetric models (bge,
e5) use different prefixes for documents and search queries.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    model_id: str  # stable id baked into cache keys (e.g. "fastembed:BAAI/bge-small-en-v1.5")
    dim: int

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...

    def release_unused_cache(self) -> None:
        """Return any GPU memory the backend holds but isn't using back to the
        device (e.g. the activation high-water from a bulk index). Does NOT
        unload the model. No-op for CPU/ONNX backends."""
        ...
