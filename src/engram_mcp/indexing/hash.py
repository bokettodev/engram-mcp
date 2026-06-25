"""Content hashing for cache keys and incremental change detection."""

from __future__ import annotations

import hashlib


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embedding_input_hash(embedder_id: str, chunker_version: str, text: str) -> str:
    """Key under which a chunk's vector is cached.

    Includes the embedder id and chunker version so a model swap or a chunking
    change naturally misses the cache instead of returning stale vectors.
    """
    h = hashlib.sha256()
    h.update(embedder_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(chunker_version.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()
