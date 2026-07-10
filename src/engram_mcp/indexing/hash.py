"""Content hashing for cache keys and incremental change detection."""

from __future__ import annotations

import hashlib


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embedding_input_hash(
    embedder_id: str, chunker_version: str, text: str, artifact_digest: str = ""
) -> str:
    """Key under which a chunk's vector is cached.

    Includes the embedder id (repo + pinned revision + dim + pooling +
    backend, see embeddings/hf_pin.py::canonical_model_id) and chunker version
    so a model swap, a revision bump, or a chunking change naturally misses
    the cache instead of returning stale vectors.

    ``artifact_digest`` is additional defense in depth: a best-effort content
    digest of the actual model artifact loaded, folded in when the caller has
    one (see ``embeddings/base.py``'s ``EmbeddingProvider.artifact_digest``).
    Passing ``""`` (the default -- used by callers that can't cheaply resolve
    it, e.g. the torch-free index-planning path) reproduces the pre-digest
    hash; it does not weaken the embedder-id/revision component above.
    """
    h = hashlib.sha256()
    h.update(embedder_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(chunker_version.encode("utf-8"))
    h.update(b"\x00")
    h.update(artifact_digest.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()
