"""Resolve a Hugging Face model snapshot pinned to an exact revision.

FastEmbed (as vendored in this project's lockfile) resolves ONNX model
downloads through ``ModelManagement.download_model``, which never threads a
``revision`` kwarg from ``TextEmbedding``/``TextCrossEncoder`` down to its
internal ``huggingface_hub.snapshot_download`` call -- it always resolves and
downloads whatever is at the tip of the source repo's default branch
(``model_info(hf_source_repo).sha``, unconditionally). So a plain
``TextEmbedding(model_name)`` cannot be pinned through FastEmbed's own public
API.

To get a real pin anyway, we resolve the exact-revision snapshot ourselves via
``huggingface_hub.snapshot_download(..., revision=<sha>)`` and hand FastEmbed
the resulting local directory through its ``specific_model_path`` escape
hatch (``ModelManagement.download_model`` returns that path verbatim,
bypassing FastEmbed's own revision-less download entirely).

sentence-transformers accepts ``revision=`` directly and does not need this;
it is used here only to compute a best-effort artifact digest for that
backend too (see ``blob_digest``).
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Config/tokenizer files needed by both FastEmbed's ONNX loader and
# sentence-transformers, small enough to always fetch alongside the weight
# file being pinned.
_BASE_PATTERNS = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
)


def canonical_model_id(
    *, backend: str, repo_id: str, revision: str, dim: int, pooling: str, normalize: bool = True
) -> str:
    """Build the identity string stored as the manifest ``embedder_id`` and
    folded into the global embedding-cache key.

    Includes everything that changes the vectors a caller gets back: the repo,
    the pinned revision (a bare repo name is mutable upstream -- see
    config.py), the output dimension, the pooling strategy, and whether
    output vectors are L2-normalized. ``backend`` records which *loader*
    format the manifest expects at query time (engram always queries via
    FastEmbed/ONNX CPU, even for a CUDA-built index -- see
    ``embeddings/factory.py``'s canonical/backend id split), not which
    backend happened to build the index.

    Deliberately excludes anything that doesn't change the vectors (device,
    batch size, thread count): those live in ``backend_id`` instead, which is
    for loaded-model diagnostics/accounting, not compatibility.
    """
    norm = "1" if normalize else "0"
    return f"{backend}:{repo_id}@{revision}#dim={dim};pool={pooling.lower()};norm={norm}"


def pinned_snapshot_dir(
    repo_id: str, revision: str, *, extra_patterns: tuple[str, ...] = ()
) -> str:
    """Download (or reuse the local cache of) the exact-revision snapshot.

    Returns a local directory path. ``revision`` must be a commit SHA, not a
    branch name, so the download is reproducible and safe to cache
    indefinitely (an immutable revision never needs revalidation against the
    remote once it is on disk).
    """
    from huggingface_hub import snapshot_download

    patterns = [*_BASE_PATTERNS, *extra_patterns]
    return snapshot_download(repo_id=repo_id, revision=revision, allow_patterns=patterns)


def blob_digest(snapshot_dir: str, filename: str) -> str:
    """Cheap content-identity for one artifact file in a resolved snapshot.

    huggingface_hub's local cache stores each downloaded file as a symlink
    into a content-addressed blob store, named by the file's own content
    hash. Reading that symlink target is a stat, not a rehash of a
    (potentially hundreds-of-MB) model file, so this stays cheap even for the
    ONNX weight file itself.

    Falls back to hashing the file's bytes directly when the cache layout
    doesn't use symlinks (e.g. ``HF_HUB_LOCAL_DIR_USE_SYMLINKS=0``, or a
    platform where huggingface_hub falls back to copies). Returns ``""`` if
    the file can't be found at all -- callers must treat that as "digest
    unavailable", not a fatal error: it is best-effort defense in depth on
    top of the revision pin, not the primary compatibility gate.
    """
    path = Path(snapshot_dir) / filename
    try:
        if path.is_symlink():
            target = os.readlink(path)
            return Path(target).name
        if path.is_file():
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()
    except OSError as exc:
        logger.debug("could not compute artifact digest for %s: %r", path, exc)
    return ""
