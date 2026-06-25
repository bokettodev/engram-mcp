"""Embedder profiles + a cached factory.

All profiles are LOCAL (no cloud API). Two backends:

  FastEmbed (ONNX, light, no torch) — the default / no-GPU path:
    local_fast     - bge-small-en-v1.5 (384d) CPU. Default.
    local_gpu      - bge-small-en-v1.5 (384d) CUDA (falls back to CPU).
    local_quality  - bge-base-en-v1.5 (768d), CUDA if available.

  sentence-transformers (torch, GPU-recommended) — the code-specialized path,
  behind the optional `code` extra (`uv sync --extra code`):
    local_code_fast    - jina-code-embeddings-0.5b (512d MRL). CC-BY-NC-4.0.
    local_code_quality - jina-code-embeddings-1.5b (1024d MRL). CC-BY-NC-4.0.
    local_code_apache  - Qwen3-Embedding-0.6B (1024d). Apache-2.0.

model_id is device-independent but model+dim-specific, so it keys the index +
cache: switching model/dim invalidates cleanly; switching CPU<->GPU does not.
"""

from __future__ import annotations

import os
from functools import lru_cache

from engram_mcp.embeddings.fastembed_provider import FastEmbedProvider

# name -> (model_name, device)
_FASTEMBED: dict[str, tuple[str, str]] = {
    "local_fast": ("BAAI/bge-small-en-v1.5", "cpu"),
    "local_gpu": ("BAAI/bge-small-en-v1.5", "cuda"),
    "local_quality": ("BAAI/bge-base-en-v1.5", "auto"),
}

# name -> (model_name, device, truncate_dim, query_prompt, passage_prompt)
# jina-code prompt keys (verified from the model): nl2code_{query,document} for
# "find code from a natural-language description" retrieval.
_CODE: dict[str, tuple[str, str, int | None, str | None, str | None]] = {
    "local_code_fast": ("jinaai/jina-code-embeddings-0.5b", "auto", 512, "nl2code_query", "nl2code_document"),
    "local_code_quality": ("jinaai/jina-code-embeddings-1.5b", "auto", 1024, "nl2code_query", "nl2code_document"),
    "local_code_apache": ("Qwen/Qwen3-Embedding-0.6B", "auto", 1024, "query", None),
}

PROFILES = frozenset(_FASTEMBED) | frozenset(_CODE)
# Override the default embedder with the ENGRAM_PROFILE env var (e.g. set it to
# local_code_fast once to make jina-code the default everywhere, without code edits).
DEFAULT_PROFILE = os.environ.get("ENGRAM_PROFILE", "local_fast")


@lru_cache(maxsize=8)
def _fastembed(model_name: str, device: str) -> FastEmbedProvider:
    return FastEmbedProvider(model_name, device=device)


@lru_cache(maxsize=8)
def _st(model_name, device, truncate_dim, query_prompt, passage_prompt):
    from engram_mcp.embeddings.sentence_transformers_provider import (
        SentenceTransformersProvider,
    )

    return SentenceTransformersProvider(
        model_name, device=device, truncate_dim=truncate_dim,
        query_prompt=query_prompt, passage_prompt=passage_prompt,
    )


def make_provider(profile: str | None = None):
    profile = profile or DEFAULT_PROFILE
    if profile in _FASTEMBED:
        return _fastembed(*_FASTEMBED[profile])
    if profile in _CODE:
        return _st(*_CODE[profile])
    raise ValueError(f"unknown profile {profile!r}; choices: {', '.join(sorted(PROFILES))}")


def provider_for_model_id(model_id: str, device: str = "auto"):
    """Build the provider matching a manifest's recorded embedder id."""
    if model_id.startswith("fastembed:"):
        return _fastembed(model_id.split(":", 1)[1], device)
    if model_id.startswith("st:"):
        # st:<model>@<dim>#<query_prompt>/<passage_prompt>
        spec, _, ptag = model_id[len("st:"):].partition("#")
        model, _, dim_tag = spec.rpartition("@")
        truncate = None if dim_tag in ("full", "") else int(dim_tag)
        qp, _, pp = ptag.partition("/")
        qp = None if qp in ("none", "") else qp
        pp = None if pp in ("none", "") else pp
        return _st(model, device, truncate, qp, pp)
    raise ValueError(f"unsupported embedder id: {model_id!r}")


def provider_for_project(root, device: str = "auto"):
    """Return the embedder a project was indexed with (so search matches)."""
    from engram_mcp import manifest, paths

    pdir = paths.project_dir(root, create=False)
    m = manifest.load_project(pdir) if pdir.exists() else None
    if m is None or not m.embedder_id:
        return make_provider(DEFAULT_PROFILE)
    return provider_for_model_id(m.embedder_id, device)
