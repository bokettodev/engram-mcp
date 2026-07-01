"""Embedder profiles + a cached factory.

All profiles are LOCAL (no cloud API), Apache-2.0, and multilingual (incl.
Russian) + code. Naming: local_<cpu|gpu>_<small|large>. Two backends:

  cpu — FastEmbed/ONNX, no torch, ~0 VRAM (Granite R2). The default path:
    local_cpu_small - granite-embedding-97m-multilingual-r2  (384d). Default.
    local_cpu_large - granite-embedding-311m-multilingual-r2 (768d).

  gpu — sentence-transformers (torch), behind the optional `gpu` extra
  (`uv sync --extra gpu`). Qwen3-Embedding, stronger but loads into VRAM:
    local_gpu_small - Qwen3-Embedding-0.6B (1024d).
    local_gpu_large - Qwen3-Embedding-4B (1024d MRL). Strongest practical pick.

model_id is device-independent but model+dim-specific, so it keys the index +
cache: switching model/dim invalidates cleanly; switching CPU<->GPU does not.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache

from engram_mcp import errors
from engram_mcp.embeddings.fastembed_provider import FastEmbedProvider

# Unified profile names: local_<cpu|gpu>_<small|large>.
#   cpu = FastEmbed/ONNX, no torch, ~0 VRAM   |   gpu = torch, VRAM
# Every model is multilingual (incl. Russian) + code, Apache-2.0.

# cpu path — name -> (model_name, device). Granite R2 via FastEmbed custom ONNX.
_FASTEMBED: dict[str, tuple[str, str]] = {
    "local_cpu_small": ("ibm-granite/granite-embedding-97m-multilingual-r2", "cpu"),
    "local_cpu_large": ("ibm-granite/granite-embedding-311m-multilingual-r2", "cpu"),
}

# gpu path — name -> (model_name, device, truncate_dim, query_prompt, passage_prompt).
# Qwen3-Embedding family; MRL truncates the 4B's native 2560d to 1024 for index
# parity at <few % quality loss.
_QWEN: dict[str, tuple[str, str, int | None, str | None, str | None]] = {
    "local_gpu_small": ("Qwen/Qwen3-Embedding-0.6B", "auto", 1024, "query", None),
    "local_gpu_large": ("Qwen/Qwen3-Embedding-4B", "auto", 1024, "query", None),
}

PROFILES = frozenset(_FASTEMBED) | frozenset(_QWEN)
# Index-time default profile. ENGRAM_DEFAULT_INDEX_PROFILE is the explicit knob;
# ENGRAM_PROFILE remains as a backwards-compatible alias. Default is the light
# no-VRAM multilingual model so an always-on server stays cheap across clients.
DEFAULT_PROFILE = (
    os.environ.get("ENGRAM_DEFAULT_INDEX_PROFILE")
    or os.environ.get("ENGRAM_PROFILE")
    or "local_cpu_small"
)

_LOADED_MODEL_IDS: set[str] = set()
_LOADED_LOCK = threading.Lock()


def validate_profile(profile: str) -> None:
    if profile not in PROFILES:
        raise errors.EngramError(
            f"unknown profile {profile!r}; choices: {', '.join(sorted(PROFILES))}",
            errors.E_UNKNOWN_PROFILE,
        )


def default_index_profile() -> str:
    validate_profile(DEFAULT_PROFILE)
    return DEFAULT_PROFILE


def _remember_loaded(provider) -> None:
    with _LOADED_LOCK:
        _LOADED_MODEL_IDS.add(provider.model_id)


def loaded_model_ids() -> list[str]:
    with _LOADED_LOCK:
        return sorted(_LOADED_MODEL_IDS)


def is_model_loaded(model_id: str) -> bool:
    with _LOADED_LOCK:
        return model_id in _LOADED_MODEL_IDS


@lru_cache(maxsize=8)
def _fastembed(model_name: str, device: str) -> FastEmbedProvider:
    from engram_mcp.net import guard_download

    with guard_download(model_name):
        return FastEmbedProvider(model_name, device=device)


@lru_cache(maxsize=8)
def _st(model_name, device, truncate_dim, query_prompt, passage_prompt):
    from engram_mcp.embeddings.sentence_transformers_provider import (
        SentenceTransformersProvider,
    )
    from engram_mcp.net import guard_download

    with guard_download(model_name):
        return SentenceTransformersProvider(
            model_name, device=device, truncate_dim=truncate_dim,
            query_prompt=query_prompt, passage_prompt=passage_prompt,
        )


def make_provider(profile: str | None = None):
    profile = profile or default_index_profile()
    validate_profile(profile)
    if profile in _FASTEMBED:
        provider = _fastembed(*_FASTEMBED[profile])
        _remember_loaded(provider)
        return provider
    if profile in _QWEN:
        provider = _st(*_QWEN[profile])
        _remember_loaded(provider)
        return provider
    raise AssertionError("validated profile missing from provider tables")


def provider_for_model_id(model_id: str, device: str = "auto"):
    """Build the provider matching a manifest's recorded embedder id."""
    if model_id.startswith("fastembed:"):
        model_name = model_id.split(":", 1)[1]
        if not model_name:
            raise errors.EngramError(
                f"unsupported embedder id: {model_id!r}",
                errors.E_UNKNOWN_PROFILE,
            )
        # Custom ONNX models (Granite) are the no-VRAM `local_cpu_*` profiles:
        # pin replay to CPU so it (a) reuses the same cached provider built at
        # index time (device is part of the lru key) and (b) can never quietly
        # run on CUDA and break the advertised ~0-VRAM contract.
        from engram_mcp.embeddings.fastembed_provider import _CUSTOM_ONNX

        dev = "cpu" if model_name in _CUSTOM_ONNX else device
        provider = _fastembed(model_name, dev)
        _remember_loaded(provider)
        return provider
    if model_id.startswith("st:"):
        # st:<model>@<dim>#<query_prompt>/<passage_prompt>
        spec, _, ptag = model_id[len("st:"):].partition("#")
        model, _, dim_tag = spec.rpartition("@")
        if not model:
            raise errors.EngramError(
                f"unsupported embedder id: {model_id!r}",
                errors.E_UNKNOWN_PROFILE,
            )
        try:
            truncate = None if dim_tag in ("full", "") else int(dim_tag)
        except ValueError as exc:
            raise errors.EngramError(
                f"unsupported embedder id: {model_id!r}",
                errors.E_UNKNOWN_PROFILE,
            ) from exc
        qp, _, pp = ptag.partition("/")
        qp = None if qp in ("none", "") else qp
        pp = None if pp in ("none", "") else pp
        provider = _st(model, device, truncate, qp, pp)
        _remember_loaded(provider)
        return provider
    raise errors.EngramError(
        f"unsupported embedder id: {model_id!r}",
        errors.E_UNKNOWN_PROFILE,
        hint="This index was built with an embedder this Engram version does not know how to load.",
    )


def provider_for_project(root, device: str = "auto"):
    """Return the embedder recorded in a project's manifest.

    Kept for compatibility with older callers. It intentionally does not fall
    back to the default profile; read paths must be hermetic to the index.
    """
    from engram_mcp import manifest, paths

    pdir = paths.project_dir(root, create=False)
    m = manifest.load_project(pdir) if pdir.exists() else None
    if m is None or not m.embedder_id:
        raise errors.EngramError(
            f"project not indexed: {root}",
            errors.E_PROJECT_NOT_INDEXED,
        )
    return provider_for_model_id(m.embedder_id, device)
