"""Embedding provider factory for the single supported Granite embedder.

Engram stores one canonical embedder id in manifests and cache keys:

    fastembed:ibm-granite/granite-embedding-97m-multilingual-r2

Search always replays that id with FastEmbed/ONNX on CPU. Indexing uses the
same FastEmbed CPU backend by default, or a one-shot sentence-transformers CUDA
backend when explicitly requested. The CUDA backend reports the canonical
``model_id`` for compatibility/cache semantics and a distinct ``backend_id`` for
loaded-model accounting.
"""

from __future__ import annotations

import importlib.util
import os
import threading
from functools import lru_cache
from typing import Any, Callable

from engram_mcp import errors, paths
from engram_mcp.embeddings.fastembed_provider import FastEmbedProvider

GRANITE_MODEL = "ibm-granite/granite-embedding-97m-multilingual-r2"
CANONICAL_EMBEDDER_ID = f"fastembed:{GRANITE_MODEL}"
# Index device *setting*: "auto" prefers a CUDA GPU and only falls back to CPU
# when none is usable. GPU is the priority path; CPU indexing is much slower and
# meant as a fallback. "cpu"/"cuda" force the device explicitly.
SUPPORTED_INDEX_DEVICES = ("auto", "cpu", "cuda")
DEFAULT_INDEX_DEVICE = "auto"

_REMOVED_EMBEDDER_HINT = (
    "This index was built with a removed/unsupported embedder; rebuild it "
    "with `engram index --rebuild <path>`."
)

_LOADED_BACKEND_IDS: set[str] = set()
_LOADED_LOCK = threading.Lock()
ProgressSink = Callable[[dict[str, Any]], None]


def _backend_id(provider) -> str:
    return str(getattr(provider, "backend_id", provider.model_id))


def _remember_loaded(provider) -> None:
    with _LOADED_LOCK:
        _LOADED_BACKEND_IDS.add(_backend_id(provider))


def _forget_loaded(provider) -> None:
    with _LOADED_LOCK:
        _LOADED_BACKEND_IDS.discard(_backend_id(provider))


def loaded_model_ids() -> list[str]:
    """Loaded backend ids, not canonical manifest ids."""

    with _LOADED_LOCK:
        return sorted(_LOADED_BACKEND_IDS)


def is_model_loaded(backend_id: str) -> bool:
    with _LOADED_LOCK:
        return backend_id in _LOADED_BACKEND_IDS


def _unsupported_embedder(model_id: str) -> errors.EngramError:
    return errors.EngramError(
        f"unsupported embedder id: {model_id!r}",
        errors.E_UNKNOWN_PROFILE,
        hint=_REMOVED_EMBEDDER_HINT,
    )


def query_backend_id_for_model_id(model_id: str) -> str:
    """Return the search backend id for a canonical manifest id without loading."""

    if model_id == CANONICAL_EMBEDDER_ID:
        return CANONICAL_EMBEDDER_ID
    raise _unsupported_embedder(model_id)


def _cuda_available() -> bool:
    """Whether a usable CUDA GPU is present (probes torch; import failure = no)."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_index_device(
    index_device: str | None = None, *, gpu: bool = False, cpu: bool = False
) -> str:
    """The requested index-device *setting* (auto|cpu|cuda). No hardware probe —
    "auto" is returned as-is so callers that must stay torch-free (the server)
    can route without importing torch. Use ``effective_index_device`` to resolve
    "auto" to a concrete device."""

    if gpu and cpu:
        raise errors.EngramError(
            "pass only one of gpu/cpu", errors.E_BAD_REQUEST,
            hint="Use `--gpu` or `--cpu`, not both.",
        )
    if gpu:
        return "cuda"
    if cpu:
        return "cpu"
    raw = (index_device or os.environ.get("ENGRAM_INDEX_DEVICE") or DEFAULT_INDEX_DEVICE)
    device = raw.strip().lower()
    if device not in SUPPORTED_INDEX_DEVICES:
        raise errors.EngramError(
            f"unsupported index device {raw!r}",
            errors.E_BAD_REQUEST,
            hint="Use `auto`, `cpu`, or `cuda` (or unset ENGRAM_INDEX_DEVICE).",
        )
    return device


def effective_index_device(setting: str) -> str:
    """Resolve an "auto" setting to a concrete device: prefer a CUDA GPU, fall
    back to CPU only when none is usable. "cpu"/"cuda" pass through unchanged."""
    if setting == "auto":
        return "cuda" if _cuda_available() else "cpu"
    return setting


def default_index_device() -> str:
    return resolve_index_device(None)


def _emit_progress(progress: ProgressSink | None, stage: str, **fields) -> None:
    if progress is None:
        return
    event = {"stage": stage, **fields}
    try:
        progress(event)
    except TypeError:
        # Factory progress is event-only. Ignore old embedding-only callbacks.
        return


def acquire_gpu_index_lock(progress: ProgressSink | None = None, timeout: float = -1):
    """Acquire the machine-wide CUDA indexing lock.

    The lock is blocking by default. Passing a short timeout is useful in tests
    and diagnostics.
    """

    _emit_progress(
        progress,
        "waiting_for_gpu",
        unit="lock",
        done=0,
        total=None,
    )
    lock = paths.gpu_index_lock()
    lock.acquire(timeout=timeout)
    return lock


def release_gpu_index_lock(lock) -> None:
    if lock is not None:
        lock.release()


@lru_cache(maxsize=1)
def _fastembed_granite_cpu() -> FastEmbedProvider:
    from engram_mcp.net import guard_download

    with guard_download(GRANITE_MODEL):
        return FastEmbedProvider(GRANITE_MODEL, device="cpu")


def _sentence_transformers_granite_cuda():
    if importlib.util.find_spec("sentence_transformers") is None:
        raise errors.EngramError(
            "CUDA indexing needs the optional 'gpu' extra.",
            errors.E_EXTRA_MISSING,
            hint="Run `uv sync --extra gpu` or `uv run --extra gpu engram index --gpu <path>`.",
        )

    from engram_mcp.embeddings.sentence_transformers_provider import (
        SentenceTransformersProvider,
    )
    from engram_mcp.net import guard_download

    with guard_download(GRANITE_MODEL):
        return SentenceTransformersProvider(
            GRANITE_MODEL,
            device="cuda",
            truncate_dim=None,
            query_prompt=None,
            passage_prompt=None,
            canonical_id=CANONICAL_EMBEDDER_ID,
            strict_device=True,
        )


def provider_for_model_id(model_id: str):
    """Build the CPU query provider for a manifest's canonical embedder id."""

    if model_id != CANONICAL_EMBEDDER_ID:
        raise _unsupported_embedder(model_id)
    provider = _fastembed_granite_cpu()
    _remember_loaded(provider)
    return provider


def make_index_provider(index_device: str | None = None, *, progress: ProgressSink | None = None):
    """Build an index-time provider.

    CPU providers are cached. CUDA providers are intentionally not cached so a
    GPU index job can unload the model and return VRAM after the job. An "auto"
    setting resolves here (prefers GPU) — this is a torch-import point, so the
    long-lived server keeps GPU/auto jobs in a subprocess and only ever asks for
    an explicit "cpu" provider in-process.
    """

    device = effective_index_device(resolve_index_device(index_device))
    gpu_lock = None
    if device == "cpu":
        provider = _fastembed_granite_cpu()
    elif device == "cuda":
        gpu_lock = acquire_gpu_index_lock(progress)
        try:
            provider = _sentence_transformers_granite_cuda()
        except Exception:
            release_gpu_index_lock(gpu_lock)
            raise
        setattr(provider, "_engram_gpu_index_lock", gpu_lock)
    else:  # pragma: no cover - resolve_index_device validates the value
        raise AssertionError(f"unhandled index device: {device}")
    _remember_loaded(provider)
    return provider


def make_provider(index_device: str | None = None):
    """Compatibility alias for older internal callers."""

    return make_index_provider(index_device)


def release_index_provider(provider) -> None:
    """Release index backend resources after a job.

    FastEmbed CPU is cached and retained. The one-shot CUDA provider is unloaded
    so VRAM use is limited to the index job.
    """

    if provider is None:
        return
    try:
        provider.release_unused_cache()
    finally:
        gpu_lock = getattr(provider, "_engram_gpu_index_lock", None)
        try:
            if getattr(provider, "device", None) == "cuda" and hasattr(provider, "unload"):
                try:
                    provider.unload()
                finally:
                    _forget_loaded(provider)
        finally:
            if gpu_lock is not None:
                release_gpu_index_lock(gpu_lock)
                try:
                    setattr(provider, "_engram_gpu_index_lock", None)
                except Exception:
                    pass


def provider_for_project(root):
    """Return the CPU query provider recorded in a project's manifest."""

    from engram_mcp import manifest, paths

    pdir = paths.project_dir(root, create=False)
    m = manifest.load_project(pdir) if pdir.exists() else None
    if m is None or not m.embedder_id:
        raise errors.EngramError(
            f"project not indexed: {root}",
            errors.E_PROJECT_NOT_INDEXED,
        )
    return provider_for_model_id(m.embedder_id)
