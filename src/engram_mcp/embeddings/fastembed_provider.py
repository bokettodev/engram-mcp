"""Local embedding provider backed by FastEmbed (ONNX).

Engram's query path uses Granite 97m through FastEmbed on CPU. The provider's
``model_id`` is the canonical manifest/cache id. ``backend_id`` is the loaded
runtime id used for process-local diagnostics.
"""

from __future__ import annotations

import builtins
import contextlib
import logging
import sys
import threading
from typing import Sequence

from engram_mcp import config

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _utf8_text_open():
    """Default encoding-less text ``open()`` to UTF-8 for the duration of the
    block, on Windows only.

    FastEmbed may load model assets with the process default encoding, which is
    cp1252 on Windows and raises ``UnicodeDecodeError`` on UTF-8 content. UTF-8 mode
    can't be toggled after interpreter start and re-exec is unsafe for the stdio
    server, so we scope a minimal ``open`` shim to the model-load call. Binary
    opens and explicit-encoding opens are untouched."""
    if sys.platform != "win32" or sys.flags.utf8_mode:
        yield
        return
    orig = builtins.open

    def _patched(file, *args, **kwargs):
        mode = kwargs.get("mode", args[0] if args else "r")
        # inject only when the caller passed neither a binary mode nor an
        # encoding (positionally: file, mode, buffering, encoding -> len<3).
        if "b" not in mode and "encoding" not in kwargs and len(args) < 3:
            kwargs["encoding"] = "utf-8"
        return orig(file, *args, **kwargs)

    builtins.open = _patched
    try:
        yield
    finally:
        builtins.open = orig


# ONNX models not in FastEmbed's default catalog, registered on first use.
# Granite R2 is multilingual (100+ langs incl. Russian) + code, Apache-2.0, CLS
# pooling. The fastembed id is the canonical search/compatibility id even when
# an index was produced by the optional sentence-transformers CUDA backend.
_CUSTOM_ONNX: dict[str, dict] = {
    "ibm-granite/granite-embedding-97m-multilingual-r2": {"pooling": "CLS", "dim": 384},
}

# Exact upstream revision each custom-registered model is pinned to (see
# config.py for why). FastEmbed's own download path doesn't accept a
# revision, so this is enforced by resolving the snapshot ourselves (see
# ``hf_pin``) and handing FastEmbed the local directory via
# ``specific_model_path`` -- which bypasses its (revision-less) download
# entirely, so this really does pin what gets loaded.
_PINNED_REVISIONS: dict[str, str] = {
    config.DEFAULT_EMBED_MODEL: config.EMBED_MODEL_REVISION,
}

_REGISTER_LOCK = threading.Lock()


def _ensure_custom_registered(model_name: str) -> None:
    """Register a known custom ONNX model with FastEmbed (idempotent, no download)."""
    spec = _CUSTOM_ONNX.get(model_name)
    if spec is None:
        return
    from fastembed import TextEmbedding

    # check-and-add under a lock: lru_cache can compute concurrent first-use
    # misses, and add_custom_model rejects duplicates.
    with _REGISTER_LOCK:
        if any(m["model"] == model_name for m in TextEmbedding.list_supported_models()):
            return
        from fastembed.common.model_description import ModelSource, PoolingType

        try:
            TextEmbedding.add_custom_model(
                model=model_name,
                pooling=PoolingType[spec["pooling"]],
                normalization=True,
                sources=ModelSource(hf=model_name),
                dim=spec["dim"],
            )
        except ValueError:
            pass  # already registered by a racing caller


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            return "cuda"
    except Exception:  # pragma: no cover - onnxruntime always present here
        pass
    return "cpu"


def _resolve_pinned_model_path(model_name: str) -> tuple[str, str]:
    """Resolve the pinned local snapshot dir + artifact digest for a model.

    Returns ``("", "")`` for a model with no recorded pin (loaded unpinned,
    at whatever FastEmbed's own download resolves to). A pinned model that
    fails to resolve raises -- pinning exists specifically so a revision
    mismatch is never silently ignored, so a resolution failure (e.g. no
    network and nothing cached yet) must fail loud rather than quietly fall
    back to an unpinned load.
    """
    revision = _PINNED_REVISIONS.get(model_name)
    if revision is None:
        return "", ""
    from engram_mcp.embeddings import hf_pin

    snapshot_dir = hf_pin.pinned_snapshot_dir(
        model_name, revision, extra_patterns=("onnx/model.onnx",)
    )
    digest = hf_pin.blob_digest(snapshot_dir, "onnx/model.onnx")
    return snapshot_dir, digest


class FastEmbedProvider:
    def __init__(self, model_name: str = config.DEFAULT_EMBED_MODEL, device: str = "cpu"):
        from fastembed import TextEmbedding

        _ensure_custom_registered(model_name)
        self.model_name = model_name
        snapshot_dir, self.artifact_digest = _resolve_pinned_model_path(model_name)
        pin_kwargs = {"specific_model_path": snapshot_dir} if snapshot_dir else {}
        # model_id is the device-independent cache/manifest key: repo + pinned
        # revision + dim + pooling (see hf_pin.canonical_model_id). Falls back
        # to the plain repo-name id for a hypothetical model with no recorded
        # pin/spec -- in practice only the pinned Granite model is ever loaded
        # through this provider.
        pinned_revision = _PINNED_REVISIONS.get(model_name, "")
        spec = _CUSTOM_ONNX.get(model_name)
        if pinned_revision and spec is not None:
            from engram_mcp.embeddings import hf_pin

            self.model_id = hf_pin.canonical_model_id(
                backend="fastembed",
                repo_id=model_name,
                revision=pinned_revision,
                dim=spec["dim"],
                pooling=spec["pooling"],
            )
        else:
            self.model_id = f"fastembed:{model_name}"  # device-independent cache key
        self.backend_id = self.model_id
        self.device = _resolve_device(device)

        kwargs = {"cuda": True} if self.device == "cuda" else {}
        kwargs.update(pin_kwargs)
        try:
            with _utf8_text_open():
                self._model = TextEmbedding(model_name, **kwargs)
        except Exception as exc:
            if self.device == "cuda":
                logger.warning("CUDA embedder unavailable (%r); falling back to CPU", exc)
                self.device = "cpu"
                with _utf8_text_open():
                    self._model = TextEmbedding(model_name, **pin_kwargs)
            else:
                raise

        # Use the model's passage encoder when available (asymmetric models
        # apply a passage-specific prefix); falls back to plain embed().
        self._passage = getattr(self._model, "passage_embed", self._model.embed)
        # One model instance is shared across the index worker and search
        # threads; serialize inference to avoid relying on wrapper thread-safety.
        self._lock = threading.Lock()
        # Probe the dimension once so the store schema is always correct.
        probe = next(iter(self._model.embed(["dimension probe"])))
        self.dim = int(len(probe))

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._lock:
            return [v.tolist() for v in self._passage(list(texts))]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._lock:
            return [v.tolist() for v in self._model.query_embed(list(texts))]

    def release_unused_cache(self) -> None:
        """No-op: FastEmbed/ONNX doesn't hold a PyTorch CUDA allocator cache."""
        return
