"""Local embedding provider backed by FastEmbed (ONNX).

Default model: granite-embedding-97m-multilingual-r2 (384-dim). Device is "cpu" by default;
"cuda" uses the ONNX CUDA execution provider (requires the `fastembed-gpu`
package + a working CUDA runtime) and falls back to CPU if unavailable. "auto"
picks CUDA when an ONNX CUDA provider is present.

The model id intentionally excludes the device: CPU and GPU produce the same
vectors for the same model, so switching device must NOT invalidate the cache.
"""

from __future__ import annotations

import logging
import threading
from typing import Sequence

from engram_mcp import config

logger = logging.getLogger(__name__)


# ONNX models not in FastEmbed's default catalog, registered on first use.
# Granite R2 is multilingual (100+ langs incl. Russian) + code, Apache-2.0, CLS
# pooling — a no-torch, ~0-VRAM alternative to the Qwen torch profiles, which
# matters when many MCP clients each run their own server process.
_CUSTOM_ONNX: dict[str, dict] = {
    "ibm-granite/granite-embedding-97m-multilingual-r2": {"pooling": "CLS", "dim": 384},
    "ibm-granite/granite-embedding-311m-multilingual-r2": {"pooling": "CLS", "dim": 768},
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


class FastEmbedProvider:
    def __init__(self, model_name: str = config.DEFAULT_EMBED_MODEL, device: str = "cpu"):
        from fastembed import TextEmbedding

        _ensure_custom_registered(model_name)
        self.model_name = model_name
        self.model_id = f"fastembed:{model_name}"  # device-independent cache key
        self.device = _resolve_device(device)

        kwargs = {"cuda": True} if self.device == "cuda" else {}
        try:
            self._model = TextEmbedding(model_name, **kwargs)
        except Exception as exc:
            if self.device == "cuda":
                logger.warning("CUDA embedder unavailable (%r); falling back to CPU", exc)
                self.device = "cpu"
                self._model = TextEmbedding(model_name)
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
