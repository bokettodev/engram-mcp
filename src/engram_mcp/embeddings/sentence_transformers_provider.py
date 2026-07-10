"""Index-time embedder via sentence-transformers (PyTorch).

The optional CUDA index path uses the same Granite 97m vector space as the
FastEmbed CPU search path. When ``canonical_id`` is set, ``model_id`` is that
canonical manifest/cache id and ``backend_id`` records the actual runtime.

``revision`` pins the exact upstream commit sentence-transformers loads
(unlike FastEmbed, ``SentenceTransformer`` accepts ``revision=`` natively --
no snapshot-resolution workaround needed here, see ``embeddings/hf_pin.py``).

``artifact_digest`` is intentionally left empty for this backend: the
manifest's recorded digest must be comparable against whatever the *query*
path (always FastEmbed/ONNX CPU) computes, and this backend loads a
different artifact (safetensors, not the ONNX export) -- hashing it would
only produce a digest that can never match the query-time one, which would
make every GPU-built index fail its digest check the first time it's
queried. The revision pin (part of ``model_id``/``canonical_id``) is what
actually protects a GPU-built index; the artifact digest is CPU-backend-only
defense in depth on top of that.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
from typing import Sequence

from engram_mcp import errors

logger = logging.getLogger(__name__)

# Forward-pass batch size for SentenceTransformer.encode. This is the real cap on
# activation VRAM during indexing (NOT config.EMBED_BATCH_SIZE, which only sets
# the outer slice handed to the provider). Override per-host with
# ENGRAM_ST_BATCH_SIZE. Not part of model_id; it must not invalidate the index.
_DEFAULT_ST_BATCH = 16


def _env_batch_size() -> int:
    raw = os.environ.get("ENGRAM_ST_BATCH_SIZE", "").strip()
    if raw:
        try:
            n = int(raw)
            if n > 0:
                return n
        except ValueError:
            logger.warning("invalid ENGRAM_ST_BATCH_SIZE=%r; using default %d", raw, _DEFAULT_ST_BATCH)
    return _DEFAULT_ST_BATCH


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover - torch optional
        pass
    return "cpu"


class SentenceTransformersProvider:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        truncate_dim: int | None = None,
        query_prompt: str | None = None,
        passage_prompt: str | None = None,
        canonical_id: str | None = None,
        strict_device: bool = False,
        revision: str | None = None,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on extra
            raise errors.EngramError(
                "sentence-transformers is required for CUDA indexing.",
                errors.E_EXTRA_MISSING,
                hint="Run `uv sync --extra gpu` or `uv run --extra gpu engram index --gpu <path>`.",
            ) from exc

        self.model_name = model_name
        self.device = _resolve_device(device)
        self._query_prompt = query_prompt
        self._passage_prompt = passage_prompt
        self._model = None

        if strict_device and self.device == "cuda":
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - sentence-transformers depends on torch
                raise errors.EngramError(
                    "torch is required for CUDA indexing.",
                    errors.E_EXTRA_MISSING,
                    hint="Run `uv sync --extra gpu` or `uv run --extra gpu engram index --gpu <path>`.",
                ) from exc
            if not torch.cuda.is_available():
                raise errors.EngramError(
                    "CUDA indexing was requested, but torch reports CUDA is unavailable.",
                    errors.E_MODEL_LOAD_FAILED,
                    hint="Omit `--gpu` or unset ENGRAM_INDEX_DEVICE=cuda to index on CPU.",
                )

        try:
            self._model = SentenceTransformer(
                model_name, device=self.device, truncate_dim=truncate_dim, revision=revision
            )
        except Exception as exc:
            if strict_device:
                raise errors.EngramError(
                    f"failed to load sentence-transformers backend on {self.device}",
                    errors.E_MODEL_LOAD_FAILED,
                    hint=str(exc),
                ) from exc
            if self.device == "cuda":
                logger.warning("CUDA load failed (%r); falling back to CPU", exc)
                self.device = "cpu"
                self._model = SentenceTransformer(
                    model_name, device="cpu", truncate_dim=truncate_dim, revision=revision
                )
            else:
                raise

        # Prompt mode changes the output vectors, so it is part of the backend
        # id. Device is also included because loaded-model tracking is per
        # runtime, not per canonical vector space.
        dim_tag = truncate_dim if truncate_dim else "full"
        ptag = f"{query_prompt or 'none'}/{passage_prompt or 'none'}"
        rev_tag = f":rev={revision}" if revision else ""
        self.backend_id = f"st:{model_name}@{dim_tag}#{ptag}:{self.device}{rev_tag}"
        self.model_id = canonical_id or f"st:{model_name}@{dim_tag}#{ptag}{rev_tag}"
        # See module docstring: never computed for this backend, only for the
        # canonical FastEmbed/ONNX one that the query path actually uses.
        self.artifact_digest = ""
        self._lock = threading.Lock()
        self._batch_size = _env_batch_size()
        try:
            probe = self._encode(["dimension probe"], self._passage_prompt)
        except Exception:
            self.unload()
            raise
        self.dim = int(len(probe[0]))

    def _encode(self, texts: list[str], prompt_name: str | None) -> list[list[float]]:
        if self._model is None:
            raise RuntimeError("sentence-transformers provider has been unloaded")
        kwargs = {
            "normalize_embeddings": True, "convert_to_numpy": True,
            "show_progress_bar": False, "batch_size": self._batch_size,
        }
        if prompt_name:
            try:
                vecs = self._model.encode(texts, prompt_name=prompt_name, **kwargs)
                return vecs.tolist()
            except (ValueError, KeyError) as exc:
                logger.warning("prompt_name %r unsupported (%s); encoding without it", prompt_name, exc)
        return self._model.encode(texts, **kwargs).tolist()

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._lock:
            return self._encode(list(texts), self._passage_prompt)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._lock:
            return self._encode(list(texts), self._query_prompt)

    def release_unused_cache(self) -> None:
        """Free unused CUDA allocator blocks. No-op on CPU."""

        if self.device != "cuda":
            return
        try:
            import torch
        except Exception:  # pragma: no cover - torch present on the cuda path
            return
        with self._lock:
            torch.cuda.empty_cache()

    def unload(self) -> None:
        """Drop the model and return CUDA memory after a one-shot index job."""

        with self._lock:
            self._model = None
            gc.collect()
            if self.device == "cuda":
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:  # pragma: no cover - cleanup best-effort
                    pass
