"""Quality embedder via sentence-transformers (PyTorch, CUDA fp16).

This is the GPU/quality path (Qwen3-Embedding, ...). It pulls torch, so it lives
behind the optional ``gpu`` extra:

    uv sync --extra gpu         # installs sentence-transformers + torch

The model_id includes the model name + output dim so swapping model/dim
invalidates the index + cache cleanly. CPU and GPU produce the same vectors, so
device is intentionally NOT part of the id.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Sequence

logger = logging.getLogger(__name__)

# Forward-pass batch size for SentenceTransformer.encode. This is the real cap on
# activation VRAM during indexing (NOT config.EMBED_BATCH_SIZE, which only sets
# the outer slice handed to the provider). Kept small by default because the
# quality models are large and the GPU is often shared; override per-host with
# ENGRAM_ST_BATCH_SIZE. Not part of model_id — it must not invalidate the index.
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
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on extra
            raise ImportError(
                "the Qwen3 quality profiles need the optional 'gpu' extra: "
                "run `uv sync --extra gpu` (installs sentence-transformers + torch)"
            ) from exc

        self.model_name = model_name
        self.device = _resolve_device(device)
        self._query_prompt = query_prompt
        self._passage_prompt = passage_prompt
        try:
            self._model = SentenceTransformer(
                model_name, device=self.device, trust_remote_code=True, truncate_dim=truncate_dim
            )
        except Exception as exc:
            if self.device == "cuda":
                logger.warning("CUDA load failed (%r); falling back to CPU", exc)
                self.device = "cpu"
                self._model = SentenceTransformer(
                    model_name, device="cpu", trust_remote_code=True, truncate_dim=truncate_dim
                )
            else:
                raise
        # The prompt mode changes the output vectors, so it is part of the id
        # (so the index/cache invalidate cleanly when prompts change).
        dim_tag = truncate_dim if truncate_dim else "full"
        ptag = f"{query_prompt or 'none'}/{passage_prompt or 'none'}"
        self.model_id = f"st:{model_name}@{dim_tag}#{ptag}"
        self._lock = threading.Lock()
        self._batch_size = _env_batch_size()
        probe = self._encode(["dimension probe"], self._passage_prompt)
        self.dim = int(len(probe[0]))

    def _encode(self, texts: list[str], prompt_name: str | None) -> list[list[float]]:
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
        """Free the CUDA caching allocator's unused blocks (the activation
        high-water left after a bulk index) back to the device. Keeps the model
        resident. No-op on CPU. Held under the encode lock so it never races a
        concurrent forward pass."""
        if self.device != "cuda":
            return
        try:
            import torch
        except Exception:  # pragma: no cover - torch present on the cuda path
            return
        with self._lock:
            reserved_before = torch.cuda.memory_reserved()
            torch.cuda.empty_cache()
            logger.debug(
                "released CUDA cache: reserved %.0f -> %.0f MiB (allocated %.0f MiB, model resident)",
                reserved_before / 1048576,
                torch.cuda.memory_reserved() / 1048576,
                torch.cuda.memory_allocated() / 1048576,
            )
