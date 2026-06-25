"""Code-specialized embedder via sentence-transformers (PyTorch, CUDA fp16).

This is the GPU/quality path (jina-code, Qwen3-Embedding, ...). It pulls torch,
so it lives behind the optional ``code`` extra:

    uv sync --extra code        # installs sentence-transformers + torch

The model_id includes the model name + output dim so swapping model/dim
invalidates the index + cache cleanly. CPU and GPU produce the same vectors, so
device is intentionally NOT part of the id.
"""

from __future__ import annotations

import logging
import threading
from typing import Sequence

logger = logging.getLogger(__name__)


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
                "the code-embedder profiles need the optional 'code' extra: "
                "run `uv sync --extra code` (installs sentence-transformers + torch)"
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
        probe = self._encode(["dimension probe"], self._passage_prompt)
        self.dim = int(len(probe[0]))

    def _encode(self, texts: list[str], prompt_name: str | None) -> list[list[float]]:
        kwargs = {"normalize_embeddings": True, "convert_to_numpy": True, "show_progress_bar": False}
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
