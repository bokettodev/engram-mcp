"""Optional local cross-encoder rerankers.

Default backend is ``fastembed``: a torch-free ONNX cross-encoder that runs on
CPU via onnxruntime, so reranking works on the always-on, ~0-VRAM server without
the ``gpu`` extra. The default model is the multilingual
``jinaai/jina-reranker-v2-base-multilingual`` (RU+EN, to match the multilingual
Granite embedder).

License note: jina-reranker-v2-base-multilingual is CC-BY-NC-4.0. That is fine
for local/private use (engram's target), but is NON-COMMERCIAL — revisit the
model choice before shipping engram in a commercial/redistributed product. The
Apache-2.0 alternatives in FastEmbed's ONNX catalog (ms-marco-MiniLM,
bge-reranker-base) are English/zh-centric and regress Russian queries.

The ``sentence_transformers`` backend (multilingual BAAI/bge-reranker-v2-m3,
Apache-2.0) remains available only to explicit offline callers. The server
search path always requests the FastEmbed ONNX backend.
"""

from __future__ import annotations

import os
from functools import lru_cache

DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"
DEFAULT_ONNX_RERANKER = "jinaai/jina-reranker-v2-base-multilingual"
DEFAULT_BACKEND = "fastembed"


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


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER, device: str = "auto"):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on extra
            raise ImportError(
                "reranking needs the optional 'gpu' extra: run `uv sync --extra gpu`"
            ) from exc
        from engram_mcp.net import guard_download

        self.model_id = model_name
        self.backend = "sentence_transformers"
        with guard_download(model_name):
            self._ce = CrossEncoder(model_name, device=_resolve_device(device))

    def rerank(self, query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
        if not candidates:
            return candidates
        pairs = [[query, (c.get("content") or "")] for c in candidates]
        scores = self._ce.predict(pairs)
        ordered = sorted(zip(candidates, scores), key=lambda cs: float(cs[1]), reverse=True)
        ranked = []
        for c, s in ordered:
            item = dict(c)
            item["score"] = float(s)
            item["reranked"] = True
            ranked.append(item)
        return ranked[:top_k] if top_k else ranked


class FastEmbedOnnxReranker:
    """Torch-free ONNX cross-encoder reranker (CPU, via onnxruntime).

    Uses ``fastembed.rerank.cross_encoder.TextCrossEncoder`` (present in the
    pinned FastEmbed, just not re-exported at the top level). Runs on CPU so the
    server stays torch-free and ~0 VRAM; no ``gpu`` extra required.
    """

    def __init__(self, model_name: str | None = None):
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:  # pragma: no cover - fastembed is a core dep
            raise ImportError(
                "FastEmbed does not expose fastembed.rerank.cross_encoder.TextCrossEncoder; "
                "upgrade FastEmbed."
            ) from exc
        from engram_mcp.net import guard_download

        self.model_id = model_name or DEFAULT_ONNX_RERANKER
        self.backend = "fastembed"
        with guard_download(self.model_id):
            # cuda=False keeps this on the CPU onnxruntime provider (no VRAM, no torch).
            self._model = TextCrossEncoder(self.model_id, cuda=False)

    def rerank(self, query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
        if not candidates:
            return candidates
        docs = [(c.get("content") or "") for c in candidates]
        scores = list(self._model.rerank(query, docs))
        ordered = sorted(zip(candidates, scores), key=lambda cs: float(cs[1]), reverse=True)
        ranked = []
        for c, s in ordered:
            item = dict(c)
            item["score"] = float(s)
            item["reranked"] = True
            ranked.append(item)
        return ranked[:top_k] if top_k else ranked


def _backend(backend: str | None) -> str:
    value = (backend or DEFAULT_BACKEND).strip().lower()
    if value in {"st", "crossencoder", "cross_encoder"}:
        return "sentence_transformers"
    if value in {"fastembed", "onnx"}:
        return "fastembed"
    if value != "sentence_transformers":
        raise ImportError("reranker backend must be 'sentence_transformers' or 'fastembed'")
    return value


@lru_cache(maxsize=4)
def get_reranker(
    model_name: str | None = None,
    device: str = "auto",
    backend: str | None = None,
):
    selected = _backend(backend)
    if selected == "fastembed":
        onnx_model = model_name or os.environ.get("ENGRAM_RERANKER_MODEL") or DEFAULT_ONNX_RERANKER
        return FastEmbedOnnxReranker(onnx_model)
    return CrossEncoderReranker(model_name or DEFAULT_RERANKER, device=device)
