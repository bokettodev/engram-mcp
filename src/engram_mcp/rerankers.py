"""Optional local cross-encoder reranker.

The only backend is ``fastembed``: a torch-free ONNX cross-encoder that runs
on CPU via onnxruntime, so reranking works on the always-on, ~0-VRAM server
without the ``gpu`` extra. The default model is the multilingual
``jinaai/jina-reranker-v2-base-multilingual`` (RU+EN, to match the
multilingual Granite embedder).

License note: jina-reranker-v2-base-multilingual is CC-BY-NC-4.0. That is fine
for local/private use (engram's target), but is NON-COMMERCIAL — revisit the
model choice before shipping engram in a commercial/redistributed product. The
Apache-2.0 alternatives in FastEmbed's ONNX catalog (ms-marco-MiniLM,
bge-reranker-base) are English/zh-centric and regress Russian queries.
"""

from __future__ import annotations

import os
from functools import lru_cache

from engram_mcp import config

DEFAULT_ONNX_RERANKER = "jinaai/jina-reranker-v2-base-multilingual"
DEFAULT_BACKEND = "fastembed"

# Exact upstream revision the default reranker is pinned to (see config.py for
# why a bare repo name is not a safe identity). Only the *default* model id is
# pinned -- an operator-supplied ``ENGRAM_RERANKER_MODEL`` override is loaded
# unpinned, at whatever that repo's current tip is.
_PINNED_REVISIONS = {
    DEFAULT_ONNX_RERANKER: config.RERANKER_ONNX_MODEL_REVISION,
}


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
        # Same limitation as the embedder (see embeddings/fastembed_provider.py):
        # FastEmbed's own download path doesn't accept a revision, so a pin is
        # only enforced for the default model id, via a pre-resolved snapshot
        # dir handed in through `specific_model_path`.
        revision = _PINNED_REVISIONS.get(self.model_id)
        pin_kwargs = {}
        if revision:
            from engram_mcp.embeddings import hf_pin

            pin_kwargs["specific_model_path"] = hf_pin.pinned_snapshot_dir(
                self.model_id, revision, extra_patterns=("onnx/model.onnx",)
            )
        with guard_download(self.model_id):
            # cuda=False keeps this on the CPU onnxruntime provider (no VRAM, no torch).
            self._model = TextCrossEncoder(self.model_id, cuda=False, **pin_kwargs)

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
    if value in {"fastembed", "onnx"}:
        return "fastembed"
    raise ImportError("reranker backend must be 'fastembed'")


@lru_cache(maxsize=4)
def get_reranker(
    model_name: str | None = None,
    backend: str | None = None,
):
    _backend(backend)  # validates; only 'fastembed' is supported
    onnx_model = model_name or os.environ.get("ENGRAM_RERANKER_MODEL") or DEFAULT_ONNX_RERANKER
    return FastEmbedOnnxReranker(onnx_model)
