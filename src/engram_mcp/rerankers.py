"""Optional local cross-encoder reranker (behind the `gpu` extra).

Reranks the top candidates from vector/hybrid search by scoring each
(query, code) pair with a cross-encoder — the second-biggest quality lever
after the embedder, per the research, and cheap for a single-user MCP.

Default: BAAI/bge-reranker-v2-m3 (Apache-2.0), a standard sentence-transformers
CrossEncoder. (Qwen3-Reranker is stronger but needs custom yes/no-logit
handling, so it is not the default here.)
"""

from __future__ import annotations

from functools import lru_cache

DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


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
            c = dict(c)
            c["score"] = float(s)
            c["reranked"] = True
            ranked.append(c)
        return ranked[:top_k] if top_k else ranked


@lru_cache(maxsize=2)
def get_reranker(model_name: str = DEFAULT_RERANKER, device: str = "auto") -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name, device=device)
