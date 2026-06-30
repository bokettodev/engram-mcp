"""Hybrid retrieval: dense vector + full-text (BM25) fused with RRF, then
deterministic symbol/path boosts. Rank-based fusion avoids mixing score scales.
"""

from __future__ import annotations

import re
from math import exp

_TOKEN = re.compile(r"[A-Za-z0-9_]+")

# Surface signals that a query is targeting an exact token (identifier, literal,
# path) rather than a concept — those benefit from the BM25 half of hybrid.
_IDENT_SIGNALS = (
    re.compile(r"[a-z][A-Z]"),  # camelCase
    re.compile(r"[A-Za-z]_[A-Za-z]"),  # snake_case / ALL_CAPS_WORD
    re.compile(r"\b[A-Z][A-Z0-9]{2,}\b"),  # ALLCAPS token
    re.compile(r"""['"][^'"]+['"]"""),  # 'quoted literal'
    re.compile(r"\.(py|js|ts|tsx|go|rs|java|c|cpp|h|rb|cs|json|toml|md)\b"),  # file ext
    re.compile(r"(?<![\w])_[A-Za-z]\w+"),  # _leading_underscore identifier
)

RELEVANCE_ORDER = {"uncertain": 0, "low": 1, "medium": 2, "high": 3}
HIGH_RELEVANCE = 0.72
MEDIUM_RELEVANCE = 0.50
LOW_RELEVANCE = 0.25

_ROLE_SCORE_BOOST = {
    "executable": 0.035,
    "test": 0.010,
    "config": -0.020,
    "template": -0.025,
    "comment": -0.035,
}


def _tokens(s: str | None) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(s or "")]


def classify_query(query: str) -> str:
    """Pick a search mode for `mode=auto`: identifier/literal-ish queries get
    `hybrid` (BM25 catches exact tokens), pure natural language gets `vector`."""
    if len(query.split()) == 1:  # a bare single-token query is almost always an identifier
        return "hybrid"
    return "hybrid" if any(p.search(query) for p in _IDENT_SIGNALS) else "vector"


def _rrf(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def _role_boost(role: str | None) -> float:
    return _ROLE_SCORE_BOOST.get(role or "", 0.0)


def hybrid_search(
    store, query, qvector, k=8, where=None, candidate_k=50, return_meta: bool = False
) -> list[dict] | tuple[list[dict], dict]:
    vec_hits = store.search(qvector, k=candidate_k, where=where)
    if hasattr(store, "search_text_with_status"):
        fts_hits, fts_warning = store.search_text_with_status(query, k=candidate_k, where=where)
    else:
        fts_hits = store.search_text(query, k=candidate_k, where=where)
        fts_warning = None

    fused: dict[str, dict] = {}

    def slot(h: dict) -> dict:
        cid = h.get("chunk_id")
        e = fused.get(cid)
        if e is None:
            e = {"hit": h, "score": 0.0}
            fused[cid] = e
        return e

    for rank, h in enumerate(vec_hits, 1):
        slot(h)["score"] += _rrf(rank)
    for rank, h in enumerate(fts_hits, 1):
        slot(h)["score"] += _rrf(rank)

    qtoks = set(_tokens(query))
    for e in fused.values():
        h = e["hit"]
        sym = h.get("symbol") or ""
        basename = (h.get("rel_path") or "").rsplit("/", 1)[-1]
        sym_toks = set(_tokens(sym))
        base_toks = set(_tokens(basename))
        if qtoks & sym_toks:
            e["score"] += 0.10 * len(qtoks & sym_toks)
        if sym.lower() in qtoks:  # the exact symbol name appears in the query
            e["score"] += 0.20
        if qtoks & base_toks:
            e["score"] += 0.05
        e["score"] += _role_boost(h.get("chunk_role"))

    ranked = sorted(fused.values(), key=lambda e: e["score"], reverse=True)
    out = []
    for e in ranked[:k]:
        h = dict(e["hit"])
        h["score"] = round(e["score"], 6)
        out.append(h)
    meta = {
        "warnings": [fts_warning] if fts_warning else [],
        "mode_used": "vector" if fts_warning else "hybrid",
    }
    return (out, meta) if return_meta else out


def normalize_score(raw_score: float, mode_used: str, rank: int, reranked: bool = False) -> float:
    """Map backend-specific scores into a coarse 0..1 decision signal."""

    if reranked:
        base = 1.0 / (1.0 + exp(-max(-20.0, min(20.0, raw_score))))
    elif mode_used == "hybrid":
        # RRF + local boosts: strong exact-token hits are commonly around .20-.35.
        base = max(0.0, min(1.0, raw_score / 0.30))
    else:
        # Lance vector distance is better when smaller; search_project stores
        # score = -distance for vector hits.
        distance = max(0.0, -raw_score)
        base = 1.0 / (1.0 + distance)
    rank_prior = max(0.0, 1.0 - (rank - 1) * 0.12)
    return round(max(0.0, min(1.0, 0.80 * base + 0.20 * rank_prior)), 3)


def relevance_bucket(score_normalized: float) -> str:
    if score_normalized >= HIGH_RELEVANCE:
        return "high"
    if score_normalized >= MEDIUM_RELEVANCE:
        return "medium"
    if score_normalized >= LOW_RELEVANCE:
        return "low"
    return "uncertain"


def relevance_at_least(value: str, threshold: str | None) -> bool:
    if threshold is None:
        return True
    if threshold not in RELEVANCE_ORDER:
        raise ValueError(
            "min_relevance must be one of: "
            + ", ".join(sorted(RELEVANCE_ORDER, key=RELEVANCE_ORDER.get))
        )
    return RELEVANCE_ORDER[value] >= RELEVANCE_ORDER[threshold]


def match_reason(hit: dict, query: str, mode_used: str) -> str:
    qtoks = set(_tokens(query))
    sym = hit.get("symbol") or ""
    path = hit.get("rel_path") or ""
    sym_toks = set(_tokens(sym))
    path_toks = set(_tokens(path.rsplit("/", 1)[-1]))
    if sym and (sym.lower() in qtoks or qtoks & sym_toks):
        return "query token matches the indexed symbol"
    if qtoks & path_toks:
        return "query token matches the file name"
    if mode_used == "hybrid":
        return "ranked by vector similarity plus full-text retrieval"
    return "ranked by vector similarity to the indexed source"
