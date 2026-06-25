"""Retrieval evaluation with per-category breakdown.

Eval file: a JSON list of cases, each:
  {"query": str,
   "expected_path": str            # single expected file, OR
   "expected_paths": [str, ...],   # any-of (ambiguous queries),
   "expected_symbol": str?,        # optional substring match on symbol
   "category": str?}               # nl | exact_symbol | partial_id | ... }

A case hits at rank i when the i-th result is in the expected path set (and, if
given, its symbol contains expected_symbol). Reports hit@1/5/10 + MRR overall
and per category, so the vector-vs-hybrid tradeoff can be judged by query type.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CategoryStats:
    n: int
    hit1: float
    hit5: float
    hit10: float
    mrr: float


@dataclass
class EvalReport:
    overall: CategoryStats
    by_category: dict[str, CategoryStats]
    mean_latency_ms: float
    rows: list[dict] = field(default_factory=list)


def load_cases(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _expected_set(case: dict) -> set[str]:
    paths = case.get("expected_paths")
    if paths:
        return set(paths)
    one = case.get("expected_path")
    return {one} if one else set()


def _stats(ranks: list[int | None]) -> CategoryStats:
    n = len(ranks)
    if n == 0:
        return CategoryStats(0, 0.0, 0.0, 0.0, 0.0)
    return CategoryStats(
        n=n,
        hit1=sum(1 for r in ranks if r == 1) / n,
        hit5=sum(1 for r in ranks if r and r <= 5) / n,
        hit10=sum(1 for r in ranks if r and r <= 10) / n,
        mrr=sum(1.0 / r for r in ranks if r) / n,
    )


def run_evaluation(
    root, provider, cases, k: int = 10, mode: str = "auto", rerank: bool = False
) -> EvalReport:
    from engram_mcp.pipeline import search_project

    rows: list[dict] = []
    latencies: list[float] = []
    for case in cases:
        query = case["query"]
        expected = _expected_set(case)
        exp_sym = case.get("expected_symbol")
        category = case.get("category", "uncategorized")
        t0 = time.time()
        hits = search_project(root, provider, query, k=k, mode=mode, rerank=rerank)
        latencies.append((time.time() - t0) * 1000)
        rank = None
        for i, h in enumerate(hits, 1):
            path_ok = (not expected) or (h.get("rel_path") in expected)
            sym_ok = exp_sym is None or exp_sym in (h.get("symbol") or "")
            if path_ok and sym_ok:
                rank = i
                break
        rows.append(
            {
                "query": query,
                "category": category,
                "expected": sorted(expected),
                "rank": rank,
                "top": hits[0]["rel_path"] if hits else None,
            }
        )

    by_cat_ranks: dict[str, list] = defaultdict(list)
    for r in rows:
        by_cat_ranks[r["category"]].append(r["rank"])
    by_category = {c: _stats(rk) for c, rk in by_cat_ranks.items()}
    overall = _stats([r["rank"] for r in rows])
    mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
    return EvalReport(overall=overall, by_category=by_category, mean_latency_ms=mean_lat, rows=rows)
