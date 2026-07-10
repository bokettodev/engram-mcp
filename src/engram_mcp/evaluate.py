"""Retrieval evaluation with paraphrase/distractor metrics.

Eval file: a JSON list of cases, each:
  {"query": str,
   "expected_path": str            # single expected file, OR
   "expected_paths": [str, ...],   # any-of (ambiguous queries),
   "expected_symbol": str?,        # optional substring match on symbol
   "distractor_paths": [str, ...], # hard negatives that should not outrank target
   "category": str?}               # nl | paraphrase | exact_symbol | ...

Runtime evaluation is LLM-free and uses the same torch-free ONNX reranker as
server search when rerank is enabled.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

_TOKEN = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class CategoryStats:
    n: int
    hit1: float
    hit5: float
    hit10: float
    mrr: float
    hnsr5: float
    hnsr10: float
    delta_rank: float | None


@dataclass
class EvalReport:
    overall: CategoryStats
    by_category: dict[str, CategoryStats]
    by_overlap_bucket: dict[str, CategoryStats]
    mean_latency_ms: float
    rows: list[dict] = field(default_factory=list)
    rerank_requested: bool = False
    rerank_applied_count: int = 0
    rerank_skipped_reasons: dict[str, int] = field(default_factory=dict)


def load_cases(path: str | Path) -> list[dict]:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_cases(cases)
    return cases


def validate_cases(cases: list[dict]) -> None:
    if not isinstance(cases, list):
        raise ValueError("eval fixture must be a JSON list")
    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"eval case {i} must be an object")
        if not isinstance(case.get("query"), str) or not case["query"].strip():
            raise ValueError(f"eval case {i} must have a non-empty query")
        if "category" in case and not isinstance(case["category"], str):
            raise ValueError(f"eval case {i} category must be a string")
        has_one = isinstance(case.get("expected_path"), str) and bool(case["expected_path"])
        paths = case.get("expected_paths")
        has_many = (
            isinstance(paths, list)
            and bool(paths)
            and all(isinstance(p, str) and p for p in paths)
        )
        if not has_one and not has_many:
            raise ValueError(f"eval case {i} must have expected_path or expected_paths")
        if "expected_symbol" in case and not isinstance(case["expected_symbol"], str):
            raise ValueError(f"eval case {i} expected_symbol must be a string")
        distractors = case.get("distractor_paths", [])
        if not isinstance(distractors, list) or not all(
            isinstance(p, str) and p for p in distractors
        ):
            raise ValueError(f"eval case {i} distractor_paths must be a list of strings")


def _expected_set(case: dict) -> set[str]:
    paths = case.get("expected_paths")
    if paths:
        return set(paths)
    one = case.get("expected_path")
    return {one} if one else set()


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(text or "")}


def _read_expected_text(root: Path, expected: set[str]) -> str:
    """Whole-file text for `expected` paths.

    Used only to validate fixture *construction* (e.g. that a paraphrase
    query doesn't share vocabulary with its target file at all -- see
    test_eval_fixtures.py). `run_evaluation`'s lexical-overlap bucketing uses
    `_expected_chunk_text` instead: the actual retrieval targets, not the
    whole file (see its docstring for why).
    """
    parts = []
    for rel in expected:
        try:
            parts.append((root / rel).read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(parts)


def _expected_chunk_text(store, expected: set[str], expected_symbol: str | None) -> str:
    """Text of the chunk(s) that would count as a correct hit for a case.

    `_rank_for` scores a hit as correct when its `rel_path` is in `expected`
    and (if `expected_symbol` is set) `expected_symbol` is a substring of its
    `symbol` -- this mirrors that predicate against the indexed corpus itself
    rather than against the top-k search results, so it's defined even for a
    case the search missed entirely. Using the whole file's text (the
    previous behavior) diluted lexical overlap for any file with more than a
    handful of lines: a query naming one constant or function scored "low
    overlap" purely because the rest of a large file didn't mention it, and
    73/80 of the checked-in eval cases collapsed into that one bucket. Falls
    back to all of a file's chunks if a `expected_symbol` filter matches none
    of them (e.g. the symbol isn't chunk-indexed), so a case never silently
    loses its overlap signal.
    """
    parts: list[str] = []
    for rel in expected:
        rows = store.by_rel_path(rel)
        if expected_symbol:
            matched = [r for r in rows if expected_symbol in (r.get("symbol") or "")]
            rows = matched or rows
        parts.extend(r.get("content") or "" for r in rows)
    return "\n".join(parts)


def _jaccard(query: str, target_text: str) -> float:
    q = _tokens(query)
    t = _tokens(target_text)
    if not q or not t:
        return 0.0
    return len(q & t) / len(q | t)


def _overlap_bucket(score: float) -> str:
    if score == 0:
        return "zero"
    if score < 0.03:
        return "low"
    if score < 0.10:
        return "medium"
    return "high"


def _stats(rows: list[dict]) -> CategoryStats:
    n = len(rows)
    if n == 0:
        return CategoryStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None)
    ranks = [r["rank"] for r in rows]
    deltas = [r["delta_rank"] for r in rows if r["delta_rank"] is not None]
    return CategoryStats(
        n=n,
        hit1=sum(1 for r in ranks if r == 1) / n,
        hit5=sum(1 for r in ranks if r and r <= 5) / n,
        hit10=sum(1 for r in ranks if r and r <= 10) / n,
        mrr=sum(1.0 / r for r in ranks if r) / n,
        hnsr5=sum(1 for r in rows if r["hnsr5"]) / n,
        hnsr10=sum(1 for r in rows if r["hnsr10"]) / n,
        delta_rank=(sum(deltas) / len(deltas) if deltas else None),
    )


def _rank_for(
    hits: list[dict], expected: set[str], expected_symbol: str | None
) -> int | None:
    for i, h in enumerate(hits, 1):
        path_ok = (not expected) or (h.get("rel_path") in expected)
        sym_ok = expected_symbol is None or expected_symbol in (h.get("symbol") or "")
        if path_ok and sym_ok:
            return i
    return None


def _distractor_rank(hits: list[dict], distractors: set[str]) -> int | None:
    if not distractors:
        return None
    for i, h in enumerate(hits, 1):
        if h.get("rel_path") in distractors:
            return i
    return None


# --- Baseline recording + non-inferiority gate --------------------------------
#
# Embedding is not bit-reproducible across platforms/onnxruntime versions, so a
# baseline gate can never require an exact metric match without becoming flaky.
# Instead each metric may drop by at most `DEFAULT_NONINFERIORITY_MARGIN`
# (absolute) below its recorded baseline value before the run is considered a
# regression -- non-inferiority, not equality.
DEFAULT_NONINFERIORITY_MARGIN = 0.05
BASELINE_SCHEMA_VERSION = 1


def _metrics(stats: CategoryStats) -> dict:
    return {
        "n": stats.n,
        "hit1": round(stats.hit1, 4),
        "hit5": round(stats.hit5, 4),
        "hit10": round(stats.hit10, 4),
        "mrr": round(stats.mrr, 4),
    }


def report_to_baseline(
    report: EvalReport, *, evalfile: str, mode: str, rerank: bool = False
) -> dict:
    """Serialize the metrics that matter for regression-gating (not the full
    per-row report -- that's diagnostic output, not something to freeze)."""
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "evalfile": str(evalfile),
        "mode": mode,
        "rerank": bool(rerank),
        "overall": _metrics(report.overall),
        "by_category": {cat: _metrics(s) for cat, s in sorted(report.by_category.items())},
    }


def save_baseline(
    path: str | Path, report: EvalReport, *, evalfile: str, mode: str, rerank: bool = False
) -> None:
    data = report_to_baseline(report, evalfile=evalfile, mode=mode, rerank=rerank)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def load_baseline(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "overall" not in data:
        raise ValueError(f"not a valid eval baseline file: {path}")
    return data


def compare_to_baseline(
    report: EvalReport, baseline: dict, margin: float = DEFAULT_NONINFERIORITY_MARGIN
) -> dict:
    """Non-inferiority check: a metric fails only if it drops more than
    `margin` (absolute) below the recorded baseline value. Checks overall
    hit@1/5/10 + MRR, and the same per category for every category present in
    the baseline (so a regression localized to one category, e.g. partial_id,
    is still caught even if the overall numbers look fine).
    """
    checks: list[dict] = []

    def _check(label: str, current: float, expected: float | None) -> None:
        if expected is None:
            return
        delta = round(current - expected, 4)
        checks.append({
            "metric": label,
            "current": round(current, 4),
            "baseline": round(expected, 4),
            "delta": delta,
            "ok": delta >= -margin,
        })

    b_overall = baseline.get("overall") or {}
    o = report.overall
    _check("overall.hit1", o.hit1, b_overall.get("hit1"))
    _check("overall.hit5", o.hit5, b_overall.get("hit5"))
    _check("overall.hit10", o.hit10, b_overall.get("hit10"))
    _check("overall.mrr", o.mrr, b_overall.get("mrr"))

    for cat, b_stats in sorted((baseline.get("by_category") or {}).items()):
        s = report.by_category.get(cat)
        if s is None:
            checks.append({
                "metric": f"category[{cat}].hit5", "current": None,
                "baseline": b_stats.get("hit5"), "delta": None, "ok": False,
                "reason": f"category {cat!r} is in the baseline but absent from this run",
            })
            continue
        _check(f"category[{cat}].hit1", s.hit1, b_stats.get("hit1"))
        _check(f"category[{cat}].hit5", s.hit5, b_stats.get("hit5"))
        _check(f"category[{cat}].hit10", s.hit10, b_stats.get("hit10"))
        _check(f"category[{cat}].mrr", s.mrr, b_stats.get("mrr"))

    failures = [c for c in checks if not c["ok"]]
    return {"ok": not failures, "margin": margin, "checks": checks, "failures": failures}


def run_evaluation(
    root, provider, cases, k: int = 10, mode: str = "auto", rerank: bool = False
) -> EvalReport:
    from engram_mcp.index_repository import load_query_index
    from engram_mcp.query_service import search_project

    root = Path(root)
    qi = load_query_index(root)
    rows: list[dict] = []
    latencies: list[float] = []
    for case in cases:
        query = case["query"]
        expected = _expected_set(case)
        distractors = set(case.get("distractor_paths") or [])
        exp_sym = case.get("expected_symbol")
        category = case.get("category", "uncategorized")
        t0 = time.time()
        outcome = search_project(
            root, provider, query, k=k, mode=mode, rerank=rerank,
            return_meta=True, _query_index=qi,
        )
        latencies.append((time.time() - t0) * 1000)
        hits = outcome["hits"]
        rank = _rank_for(hits, expected, exp_sym)
        hard_rank = _distractor_rank(hits, distractors)
        target_text = _expected_chunk_text(qi.store, expected, exp_sym)
        overlap = _jaccard(query, target_text)
        delta_rank = None
        if rank is not None and hard_rank is not None:
            delta_rank = hard_rank - rank
        elif rank is not None and distractors:
            delta_rank = (k + 1) - rank
        rows.append(
            {
                "query": query,
                "category": category,
                "expected": sorted(expected),
                "distractors": sorted(distractors),
                "rank": rank,
                "hard_negative_rank": hard_rank,
                "delta_rank": delta_rank,
                "hnsr5": bool(rank and rank <= 5 and (hard_rank is None or hard_rank > 5)),
                "hnsr10": bool(rank and rank <= 10 and (hard_rank is None or hard_rank > 10)),
                "lexical_overlap": round(overlap, 4),
                "overlap_bucket": _overlap_bucket(overlap),
                "top": hits[0]["rel_path"] if hits else None,
                "rerank_applied": bool(outcome.get("rerank_applied")),
                "rerank_skipped_reason": outcome.get("rerank_skipped_reason"),
            }
        )

    by_cat_rows: dict[str, list] = defaultdict(list)
    by_bucket_rows: dict[str, list] = defaultdict(list)
    for r in rows:
        by_cat_rows[r["category"]].append(r)
        by_bucket_rows[r["overlap_bucket"]].append(r)
    by_category = {c: _stats(rs) for c, rs in by_cat_rows.items()}
    by_overlap_bucket = {c: _stats(rs) for c, rs in by_bucket_rows.items()}
    overall = _stats(rows)
    mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
    skipped: dict[str, int] = {}
    for row in rows:
        reason = row.get("rerank_skipped_reason")
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
    return EvalReport(
        overall=overall,
        by_category=by_category,
        by_overlap_bucket=by_overlap_bucket,
        mean_latency_ms=mean_lat,
        rows=rows,
        rerank_requested=rerank,
        rerank_applied_count=sum(1 for row in rows if row.get("rerank_applied")),
        rerank_skipped_reasons=skipped,
    )
