"""Hybrid retrieval fusion/boosts (no model) + end-to-end modes + eval (model)."""

from __future__ import annotations

from engram_mcp import retrieval


class _FakeStore:
    def search(self, vector, k, where=None):
        return [
            {"chunk_id": "a", "symbol": "foo", "rel_path": "x.py"},
            {"chunk_id": "b", "symbol": "bar", "rel_path": "y.py"},
        ]

    def search_text(self, query, k, where=None):
        return [{"chunk_id": "b", "symbol": "bar", "rel_path": "y.py"}]


class _FakeChunkStore:
    """Minimal stand-in for LanceStore.by_rel_path, no index/model needed."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def by_rel_path(self, rel_path: str, k: int = 500) -> list[dict]:
        return [r for r in self._rows if r["rel_path"] == rel_path]


def test_expected_chunk_text_uses_matching_chunk_not_whole_file():
    from engram_mcp import evaluate

    rows = [
        {
            "rel_path": "big.py", "symbol": "unrelated_helper",
            "content": "def unrelated_helper():\n    pass\n" * 50,
        },
        {"rel_path": "big.py", "symbol": "MAX_FILE_BYTES", "content": "MAX_FILE_BYTES = 8_000_000\n"},
    ]
    text = evaluate._expected_chunk_text(_FakeChunkStore(rows), {"big.py"}, "MAX_FILE_BYTES")
    assert text == "MAX_FILE_BYTES = 8_000_000\n"
    assert "unrelated_helper" not in text


def test_expected_chunk_text_unions_all_chunks_without_a_symbol_filter():
    from engram_mcp import evaluate

    rows = [
        {"rel_path": "a.py", "symbol": "x", "content": "x = 1\n"},
        {"rel_path": "a.py", "symbol": "y", "content": "y = 2\n"},
    ]
    text = evaluate._expected_chunk_text(_FakeChunkStore(rows), {"a.py"}, None)
    assert "x = 1" in text and "y = 2" in text


def test_expected_chunk_text_falls_back_when_symbol_filter_matches_nothing():
    from engram_mcp import evaluate

    rows = [{"rel_path": "a.py", "symbol": "other", "content": "other = 1\n"}]
    text = evaluate._expected_chunk_text(_FakeChunkStore(rows), {"a.py"}, "missing_symbol")
    assert "other = 1" in text  # fallback to all of the file's chunks, never empty


def _stats(hit1=0.8, hit5=0.9, hit10=0.95, mrr=0.85, n=10):
    from engram_mcp.evaluate import CategoryStats

    return CategoryStats(n=n, hit1=hit1, hit5=hit5, hit10=hit10, mrr=mrr, hnsr5=0.0, hnsr10=0.0, delta_rank=None)


def _report(overall, by_category=None):
    from engram_mcp.evaluate import EvalReport

    return EvalReport(overall=overall, by_category=by_category or {}, by_overlap_bucket={}, mean_latency_ms=1.0)


def test_report_to_baseline_round_trips_through_save_and_load(tmp_path):
    from engram_mcp import evaluate

    report = _report(_stats(), {"partial_id": _stats(hit1=0.3, hit5=0.5, n=7)})
    path = tmp_path / "baseline.json"
    evaluate.save_baseline(path, report, evalfile="evals/self.json", mode="auto")
    baseline = evaluate.load_baseline(path)
    assert baseline["mode"] == "auto"
    assert baseline["evalfile"] == "evals/self.json"
    assert baseline["overall"]["hit5"] == 0.9
    assert baseline["by_category"]["partial_id"]["hit5"] == 0.5


def test_compare_to_baseline_passes_within_margin():
    from engram_mcp import evaluate

    baseline = {"overall": {"hit1": 0.60, "hit5": 0.80, "hit10": 0.85, "mrr": 0.65}, "by_category": {}}
    # a small drop within the default margin (0.05) is not a regression
    report = _report(_stats(hit1=0.58, hit5=0.78, hit10=0.85, mrr=0.64))
    result = evaluate.compare_to_baseline(report, baseline)
    assert result["ok"] is True
    assert result["failures"] == []


def test_compare_to_baseline_fails_on_a_real_regression():
    from engram_mcp import evaluate

    baseline = {"overall": {"hit1": 0.60, "hit5": 0.80, "hit10": 0.85, "mrr": 0.65}, "by_category": {}}
    # hit5 drops 0.20, well past the default 0.05 margin
    report = _report(_stats(hit1=0.60, hit5=0.60, hit10=0.85, mrr=0.65))
    result = evaluate.compare_to_baseline(report, baseline)
    assert result["ok"] is False
    failed_metrics = {f["metric"] for f in result["failures"]}
    assert "overall.hit5" in failed_metrics


def test_compare_to_baseline_catches_a_per_category_regression_even_if_overall_holds():
    from engram_mcp import evaluate

    baseline = {
        "overall": {"hit1": 0.60, "hit5": 0.80, "hit10": 0.85, "mrr": 0.65},
        "by_category": {"partial_id": {"hit1": 0.80, "hit5": 1.0, "hit10": 1.0, "mrr": 0.9}},
    }
    # overall numbers are flat/improved, but partial_id alone collapsed
    report = _report(
        _stats(hit1=0.60, hit5=0.82, hit10=0.88, mrr=0.66),
        {"partial_id": _stats(hit1=0.10, hit5=0.20, hit10=0.30, mrr=0.15, n=7)},
    )
    result = evaluate.compare_to_baseline(report, baseline)
    assert result["ok"] is False
    assert any(f["metric"].startswith("category[partial_id]") for f in result["failures"])


def test_compare_to_baseline_fails_when_baseline_category_goes_missing():
    from engram_mcp import evaluate

    baseline = {
        "overall": {"hit1": 0.60, "hit5": 0.80, "hit10": 0.85, "mrr": 0.65},
        "by_category": {"partial_id": {"hit1": 0.8, "hit5": 1.0, "hit10": 1.0, "mrr": 0.9}},
    }
    report = _report(_stats(hit1=0.60, hit5=0.80, hit10=0.85, mrr=0.65))  # no partial_id at all
    result = evaluate.compare_to_baseline(report, baseline)
    assert result["ok"] is False


def test_classify_query_routes_modes():
    from engram_mcp.retrieval import classify_query

    assert classify_query("EmbeddingCache") == "hybrid"  # camelCase
    assert classify_query("MAX_FILE_BYTES") == "hybrid"  # ALL_CAPS
    assert classify_query("_search_text helper") == "hybrid"  # snake_case
    assert classify_query("chunks_g table name") == "hybrid"  # snake_case
    assert classify_query("open the config.py module") == "hybrid"  # file ext
    assert classify_query('the literal "project not indexed"') == "hybrid"  # quoted
    assert classify_query("who calls _embed") == "hybrid"  # leading-underscore ident
    assert classify_query("main") == "hybrid"  # bare single token
    assert classify_query("_rrf") == "hybrid"
    assert classify_query("where are access rights validated") == "vector"
    assert classify_query("how does the embedding cache work") == "vector"


def test_rrf_fuses_and_boosts_exact_symbol():
    out = retrieval.hybrid_search(_FakeStore(), "the bar function", [0.0], k=2)
    assert [h["chunk_id"] for h in out][0] == "b"  # in both lists + symbol matches query
    assert all("score" in h for h in out)
    assert out[0]["score"] >= out[1]["score"]


def test_end_to_end_modes_find_symbol(tmp_path, monkeypatch, provider):
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path / "home"))
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "math_utils.py").write_text(
        "def add_numbers(a, b):\n    '''sum two numbers'''\n    return a + b\n", encoding="utf-8"
    )
    (proj / "io_utils.py").write_text(
        "def read_file(p):\n    '''read a file from disk'''\n    return open(p).read()\n",
        encoding="utf-8",
    )

    from engram_mcp.pipeline import index_project, search_project

    index_project(proj, provider)
    for mode in ("hybrid", "vector"):
        hits = search_project(proj, provider, "function that adds two numbers", k=3, mode=mode)
        assert hits, mode
        assert hits[0]["rel_path"] == "math_utils.py", mode
        assert "score" in hits[0]


def test_evaluation_harness(tmp_path, monkeypatch, provider):
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path / "home"))
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "math_utils.py").write_text("def add_numbers(a, b):\n    return a + b\n", encoding="utf-8")

    from engram_mcp import evaluate
    from engram_mcp.pipeline import index_project

    index_project(proj, provider)
    cases = [
        {"query": "add two numbers", "expected_path": "math_utils.py",
         "expected_symbol": "add_numbers", "category": "nl",
         "distractor_paths": ["missing.py"]},
        {"query": "function add_numbers", "expected_paths": ["math_utils.py"], "category": "exact_symbol"},
    ]
    report = evaluate.run_evaluation(proj, provider, cases, k=5)
    assert report.overall.n == 2
    assert report.overall.hit5 == 1.0
    assert report.overall.hnsr5 == 1.0
    assert report.rows[0]["delta_rank"] is not None
    assert report.by_overlap_bucket
    assert "nl" in report.by_category and "exact_symbol" in report.by_category
    assert all(r["rank"] is not None for r in report.rows)
