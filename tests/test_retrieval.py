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
         "expected_symbol": "add_numbers", "category": "nl"},
        {"query": "function add_numbers", "expected_paths": ["math_utils.py"], "category": "exact_symbol"},
    ]
    report = evaluate.run_evaluation(proj, provider, cases, k=5)
    assert report.overall.n == 2
    assert report.overall.hit5 == 1.0
    assert "nl" in report.by_category and "exact_symbol" in report.by_category
    assert all(r["rank"] is not None for r in report.rows)
