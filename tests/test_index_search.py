"""Integration tests: index a tiny project, then semantic-search it.

Uses the real FastEmbed model via the session `provider` fixture (conftest).
"""

from __future__ import annotations


def test_index_then_search_finds_relevant_symbol(tmp_path, monkeypatch, provider):
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path / "home"))

    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "math_utils.py").write_text(
        "def add_numbers(a, b):\n"
        "    '''Return the sum of two numbers.'''\n"
        "    return a + b\n",
        encoding="utf-8",
    )
    (proj / "io_utils.py").write_text(
        "def read_file(path):\n"
        "    '''Read a file from disk and return its text.'''\n"
        "    with open(path) as fh:\n"
        "        return fh.read()\n",
        encoding="utf-8",
    )

    from engram_mcp.pipeline import index_project, search_project

    stats = index_project(proj, provider)
    assert stats.mode == "full"
    assert stats.files == 2
    assert stats.chunks >= 2
    assert stats.embedded_unique >= 2

    hits = search_project(proj, provider, "function that adds two numbers together", k=2)
    assert hits
    assert hits[0]["rel_path"] == "math_utils.py"
    assert hits[0]["symbol"] == "add_numbers"


def test_incremental_second_index_is_noop(tmp_path, monkeypatch, provider):
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path / "home"))
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")

    from engram_mcp.pipeline import index_project

    first = index_project(proj, provider)
    assert first.mode == "full"
    second = index_project(proj, provider)
    assert second.mode == "incremental"
    assert second.embedded_unique == 0
    assert second.added == 0 and second.changed == 0 and second.deleted == 0
    assert second.unchanged >= 1


def test_full_rebuild_reuses_cache(tmp_path, monkeypatch, provider):
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path / "home"))
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")

    from engram_mcp.pipeline import index_project

    first = index_project(proj, provider)
    rebuilt = index_project(proj, provider, full_rebuild=True)
    assert rebuilt.mode == "full"
    # identical content -> embeddings come from the global cache
    assert rebuilt.embedded_unique == 0
    assert rebuilt.reused_unique >= first.embedded_unique
