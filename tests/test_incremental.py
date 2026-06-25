"""Incremental indexing, single-file reindex, and project removal."""

from __future__ import annotations

import pytest


def test_incremental_change_add_delete(tmp_path, monkeypatch, provider):
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path / "home"))
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "a.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (proj / "b.py").write_text("def subtract(a, b):\n    return a - b\n", encoding="utf-8")

    from engram_mcp.pipeline import index_project, search_project

    index_project(proj, provider)

    (proj / "a.py").write_text("def add(a, b, c):\n    return a + b + c\n", encoding="utf-8")
    (proj / "c.py").write_text("def multiply(a, b):\n    return a * b\n", encoding="utf-8")
    (proj / "b.py").unlink()

    stats = index_project(proj, provider)
    assert stats.mode == "incremental"
    assert stats.changed == 1
    assert stats.added == 1
    assert stats.deleted == 1

    # deleted file's chunks are gone
    hits = search_project(proj, provider, "subtract two numbers", k=5)
    assert all(h["rel_path"] != "b.py" for h in hits)
    # added file is searchable
    hits2 = search_project(proj, provider, "multiply two numbers", k=5)
    assert any(h["rel_path"] == "c.py" for h in hits2)


def test_reindex_single_file(tmp_path, monkeypatch, provider):
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path / "home"))
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")

    from engram_mcp.pipeline import index_project, reindex_file, search_project

    index_project(proj, provider)
    (proj / "a.py").write_text("def banana_split(x):\n    return x\n", encoding="utf-8")
    res = reindex_file(proj, provider, "a.py")
    assert res["action"] == "reindexed"

    hits = search_project(proj, provider, "banana split function", k=1)
    assert hits and "banana_split" in (hits[0]["symbol"] or "")


def test_remove_project(tmp_path, monkeypatch, provider):
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path / "home"))
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")

    from engram_mcp.pipeline import (
        ProjectNotIndexedError,
        index_project,
        remove_project,
        search_project,
    )

    index_project(proj, provider)
    assert remove_project(proj) is True
    with pytest.raises(ProjectNotIndexedError):
        search_project(proj, provider, "anything")
