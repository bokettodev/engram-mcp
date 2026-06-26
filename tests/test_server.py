"""Tests for the MCP server logic + tool registration."""

from __future__ import annotations

import asyncio
import os
import time

import pytest


def test_tools_are_registered():
    """No model needed: the four tools must be exposed by the FastMCP app."""
    from engram_mcp import server

    tools = asyncio.run(server.mcp.list_tools())
    names = {t.name for t in tools}
    assert {"index_project", "index_status", "search_code", "list_indexed_projects"} <= names


def test_read_only_mode_hides_mutating_tools(monkeypatch):
    """ENGRAM_READONLY=1 exposes only the read tools; mutating tools are withheld."""
    import importlib

    from engram_mcp import server as server_mod

    monkeypatch.setenv("ENGRAM_READONLY", "1")
    importlib.reload(server_mod)
    try:
        names = {t.name for t in asyncio.run(server_mod.mcp.list_tools())}
        assert {"search_code", "find_definition", "index_status", "list_indexed_projects"} <= names
        assert not ({"index_project", "reindex_file", "remove_project"} & names)
    finally:
        # restore the full tool surface for subsequent tests
        monkeypatch.delenv("ENGRAM_READONLY", raising=False)
        importlib.reload(server_mod)


def test_unknown_job_and_empty_list(tmp_path, monkeypatch):
    from engram_mcp import server

    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path / "home"))
    assert "error" in server.get_status("does-not-exist")
    assert server.list_projects() == {"projects": []}


def test_bad_path_raises(tmp_path):
    from engram_mcp import server

    with pytest.raises(ValueError):
        server.start_index_job(str(tmp_path / "nope"))


@pytest.mark.skipif(
    os.environ.get("ENGRAM_SKIP_MODEL") == "1", reason="model-dependent test disabled"
)
def test_async_index_job_then_search(tmp_path, monkeypatch):
    from engram_mcp import server

    try:
        server._get_provider()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"embedder unavailable: {exc}")

    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path / "home"))
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "m.py").write_text(
        "def add(a, b):\n    '''sum two numbers'''\n    return a + b\n", encoding="utf-8"
    )

    started = server.start_index_job(str(proj))
    job_id = started["job_id"]
    assert started["status"] in ("queued", "running")

    status = {}
    for _ in range(120):
        status = server.get_status(job_id)
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert status["status"] == "done", status
    assert status["files"] == 1

    out = server.do_search(str(proj), "function that adds two numbers", k=1)
    assert out["count"] == 1
    assert out["results"][0]["symbol"] == "add"

    listed = server.list_projects()
    assert any(p["root_path"] == str(proj.resolve()) for p in listed["projects"])
