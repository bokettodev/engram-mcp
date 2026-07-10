"""Tests for the MCP server logic + tool registration."""

from __future__ import annotations

import asyncio
import io
import os
import threading
import time

import pytest


def test_tools_are_registered():
    """No model needed: the four tools must be exposed by the FastMCP app."""
    from engram_mcp import server

    tools = asyncio.run(server.mcp.list_tools())
    names = {t.name for t in tools}
    assert {
        "index_project", "cancel_index", "index_status", "search_code",
        "list_indexed_projects", "get_chunk", "model_status", "server_info",
    } <= names


def test_read_only_mode_hides_mutating_tools(monkeypatch):
    """ENGRAM_READONLY=1 exposes only the read tools; mutating tools are withheld."""
    import importlib

    from engram_mcp import server as server_mod

    monkeypatch.setenv("ENGRAM_READONLY", "1")
    importlib.reload(server_mod)
    try:
        names = {t.name for t in asyncio.run(server_mod.mcp.list_tools())}
        assert {
            "search_code", "find_definition", "get_chunk", "model_status",
            "index_status", "list_indexed_projects", "server_info",
        } <= names
        assert not (
            {"index_project", "cancel_index", "reindex_file", "remove_project"} & names
        )
    finally:
        # restore the full tool surface for subsequent tests
        monkeypatch.delenv("ENGRAM_READONLY", raising=False)
        importlib.reload(server_mod)


def test_unknown_job_and_empty_list(tmp_path, monkeypatch):
    from engram_mcp import server

    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path / "home"))
    assert "error" in server.get_status("does-not-exist")
    listed = server.list_projects()
    assert listed["projects"] == []
    assert listed["errors"] == []
    assert listed["projects_empty"] is True
    assert listed["home_exists"] is False
    assert listed["data_home"] == str(tmp_path / "home")


def test_read_only_list_projects_does_not_prune_orphans(tmp_path, monkeypatch):
    from engram_mcp import inventory, paths

    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path / "home"))
    paths._reset_data_home_for_tests()
    base = paths.data_home() / "projects" / "orphan"
    base.mkdir(parents=True)
    missing_root = tmp_path / "missing-root"
    (base / "project.json").write_text(
        (
            "{"
            '"project_id":"orphan",'
            f'"root_path":"{missing_root.as_posix()}",'
            '"schema_version":3'
            "}"
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ENGRAM_READONLY", "1")

    listed = inventory.list_indexed_projects(prune_orphans=True)

    assert base.exists()
    assert listed["gc"] == {"pruned": [], "errors": []}


def _write_orphan_project(tmp_path, monkeypatch, project_id="orphan"):
    from engram_mcp import paths

    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path / "home"))
    paths._reset_data_home_for_tests()
    base = paths.data_home() / "projects" / project_id
    base.mkdir(parents=True)
    missing_root = tmp_path / "missing-root"
    (base / "project.json").write_text(
        (
            "{"
            f'"project_id":"{project_id}",'
            f'"root_path":"{missing_root.as_posix()}",'
            '"schema_version":3'
            "}"
        ),
        encoding="utf-8",
    )
    return base


def test_list_indexed_projects_default_does_not_prune_orphans(tmp_path, monkeypatch):
    """A mere listing call must never delete an index just because its
    recorded root looks momentarily missing (disconnected drive, unmounted
    share, renamed workspace). prune_orphans defaults to False.
    """

    from engram_mcp import inventory

    base = _write_orphan_project(tmp_path, monkeypatch)

    listed = inventory.list_indexed_projects()

    assert base.exists()
    assert listed["gc"] == {"pruned": [], "errors": []}


def test_mcp_tool_list_indexed_projects_default_does_not_prune_orphans(tmp_path, monkeypatch):
    from engram_mcp import server

    base = _write_orphan_project(tmp_path, monkeypatch)

    listed = asyncio.run(server.list_indexed_projects())

    assert base.exists()
    assert listed["gc"] == {"pruned": [], "errors": []}


def test_engram_gc_prune_cli_path_still_deletes_orphans(tmp_path, monkeypatch):
    """Pruning remains reachable, but only via the explicit operator CLI path."""

    from engram_mcp.inventory import gc_orphans

    base = _write_orphan_project(tmp_path, monkeypatch)

    dry_run = gc_orphans(prune=False)
    assert base.exists()
    assert dry_run["orphans"] and dry_run["pruned"] == []

    pruned = gc_orphans(prune=True)
    assert not base.exists()
    assert pruned["pruned"]


def test_server_info_reports_fastembed_reranker_without_loading(monkeypatch):
    from engram_mcp import server
    from engram_mcp.rerankers import DEFAULT_ONNX_RERANKER

    monkeypatch.delenv("ENGRAM_RERANKER_MODEL", raising=False)
    monkeypatch.setenv("ENGRAM_RERANK_CANDIDATE_K", "13")
    out = server.do_server_info()
    assert "reranker_default_backend" not in out
    assert "fastembed_onnx_reranker" not in out
    assert out["reranker"]["default_backend"] == "fastembed"
    assert out["reranker"]["onnx_model"] == DEFAULT_ONNX_RERANKER
    assert out["reranker"]["onnx_available"] is True
    assert out["reranker"]["candidate_k_default"] == 13
    assert "CC-BY-NC-4.0" in out["reranker"]["license_note"]


def test_job_snapshot_timestamps_and_update_sequence(tmp_path):
    from engram_mcp.jobs import JobRegistry, snapshot

    reg = JobRegistry()
    job = reg.create(str(tmp_path))
    first = snapshot(job)
    assert first["created_at"]
    assert first["updated_at"] == first["created_at"]
    assert first["update_seq"] == 0
    assert first["finished_at"] is None
    assert first["progress"]["total"] is None

    reg.update(job.job_id, status="running", stage="scanning", started_at=time.time())
    second = snapshot(reg.get(job.job_id))
    reg.update(job.job_id, stage="embedding", done_units=1, total_units=3)
    third = snapshot(reg.get(job.job_id))

    assert second["update_seq"] == 1
    assert third["update_seq"] == 2
    assert third["updated_at"] >= second["updated_at"]
    assert third["duration_sec"] >= 0
    assert third["seconds_since_update"] >= 0
    assert third["progress"] == {"unit": None, "done": 1, "total": 3}


def test_bad_path_raises(tmp_path):
    from engram_mcp import server

    with pytest.raises(ValueError):
        server.start_index_job(str(tmp_path / "nope"))


@pytest.mark.skipif(
    os.environ.get("ENGRAM_SKIP_MODEL") == "1", reason="model-dependent test disabled"
)
def test_async_index_job_then_search(tmp_path, monkeypatch):
    """End-to-end: index_project's real subprocess path, then search.

    index_device defaults to "auto", which for a project this small routes to
    "cpu" -- but even "cpu" now runs in the real `engram index --json`
    subprocess (see server._subprocess_index), never in-process, so this
    exercises the actual child-process spawn/JSON-stream contract.
    """
    from engram_mcp import server
    from engram_mcp.embeddings import factory

    try:
        factory.provider_for_model_id(factory.CANONICAL_EMBEDDER_ID)
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
    for _ in range(240):
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


@pytest.mark.skipif(
    os.environ.get("ENGRAM_SKIP_MODEL") == "1", reason="model-dependent test disabled"
)
def test_search_query_lock_not_held_while_cpu_index_job_is_in_flight(tmp_path, monkeypatch):
    """A running "cpu" index job must never hold the real cached query
    provider's inference lock -- search would otherwise queue behind it for
    the whole embedding batch (measured ~19s for a 256-item batch before this
    fix). Uses the actual process-wide cached FastEmbed singleton (the same
    one search uses), not a fake, so this only proves something if the index
    job genuinely never touches it.
    """
    from engram_mcp import server
    from engram_mcp.embeddings import factory

    try:
        query_provider = factory.provider_for_model_id(factory.CANONICAL_EMBEDDER_ID)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"embedder unavailable: {exc}")

    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path / "home"))
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")

    reached_embedding = threading.Event()
    release_subprocess = threading.Event()

    class StuckStdout:
        def __iter__(self):
            yield (
                '{"event": "progress", "version": 1, "seq": 1, "stage": "embedding", '
                '"unit": "embeddings", "done": 1, "total": 999, "chunks": 999, '
                '"embedded": 1, "reused": 0}\n'
            )
            reached_embedding.set()
            release_subprocess.wait(timeout=10)

    class StuckPopen:
        stdout = StuckStdout()
        stderr = io.StringIO("")
        returncode = 0

        def wait(self):
            return self.returncode

    monkeypatch.setattr(server.subprocess, "Popen", lambda *a, **k: StuckPopen())

    try:
        server.start_index_job(str(proj), index_device="cpu")
        assert reached_embedding.wait(timeout=2)

        # The index job is "in flight" (its subprocess is stuck mid-batch).
        # The real cached query provider's lock must still be free.
        acquired = query_provider._lock.acquire(blocking=False)
        try:
            assert acquired, "the cached query provider's lock was held by the index job"
        finally:
            if acquired:
                query_provider._lock.release()
    finally:
        release_subprocess.set()
