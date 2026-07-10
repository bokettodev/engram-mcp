"""ENGRAM_GIT_ANALYTICS env-var default precedence for the CLI and MCP entry
points. Both must resolve: explicit argument > ENGRAM_GIT_ANALYTICS > the
built-in default (enabled). Neither test loads a real embedding model, so
they run under ENGRAM_SKIP_MODEL=1.

The MCP-level tests fake the index subprocess boundary (every index device,
including "cpu", now always dispatches through _subprocess_index -- see the
CPU-indexing-blocks-search fix) and assert on the child command line, since
the actual env-var -> bool resolution for an unset git_analytics now happens
in the child `engram index` process, not in this one; that resolution is
covered separately by test_cli_index_git_analytics_env_precedence below.
"""

from __future__ import annotations

import asyncio
import io
import time

from engram_mcp.pipeline import IndexStats


def _make_project(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    return proj


def test_cli_index_git_analytics_env_precedence(tmp_path, monkeypatch) -> None:
    from engram_mcp import cli, pipeline
    from engram_mcp.embeddings import factory

    proj = _make_project(tmp_path)
    captured: list[bool] = []

    class _FakeProvider:
        model_id = "test:fake-git-analytics-env"
        backend_id = model_id
        dim = 4
        device = "cpu"

        def release_unused_cache(self) -> None:
            pass

    def fake_make_index_provider(index_device=None, *, progress=None):
        return _FakeProvider()

    def fake_release(provider) -> None:
        pass

    def fake_index_project(
        root,
        provider,
        *,
        full_rebuild=False,
        progress=None,
        git_analytics=True,
        git_max_commits=None,
        git_fix_regex=None,
    ):
        captured.append(git_analytics)
        return IndexStats(
            files=1, chunks=1, embedded_unique=1, reused_unique=0,
            seconds=0.001, chunks_per_sec=1.0, mode="full", added=1,
        )

    monkeypatch.setattr(factory, "make_index_provider", fake_make_index_provider)
    monkeypatch.setattr(factory, "release_index_provider", fake_release)
    monkeypatch.setattr(pipeline, "index_project", fake_index_project)

    # ENGRAM_GIT_ANALYTICS=0 -> default off; explicit --git-analytics still turns it on.
    monkeypatch.setenv("ENGRAM_GIT_ANALYTICS", "0")
    assert cli.main(["index", str(proj), "--cpu"]) == 0
    assert captured[-1] is False
    assert cli.main(["index", str(proj), "--cpu", "--git-analytics"]) == 0
    assert captured[-1] is True

    # ENGRAM_GIT_ANALYTICS=1 -> default on; explicit --no-git-analytics still turns it off.
    monkeypatch.setenv("ENGRAM_GIT_ANALYTICS", "1")
    assert cli.main(["index", str(proj), "--cpu"]) == 0
    assert captured[-1] is True
    assert cli.main(["index", str(proj), "--cpu", "--no-git-analytics"]) == 0
    assert captured[-1] is False

    # Unset env var -> built-in default (enabled).
    monkeypatch.delenv("ENGRAM_GIT_ANALYTICS", raising=False)
    assert cli.main(["index", str(proj), "--cpu"]) == 0
    assert captured[-1] is True


class _FakeIndexStdout:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)


class _FakeIndexPopen:
    """Fakes the `engram_mcp.cli index --json` subprocess: records argv and
    always reports a trivial successful index, so a test can drive a real
    index job end to end through start_index_job/_index_worker without a
    real model or a real child process."""

    _RESULT_LINE = (
        '{"event": "result", "version": 1, "ok": true, "mode": "full", "files": 1, '
        '"chunks": 1, "embedded_unique": 1, "reused_unique": 0, '
        '"embedder_id": "fastembed:granite", "backend_id": "fastembed:granite", '
        '"device": "cpu", "seconds": 0.01}\n'
    )

    def __init__(self, cmd, **_kwargs) -> None:
        self.cmd = cmd
        self.stdout = _FakeIndexStdout([self._RESULT_LINE])
        self.stderr = io.StringIO("")
        self.returncode = 0

    def wait(self) -> int:
        return self.returncode


def _run_cpu_job_capture_cmd(server, proj, monkeypatch, **kwargs) -> list[str]:
    """Start a "cpu" index job (index_device forces no auto-routing plan),
    fake the subprocess Popen call it always dispatches to, wait for the job
    to finish, and return the argv the child was launched with."""
    captured: list[list[str]] = []

    def fake_popen(cmd, **_k):
        captured.append(cmd)
        return _FakeIndexPopen(cmd)

    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)
    started = server.start_index_job(str(proj), index_device="cpu", **kwargs)
    job_id = started["job_id"]
    status: dict = {}
    for _ in range(200):
        status = server.get_status(job_id)
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.01)
    assert status["status"] == "done", status
    assert captured, "expected the index subprocess to be launched"
    return captured[-1]


def test_mcp_index_project_git_analytics_env_precedence(tmp_path, monkeypatch) -> None:
    from engram_mcp import server

    proj = _make_project(tmp_path)

    # ENGRAM_GIT_ANALYTICS=0 -> unset git_analytics passes no flag (child
    # resolves the env var itself); explicit True still forces the flag on.
    monkeypatch.setenv("ENGRAM_GIT_ANALYTICS", "0")
    cmd = _run_cpu_job_capture_cmd(server, proj, monkeypatch)
    assert "--git-analytics" not in cmd and "--no-git-analytics" not in cmd
    cmd = _run_cpu_job_capture_cmd(server, proj, monkeypatch, git_analytics=True)
    assert "--git-analytics" in cmd

    # ENGRAM_GIT_ANALYTICS=1 -> still no flag when unset; explicit False forces it off.
    monkeypatch.setenv("ENGRAM_GIT_ANALYTICS", "1")
    cmd = _run_cpu_job_capture_cmd(server, proj, monkeypatch)
    assert "--git-analytics" not in cmd and "--no-git-analytics" not in cmd
    cmd = _run_cpu_job_capture_cmd(server, proj, monkeypatch, git_analytics=False)
    assert "--no-git-analytics" in cmd

    # Unset env var -> still no explicit flag; the child falls back to its
    # own built-in default (enabled), covered by
    # test_cli_index_git_analytics_env_precedence above.
    monkeypatch.delenv("ENGRAM_GIT_ANALYTICS", raising=False)
    cmd = _run_cpu_job_capture_cmd(server, proj, monkeypatch)
    assert "--git-analytics" not in cmd and "--no-git-analytics" not in cmd


def test_mcp_index_project_tool_threads_git_analytics_argument(tmp_path, monkeypatch) -> None:
    """The `index_project` MCP tool coroutine itself (not just start_index_job)
    must thread an explicit `git_analytics` argument through to the index
    subprocess command line as an explicit CLI flag, overriding whatever
    ENGRAM_GIT_ANALYTICS the child would otherwise resolve on its own."""
    from engram_mcp import server

    proj = _make_project(tmp_path)
    monkeypatch.setenv("ENGRAM_GIT_ANALYTICS", "1")

    captured: list[list[str]] = []

    def fake_popen(cmd, **_k):
        captured.append(cmd)
        return _FakeIndexPopen(cmd)

    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)

    async def _call() -> dict:
        return await server.index_project(str(proj), index_device="cpu", git_analytics=False)

    started = asyncio.run(_call())
    job_id = started["job_id"]
    status: dict = {}
    for _ in range(200):
        status = server.get_status(job_id)
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.01)
    assert status["status"] == "done", status
    assert captured
    assert "--no-git-analytics" in captured[-1]
    assert "--git-analytics" not in captured[-1]
