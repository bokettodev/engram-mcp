from __future__ import annotations

import threading
from pathlib import Path
from typing import Sequence

from engram_mcp import catalog, gitmeta, manifest, paths, server
from engram_mcp.pipeline import index_project, search_project, wait_for_szz_tasks


class _FakeProvider:
    model_id = "test:fake-git"
    backend_id = model_id
    dim = 4

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float("alpha" in text.lower()), float("beta" in text.lower()), 0.0, 1.0] for text in texts]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_passages(texts)

    def release_unused_cache(self) -> None:
        pass


def _project(tmp_path: Path) -> tuple[Path, _FakeProvider]:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "alpha.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    (root / "beta.py").write_text("def beta():\n    return 2\n", encoding="utf-8")
    provider = _FakeProvider()
    index_project(root, provider, full_rebuild=True)
    return root, provider


def _history_commits() -> list[dict]:
    return [
        {
            "commit": "c2",
            "parents": ["c1"],
            "ts": 200,
            "author": "Ann",
            "message": "fix ABC-1 alpha beta",
            "paths": [
                {"path": "alpha.py", "status": "M", "added": 2, "deleted": 1},
                {"path": "beta.py", "status": "M", "added": 1, "deleted": 0},
            ],
        },
        {
            "commit": "c1",
            "parents": [],
            "ts": 100,
            "author": "Bob",
            "message": "ABC-1 alpha",
            "paths": [{"path": "alpha.py", "status": "A", "added": 3, "deleted": 0}],
        },
    ]


def _ready_history(commits: list[dict] | None = None, *, head: str = "c2", max_commits=None) -> dict:
    return {
        "schema_version": 1,
        "status": "ready",
        "max_commits": max_commits,
        "head_commit": head,
        "fix_regex": gitmeta.DEFAULT_FIX_REGEX,
        "commits": commits if commits is not None else _history_commits(),
        "szz": {
            "status": "ready",
            "warning": "",
            "fix_regex": gitmeta.DEFAULT_FIX_REGEX,
            "fix_commits": 0,
            "blamed_lines": 0,
            "attributions": [],
            "warnings": [],
        },
    }


def test_project_map_include_git_false_does_not_walk_git(tmp_path, monkeypatch) -> None:
    root, _provider = _project(tmp_path)

    def boom(*_args, **_kwargs):
        raise AssertionError("git log should not be called")

    monkeypatch.setattr(gitmeta, "commit_log_with_status", boom)
    out = server.do_project_map(str(root), include_files=True, include_git=False)

    assert "git_analytics" not in out
    assert all("git" not in row for row in out["files"])


def test_project_map_include_git_disabled_by_staleness_env(tmp_path, monkeypatch) -> None:
    root, _provider = _project(tmp_path)
    monkeypatch.setenv("ENGRAM_GIT_STALENESS", "0")

    out = server.do_project_map(str(root), include_files=True, include_git=True)

    assert out["totals"]["files"] == 2
    assert out["git_analytics"]["status"] == "unavailable"
    assert out["git_analytics"]["available"] is False


def test_project_map_include_git_uncached_builds_live_in_memory(tmp_path, monkeypatch) -> None:
    root, _provider = _project(tmp_path)
    pdir = paths.project_dir(root, create=False)
    m = manifest.load_project(pdir)
    data = catalog.load_catalog(pdir, m.generation)
    assert data is not None
    data.pop("git_history", None)
    catalog.save_catalog(pdir, data)
    monkeypatch.setattr(gitmeta, "head_commit", lambda *_args, **_kwargs: "c2")
    monkeypatch.setattr(
        gitmeta,
        "commit_log_with_status",
        lambda *_args, **_kwargs: {"status": "ready", "warning": "", "commits": _history_commits()},
    )

    out = server.do_project_map(str(root), include_files=True, include_git=True, group_by="ticket")

    assert out["git_analytics"]["status"] == "uncached"
    assert out["git_analytics"]["szz"]["status"] == "computing"
    assert out["git_analytics"]["group_by"] == "ticket"
    assert out["git_analytics"]["scanned_commits"] == 2
    assert out["git_analytics"]["current_head"] == "c2"
    by_path = {row["path"]: row for row in out["files"]}
    assert by_path["alpha.py"]["git"]["changes"] == 1
    assert by_path["alpha.py"]["git"]["churn_lines"] == 6
    assert by_path["alpha.py"]["git"]["fix_density"] == 1.0
    assert "defect_hotspot_score" not in by_path["alpha.py"]["git"]
    assert by_path["alpha.py"]["git"]["cochanges"][0]["path"] == "beta.py"
    assert "lift" in by_path["alpha.py"]["git"]["cochanges"][0]


def test_project_map_prefers_cached_git_history_over_live_walk(tmp_path, monkeypatch) -> None:
    root, _provider = _project(tmp_path)
    pdir = paths.project_dir(root, create=False)
    m = manifest.load_project(pdir)
    data = catalog.load_catalog(pdir, m.generation)
    assert data is not None
    data["git_history"] = _ready_history()
    catalog.save_catalog(pdir, data)

    def boom(*_args, **_kwargs):
        raise AssertionError("live git log should not be called when sidecar history is present")

    monkeypatch.setattr(gitmeta, "head_commit", lambda *_args, **_kwargs: "c2")
    monkeypatch.setattr(gitmeta, "commit_log_with_status", boom)
    out = server.do_project_map(str(root), include_files=True)

    assert out["git_analytics"]["status"] == "ready"
    assert out["git_analytics"]["cached_commits"] == 2


def test_project_map_freshens_cache_in_memory_without_writing_sidecar(tmp_path, monkeypatch) -> None:
    root, _provider = _project(tmp_path)
    pdir = paths.project_dir(root, create=False)
    m = manifest.load_project(pdir)
    data = catalog.load_catalog(pdir, m.generation)
    assert data is not None
    data["git_history"] = _ready_history(_history_commits()[1:], head="c1")
    catalog.save_catalog(pdir, data)
    sidecar = catalog.catalog_path(pdir, m.generation)
    before_text = sidecar.read_text(encoding="utf-8")
    before_mtime = sidecar.stat().st_mtime_ns

    monkeypatch.setattr(gitmeta, "head_commit", lambda *_args, **_kwargs: "c2")
    monkeypatch.setattr(gitmeta, "is_ancestor", lambda *_args, **_kwargs: True)

    def delta_log(*_args, **kwargs):
        assert kwargs.get("rev_range") == "c1..c2"
        return {"status": "ready", "warning": "", "commits": _history_commits()[:1]}

    monkeypatch.setattr(gitmeta, "commit_log_with_status", delta_log)
    out = server.do_project_map(str(root), include_files=True)

    assert out["git_analytics"]["status"] == "freshened"
    assert out["git_analytics"]["szz"]["status"] == "computing"
    assert out["git_analytics"]["cache_head"] == "c1"
    assert out["git_analytics"]["current_head"] == "c2"
    assert out["git_analytics"]["freshened_commits"] == 1
    assert out["git_analytics"]["cached_commits"] == 1
    assert out["git_analytics"]["scanned_commits"] == 1
    assert out["git_analytics"]["analyzed_changes"] == 2
    assert sidecar.read_text(encoding="utf-8") == before_text
    assert sidecar.stat().st_mtime_ns == before_mtime


def test_index_time_git_history_is_default_on_and_can_be_disabled(tmp_path, monkeypatch) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "alpha.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    provider = _FakeProvider()

    monkeypatch.setattr(
        gitmeta,
        "history_for_catalog",
        lambda *_args, **_kwargs: {
            "schema_version": 1,
            "status": "ready",
            "max_commits": 7,
            "head_commit": "c1",
            "fix_regex": gitmeta.DEFAULT_FIX_REGEX,
            "commits": _history_commits()[:1],
        },
    )
    index_project(root, provider, full_rebuild=True, git_max_commits=7)
    wait_for_szz_tasks(timeout=5)
    pdir = paths.project_dir(root, create=False)
    m = manifest.load_project(pdir)
    data = catalog.load_catalog(pdir, m.generation)
    assert data is not None
    assert data["git_history"] == {
        "schema_version": 1,
        "status": "ready",
        "max_commits": 7,
        "head_commit": "c1",
        "fix_regex": gitmeta.DEFAULT_FIX_REGEX,
        "commits": _history_commits()[:1],
    }

    disabled = tmp_path / "disabled"
    disabled.mkdir()
    (disabled / "alpha.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    index_project(disabled, provider, full_rebuild=True, git_analytics=False)
    pdir = paths.project_dir(disabled, create=False)
    m = manifest.load_project(pdir)
    data = catalog.load_catalog(pdir, m.generation)
    assert data is not None
    assert "git_history" not in data


def test_szz_background_sidecar_does_not_block_searchable_generation(tmp_path, monkeypatch) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "alpha.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    provider = _FakeProvider()
    started = threading.Event()
    release = threading.Event()

    monkeypatch.setattr(gitmeta, "head_commit", lambda *_args, **_kwargs: "c2")
    monkeypatch.setattr(
        gitmeta,
        "history_for_catalog",
        lambda *_args, **_kwargs: {
            "schema_version": 1,
            "status": "ready",
            "max_commits": None,
            "head_commit": "c2",
            "fix_regex": gitmeta.DEFAULT_FIX_REGEX,
            "commits": _history_commits(),
        },
    )

    def slow_szz(_root, _commits, **_kwargs):
        started.set()
        assert release.wait(5)
        return {
            "status": "ready",
            "warning": "",
            "fix_regex": gitmeta.DEFAULT_FIX_REGEX,
            "fix_commits": 1,
            "blamed_lines": 1,
            "attributions": [
                {
                    "fix_commit": "c2",
                    "introducing_commit": "c1",
                    "path": "alpha.py",
                    "lines": 1,
                }
            ],
            "commit_attributions": {
                "c2": {
                    "fix_commit": "c2",
                    "status": "ready",
                    "attributions": [
                        {
                            "fix_commit": "c2",
                            "introducing_commit": "c1",
                            "path": "alpha.py",
                            "lines": 1,
                        }
                    ],
                    "blamed_lines": 1,
                    "warnings": [],
                }
            },
            "workers": 1,
            "cached_commits": 0,
            "blamed_commits": 1,
            "timings_ms": {"total": 1.0, "blame": 1.0},
            "warnings": [],
        }

    monkeypatch.setattr(gitmeta, "szz_attributions_with_status", slow_szz)

    index_project(root, provider, full_rebuild=True)
    assert started.wait(5)

    hits = search_project(root, provider, "alpha", k=1)
    assert hits
    before = server.do_project_map(str(root), include_files=True, include_git=True)
    assert before["git_analytics"]["szz"]["status"] == "computing"
    assert "defect_hotspot_score" not in before["files"][0]["git"]

    release.set()
    wait_for_szz_tasks(timeout=5)

    after = server.do_project_map(str(root), include_files=True, include_git=True)
    assert after["git_analytics"]["szz"]["status"] == "ready"
    assert after["files"][0]["git"]["defect_hotspot_score"] > 0
