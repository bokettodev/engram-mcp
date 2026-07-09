from __future__ import annotations

from pathlib import Path
from typing import Sequence

from engram_mcp import catalog, gitmeta, manifest, paths, server
from engram_mcp.pipeline import index_project


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


def test_project_map_include_git_attaches_metrics_from_live_log(tmp_path, monkeypatch) -> None:
    root, _provider = _project(tmp_path)
    monkeypatch.setattr(
        gitmeta,
        "commit_log_with_status",
        lambda *_args, **_kwargs: {"status": "ready", "warning": "", "commits": _history_commits()},
    )

    out = server.do_project_map(str(root), include_files=True, include_git=True, group_by="ticket")

    assert out["git_analytics"]["status"] == "ready"
    assert out["git_analytics"]["group_by"] == "ticket"
    assert out["git_analytics"]["scanned_commits"] == 2
    by_path = {row["path"]: row for row in out["files"]}
    assert by_path["alpha.py"]["git"]["changes"] == 1
    assert by_path["alpha.py"]["git"]["churn_lines"] == 6
    assert by_path["alpha.py"]["git"]["fix_density"] == 1.0
    assert by_path["alpha.py"]["git"]["cochanges"][0]["path"] == "beta.py"
    assert "lift" in by_path["alpha.py"]["git"]["cochanges"][0]


def test_project_map_prefers_cached_git_history_over_live_walk(tmp_path, monkeypatch) -> None:
    root, _provider = _project(tmp_path)
    pdir = paths.project_dir(root, create=False)
    m = manifest.load_project(pdir)
    data = catalog.load_catalog(pdir, m.generation)
    assert data is not None
    data["git_history"] = {
        "schema_version": 1,
        "status": "ready",
        "max_commits": 1000,
        "head_commit": "c2",
        "commits": _history_commits(),
    }
    catalog.save_catalog(pdir, data)

    def boom(*_args, **_kwargs):
        raise AssertionError("live git log should not be called when sidecar history is present")

    monkeypatch.setattr(gitmeta, "commit_log_with_status", boom)
    out = server.do_project_map(str(root), include_files=True, include_git=True)

    assert out["git_analytics"]["status"] == "ready"
    assert out["git_analytics"]["cached_commits"] == 2


def test_index_time_git_history_is_flag_gated(tmp_path, monkeypatch) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "alpha.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    provider = _FakeProvider()

    index_project(root, provider, full_rebuild=True)
    pdir = paths.project_dir(root, create=False)
    m = manifest.load_project(pdir)
    data = catalog.load_catalog(pdir, m.generation)
    assert data is not None
    assert "git_history" not in data

    monkeypatch.setattr(
        gitmeta,
        "history_for_catalog",
        lambda *_args, **_kwargs: {
            "schema_version": 1,
            "status": "ready",
            "max_commits": 7,
            "head_commit": "c1",
            "commits": _history_commits()[:1],
        },
    )
    index_project(root, provider, full_rebuild=True, git_analytics=True, git_max_commits=7)
    m = manifest.load_project(pdir)
    data = catalog.load_catalog(pdir, m.generation)

    assert data is not None
    assert data["git_history"] == {
        "schema_version": 1,
        "status": "ready",
        "max_commits": 7,
        "head_commit": "c1",
        "commits": _history_commits()[:1],
    }
