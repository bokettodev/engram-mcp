from __future__ import annotations

import os
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Sequence

from engram_mcp import (
    catalog,
    gitmeta,
    gitstore,
    gitorchestration,
    manifest,
    paths,
    pipeline,
    regexsafe,
    server,
)
from engram_mcp.pipeline import index_project, search_project


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


def _crash_regex_worker(*_args) -> None:
    os._exit(1)


def _crash_grep_worker(*_args) -> None:
    os._exit(1)


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


def _fingerprint(*, hash_value: str = "fp-c2", tip: str = "c2") -> dict:
    return {
        "status": "ready",
        "algorithm": "git-refs-sha256-v1",
        "hash": hash_value,
        "refs": 1,
        "max_commit_ts": 200 if tip == "c2" else 100,
        "tip_commit": tip,
        "common_dir": "T:/fake/.git",
    }


def _ready_history(
    commits: list[dict] | None = None,
    *,
    head: str = "c2",
    fingerprint_hash: str = "fp-c2",
    max_commits: int | None = None,
) -> dict:
    fp = _fingerprint(hash_value=fingerprint_hash, tip=head)
    return {
        "schema_version": 1,
        "logical_project_id": "unused",
        "checkout_kind": "main",
        "status": "ready",
        "max_commits": max_commits,
        "head_commit": head,
        "git_common_dir": fp["common_dir"],
        "fingerprint": {
            "algorithm": fp["algorithm"],
            "hash": fp["hash"],
            "refs": fp["refs"],
            "max_commit_ts": fp["max_commit_ts"],
            "tip_commit": fp["tip_commit"],
        },
        "fix_regex": gitmeta.DEFAULT_FIX_REGEX,
        "commits": commits if commits is not None else _history_commits(),
    }


def _save_history(root: Path, history: dict) -> str:
    pdir = paths.project_dir(root, create=False)
    m = manifest.load_project(pdir)
    assert m is not None
    payload = dict(history)
    payload["logical_project_id"] = m.logical_project_id
    gitstore.save_history(m.logical_project_id, payload)
    return m.logical_project_id


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
    assert m is not None
    history_file = gitstore.history_path(m.logical_project_id, create=False)
    if history_file.exists():
        history_file.unlink()
    monkeypatch.setattr(gitmeta, "repo_ref_fingerprint", lambda *_args, **_kwargs: _fingerprint())
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


def test_project_map_git_analytics_limits_are_clamped_with_warning(tmp_path, monkeypatch) -> None:
    """Item 3 of the search-hot-path audit: max_files_per_change, cochange_limit,
    hotspots_limit, and git_max_commits are every-caller-supplied limits an
    analytics request can raise arbitrarily high -- max_files_per_change in
    particular gates gitanalytics.cochange's O(n^2) association-rule
    generation over one change set's file list, so leaving it unbounded is a
    real quadratic-blowup vector, not just an output-size question. All four
    must clamp to their server maximum and report the clamp in
    git_analytics.warnings rather than either erroring or silently running
    unbounded.
    """
    root, _provider = _project(tmp_path)
    _save_history(root, _ready_history())
    monkeypatch.setattr(gitmeta, "repo_ref_fingerprint", lambda *_args, **_kwargs: _fingerprint())

    out = server.do_project_map(
        str(root),
        include_files=True,
        include_git=True,
        max_files_per_change=pipeline.MAX_GIT_MAX_FILES_PER_CHANGE + 100,
        cochange_limit=pipeline.MAX_GIT_COCHANGE_LIMIT + 100,
        hotspots_limit=pipeline.MAX_GIT_HOTSPOTS_LIMIT + 100,
        git_max_commits=gitorchestration.MAX_GIT_MAX_COMMITS + 100,
    )

    assert out["git_analytics"]["available"] is True
    warnings = out["git_analytics"]["warnings"]
    assert any("max_files_per_change clamped" in w for w in warnings)
    assert any("cochange_limit clamped" in w for w in warnings)
    assert any("hotspots_limit clamped" in w for w in warnings)
    assert any("git_max_commits clamped" in w for w in warnings)

    # A within-budget request is untouched and carries no clamp warning.
    within_budget = server.do_project_map(
        str(root),
        include_files=True,
        include_git=True,
        max_files_per_change=10,
        cochange_limit=5,
        hotspots_limit=5,
    )
    assert within_budget["git_analytics"]["warnings"] == []


def test_project_map_live_history_uses_short_git_timeout(tmp_path, monkeypatch) -> None:
    # Analytics enabled (the default) but no shared history cache yet, so
    # project_map must fall through to a live, short-timeout walk. A project
    # indexed with analytics *disabled* is covered separately -- it must
    # report `status == "disabled"` instead of ever reaching this live path.
    root, _provider = _project(tmp_path)
    pdir = paths.project_dir(root, create=False)
    m = manifest.load_project(pdir)
    assert m is not None
    history_file = gitstore.history_path(m.logical_project_id, create=False)
    if history_file.exists():
        history_file.unlink()
    seen: list[float | None] = []
    monkeypatch.setattr(gitmeta, "repo_ref_fingerprint", lambda *_args, **_kwargs: _fingerprint())

    def fake_log(*_args, **kwargs):
        seen.append(kwargs.get("timeout_sec"))
        return {"status": "ready", "warning": "", "commits": _history_commits()}

    monkeypatch.setattr(gitmeta, "commit_log_with_status", fake_log)

    out = server.do_project_map(str(root), include_files=True, include_git=True)

    assert out["git_analytics"]["status"] == "uncached"
    assert seen == [gitmeta._GIT_STALENESS_TIMEOUT_SEC]


def test_project_map_prefers_cached_git_history_over_live_walk(tmp_path, monkeypatch) -> None:
    root, _provider = _project(tmp_path)
    _save_history(root, _ready_history())

    def boom(*_args, **_kwargs):
        raise AssertionError("live git log should not be called when sidecar history is present")

    monkeypatch.setattr(gitmeta, "repo_ref_fingerprint", lambda *_args, **_kwargs: _fingerprint())
    monkeypatch.setattr(gitmeta, "commit_log_with_status", boom)
    out = server.do_project_map(str(root), include_files=True)

    assert out["git_analytics"]["status"] == "ready"
    assert out["git_analytics"]["cached_commits"] == 2


def test_project_map_freshens_cache_in_memory_without_writing_sidecar(tmp_path, monkeypatch) -> None:
    root, _provider = _project(tmp_path)
    logical_project_id = _save_history(
        root,
        _ready_history(_history_commits()[1:], head="c1", fingerprint_hash="fp-c1"),
    )
    sidecar = gitstore.history_path(logical_project_id, create=False)
    before_text = sidecar.read_text(encoding="utf-8")
    before_mtime = sidecar.stat().st_mtime_ns

    monkeypatch.setattr(gitmeta, "repo_ref_fingerprint", lambda *_args, **_kwargs: _fingerprint())

    def delta_log(*_args, **kwargs):
        assert kwargs.get("all_refs") is True
        assert kwargs.get("git_dir") is True
        assert kwargs.get("timeout_sec") == gitmeta._GIT_STALENESS_TIMEOUT_SEC
        return {"status": "ready", "warning": "", "commits": _history_commits()}

    monkeypatch.setattr(gitmeta, "commit_log_with_status", delta_log)
    out = server.do_project_map(str(root), include_files=True)

    assert out["git_analytics"]["status"] == "freshened"
    assert out["git_analytics"]["szz"]["status"] == "computing"
    assert out["git_analytics"]["cache_head"] == "c1"
    assert out["git_analytics"]["current_head"] == "c2"
    assert out["git_analytics"]["freshened_commits"] == 1
    assert out["git_analytics"]["cached_commits"] == 1
    assert out["git_analytics"]["scanned_commits"] == 2
    assert out["git_analytics"]["analyzed_changes"] == 2
    assert sidecar.read_text(encoding="utf-8") == before_text
    assert sidecar.stat().st_mtime_ns == before_mtime


def test_project_map_uses_stale_cached_history_on_short_timeout(tmp_path, monkeypatch) -> None:
    root, _provider = _project(tmp_path)
    _save_history(root, _ready_history(_history_commits()[1:], head="c1", fingerprint_hash="fp-c1"))
    monkeypatch.setattr(gitmeta, "repo_ref_fingerprint", lambda *_args, **_kwargs: _fingerprint())

    def timeout_log(*_args, **kwargs):
        assert kwargs.get("timeout_sec") == gitmeta._GIT_STALENESS_TIMEOUT_SEC
        return {"status": "unavailable", "warning": "git log unavailable", "commits": []}

    monkeypatch.setattr(gitmeta, "commit_log_with_status", timeout_log)

    out = server.do_project_map(str(root), include_files=True)

    assert out["git_analytics"]["status"] == "stale"
    assert out["git_analytics"]["cached_commits"] == 1
    assert out["git_analytics"]["scanned_commits"] == 0
    assert out["git_analytics"]["analyzed_changes"] == 1


def test_index_time_git_history_is_default_on_and_can_be_disabled(tmp_path, monkeypatch) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "alpha.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    provider = _FakeProvider()

    monkeypatch.setattr(
        gitmeta,
        "repo_ref_fingerprint",
        lambda *_args, **_kwargs: _fingerprint(hash_value="fp-c1", tip="c1"),
    )
    monkeypatch.setattr(
        gitmeta,
        "commit_log_with_status",
        lambda *_args, **_kwargs: {
            "status": "ready",
            "warning": "",
            "commits": _history_commits()[:1],
        },
    )
    index_project(root, provider, full_rebuild=True, git_max_commits=7)
    gitorchestration.wait_for_szz_tasks(timeout=5)
    pdir = paths.project_dir(root, create=False)
    m = manifest.load_project(pdir)
    data = catalog.load_catalog(pdir, m.generation)
    assert data is not None
    assert "git_history" not in data
    history = gitstore.load_history(m.logical_project_id)
    assert history is not None
    assert history["status"] == "ready"
    assert history["max_commits"] == 7
    assert history["head_commit"] == "c1"
    assert history["fingerprint"]["hash"] == "fp-c1"
    assert history["commits"] == _history_commits()[:1]


def test_git_max_commits_limits_shared_history(tmp_path, monkeypatch) -> None:
    root, _provider = _project(tmp_path)
    pdir = paths.project_dir(root, create=False)
    m = manifest.load_project(pdir)
    assert m is not None
    history_file = gitstore.history_path(m.logical_project_id, create=False)
    if history_file.exists():
        history_file.unlink()

    monkeypatch.setattr(gitmeta, "repo_ref_fingerprint", lambda *_args, **_kwargs: _fingerprint())

    def fake_log(*_args, **kwargs):
        assert kwargs.get("max_commits") == 1
        return {"status": "ready", "warning": "", "commits": _history_commits()[:1]}

    monkeypatch.setattr(gitmeta, "commit_log_with_status", fake_log)
    out = server.do_project_map(str(root), include_files=True, include_git=True, git_max_commits=1)

    assert out["git_analytics"]["scanned_commits"] == 1
    assert out["git_analytics"]["analyzed_changes"] == 1

    disabled = tmp_path / "disabled"
    disabled.mkdir()
    (disabled / "alpha.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    index_project(disabled, _provider, full_rebuild=True, git_analytics=False)
    pdir = paths.project_dir(disabled, create=False)
    m = manifest.load_project(pdir)
    data = catalog.load_catalog(pdir, m.generation)
    assert data is not None
    assert "git_history" not in data
    assert gitstore.load_history(m.logical_project_id) is None


def test_szz_ready_requires_matching_max_commits(tmp_path) -> None:
    source = _ready_history(max_commits=None)
    commits = _history_commits()
    szz = {
        "schema_version": gitstore.SCHEMA_VERSION,
        "logical_project_id": "logical",
        "status": "ready",
        "fix_regex": gitmeta.DEFAULT_FIX_REGEX,
        "max_commits": 1,
        "fingerprint": source["fingerprint"],
        "head_commit": "c2",
        "git_common_dir": _fingerprint()["common_dir"],
        "fix_commits": 1,
        "blamed_lines": 0,
        "attributions": [],
        "commit_attributions": {},
        "workers": 1,
        "cached_commits": 1,
        "blamed_commits": 1,
        "timings_ms": {"total": 1.0, "blame": 0.0},
        "warnings": [],
    }
    gitstore.save_szz("logical", szz)

    resolved = gitorchestration.szz_for_source(
        logical_project_id="logical",
        source=source,
        history=None,
        commits=commits,
    )

    assert resolved["status"] == "computing"
    assert resolved["max_commits"] is None


def test_szz_write_guard_keeps_different_fix_regex_results_separate(tmp_path) -> None:
    logical_project_id = paths.project_id_for(tmp_path)
    history_default = _ready_history()
    history_custom = _ready_history()
    history_default["logical_project_id"] = logical_project_id
    history_custom["logical_project_id"] = logical_project_id
    history_custom["fix_regex"] = r"(?i)\bresolve\b"
    gitstore.save_history(logical_project_id, history_custom)

    stale_payload = {
        "status": "ready",
        "warning": "",
        "fix_regex": gitmeta.DEFAULT_FIX_REGEX,
        "fix_commits": 1,
        "blamed_lines": 0,
        "attributions": [],
        "commit_attributions": {},
        "workers": 1,
        "cached_commits": 0,
        "blamed_commits": 1,
        "timings_ms": {"total": 1.0, "blame": 0.0},
        "warnings": [],
    }
    current_payload = dict(stale_payload)
    current_payload["fix_regex"] = history_custom["fix_regex"]

    stale_written = gitorchestration._write_shared_szz_if_current(
        logical_project_id=logical_project_id,
        expected_history=history_default,
        payload=stale_payload,
    )
    current_written = gitorchestration._write_shared_szz_if_current(
        logical_project_id=logical_project_id,
        expected_history=history_custom,
        payload=current_payload,
    )

    szz = gitstore.load_szz(logical_project_id)
    assert stale_written is False
    assert current_written is True
    assert szz is not None
    assert szz["fix_regex"] == history_custom["fix_regex"]


def test_szz_write_guard_rejects_payload_regex_mismatched_with_history(tmp_path) -> None:
    """A payload whose own fix_regex disagrees with the history it was
    computed against must never be written, even when the on-disk history is
    otherwise still current — writing it would desync the persisted identity
    (history says one regex, sidecar says another) and wedge SZZ reuse.
    """

    logical_project_id = paths.project_id_for(tmp_path)
    history = _ready_history()
    history["logical_project_id"] = logical_project_id
    gitstore.save_history(logical_project_id, history)

    mismatched_payload = {
        "status": "ready",
        "warning": "",
        "fix_regex": r"(?i)\bresolve\b",  # disagrees with history["fix_regex"]
        "fix_commits": 1,
        "blamed_lines": 0,
        "attributions": [],
        "commit_attributions": {},
        "workers": 1,
        "cached_commits": 0,
        "blamed_commits": 1,
        "timings_ms": {"total": 1.0, "blame": 0.0},
        "warnings": [],
    }

    written = gitorchestration._write_shared_szz_if_current(
        logical_project_id=logical_project_id,
        expected_history=history,
        payload=mismatched_payload,
    )

    assert written is False
    assert gitstore.load_szz(logical_project_id) is None


def test_ensure_shared_git_analytics_recomputes_when_history_schema_is_legacy(tmp_path, monkeypatch) -> None:
    """gitstore.SCHEMA_VERSION was bumped 1 -> 2 alongside requested-regex
    identity; a history file written by the previous schema on disk must be
    treated as stale and recomputed, never reused as if it were current.
    """

    import json

    root = tmp_path / "proj"
    root.mkdir()
    logical_project_id = paths.project_id_for(root)
    monkeypatch.setattr(gitmeta, "repo_ref_fingerprint", lambda *_args, **_kwargs: _fingerprint())
    calls = 0

    def counting_log(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return {"status": "ready", "warning": "", "commits": _history_commits()}

    monkeypatch.setattr(gitmeta, "commit_log_with_status", counting_log)
    monkeypatch.setattr(
        gitmeta,
        "szz_attributions_with_status",
        lambda *_a, **_k: {
            "status": "ready",
            "warning": "",
            "fix_regex": gitmeta.DEFAULT_FIX_REGEX,
            "fix_commits": 0,
            "blamed_lines": 0,
            "attributions": [],
            "commit_attributions": {},
            "workers": 1,
            "cached_commits": 0,
            "blamed_commits": 0,
            "timings_ms": {"total": 1.0, "blame": 0.0},
            "warnings": [],
        },
    )

    # Write a raw legacy (schema_version=1) history file directly to disk,
    # bypassing gitstore.save_history (which would force the current schema).
    assert gitstore.SCHEMA_VERSION == 2
    history_path = gitstore.history_path(logical_project_id, create=True)
    history_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "logical_project_id": logical_project_id,
                "checkout_kind": "main",
                "status": "ready",
                "max_commits": None,
                "head_commit": "c2",
                "git_common_dir": _fingerprint()["common_dir"],
                "fingerprint": _fingerprint(),
                "fix_regex": gitmeta.DEFAULT_FIX_REGEX,
                "commits": _history_commits(),
            }
        ),
        encoding="utf-8",
    )

    gitorchestration.ensure_shared_git_analytics(
        root=root,
        logical_project_id=logical_project_id,
        checkout_kind="main",
        enabled=True,
        fix_regex=None,
    )
    gitorchestration.wait_for_szz_tasks(timeout=5)

    assert calls == 1  # legacy on-disk history was not reused; git was re-walked
    refreshed = gitstore.load_history(logical_project_id)
    assert refreshed is not None
    assert refreshed["schema_version"] == gitstore.SCHEMA_VERSION


def test_szz_task_key_includes_fix_regex_and_max_commits() -> None:
    fp = _ready_history()["fingerprint"]

    default_key = gitorchestration._szz_task_key("logical", fp, gitmeta.DEFAULT_FIX_REGEX, None)
    limited_key = gitorchestration._szz_task_key("logical", fp, gitmeta.DEFAULT_FIX_REGEX, 1)
    custom_key = gitorchestration._szz_task_key("logical", fp, r"(?i)\bresolve\b", None)

    assert default_key != limited_key
    assert default_key != custom_key


def test_start_shared_szz_task_marks_thread_alive_before_releasing_lock(monkeypatch, tmp_path) -> None:
    history = _ready_history()
    started_while_locked: list[bool] = []
    gitorchestration._SZZ_TASKS.clear()

    class FakeThread:
        def __init__(self, *args, **kwargs):
            self._alive = False

        def is_alive(self):
            return self._alive

        def start(self):
            started_while_locked.append(gitorchestration._SZZ_TASKS_LOCK.locked())
            self._alive = True

    monkeypatch.setattr(gitorchestration.threading, "Thread", FakeThread)
    try:
        gitorchestration._start_shared_szz_task(
            root=tmp_path,
            logical_project_id="logical",
            git_history=history,
            previous_szz=None,
        )
    finally:
        gitorchestration._SZZ_TASKS.clear()

    assert started_while_locked == [True]


def test_incremental_preserves_custom_git_fix_regex_when_flag_omitted(tmp_path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "alpha.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    provider = _FakeProvider()
    custom = r"(?i)\bresolve\b"

    index_project(root, provider, full_rebuild=True, git_analytics=False, git_fix_regex=custom)
    (root / "alpha.py").write_text("def alpha():\n    return 2\n", encoding="utf-8")
    index_project(root, provider, git_analytics=False)

    m = manifest.load_project(paths.project_dir(root, create=False))
    assert m is not None
    assert m.git_fix_regex == custom


def test_reindex_file_preserves_custom_git_fix_regex(tmp_path) -> None:
    from engram_mcp.pipeline import reindex_file

    root = tmp_path / "proj"
    root.mkdir()
    (root / "alpha.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    provider = _FakeProvider()
    custom = r"(?i)\bresolve\b"

    index_project(root, provider, full_rebuild=True, git_analytics=False, git_fix_regex=custom)
    (root / "alpha.py").write_text("def alpha():\n    return 2\n", encoding="utf-8")
    reindex_file(root, provider, "alpha.py")

    m = manifest.load_project(paths.project_dir(root, create=False))
    assert m is not None
    assert m.git_fix_regex == custom


def test_full_rebuild_persists_sanitized_git_fix_regex(tmp_path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "alpha.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    provider = _FakeProvider()

    index_project(root, provider, full_rebuild=True, git_analytics=False, git_fix_regex="[")

    m = manifest.load_project(paths.project_dir(root, create=False))
    assert m is not None
    assert m.git_fix_regex == gitmeta.DEFAULT_FIX_REGEX


def test_unsafe_git_fix_regex_persists_effective_identity_and_reuses_szz(tmp_path, monkeypatch) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "alpha.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    provider = _FakeProvider()
    unsafe = r"(b+)+$"
    szz_calls = 0
    worker_patterns: list[str] = []

    monkeypatch.setattr(gitmeta, "repo_ref_fingerprint", lambda *_args, **_kwargs: _fingerprint())
    monkeypatch.setattr(
        gitmeta,
        "commit_log_with_status",
        lambda *_args, **_kwargs: {
            "status": "ready",
            "warning": "",
            "commits": [
                {
                    **_history_commits()[0],
                    "message": "fix bug " + ("b" * 128) + "!",
                },
                _history_commits()[1],
            ],
        },
    )

    def fake_run_worker(_target, *, pattern, flags, texts, timeout_sec):
        del flags, texts, timeout_sec
        worker_patterns.append(pattern)
        return False, None, "simulated timeout"

    def fake_szz(_root, _commits, **kwargs):
        nonlocal szz_calls
        szz_calls += 1
        assert kwargs.get("fix_regex") == gitmeta.DEFAULT_FIX_REGEX
        return {
            "status": "ready",
            "warning": "",
            "fix_regex": gitmeta.DEFAULT_FIX_REGEX,
            "fix_commits": 1,
            "blamed_lines": 0,
            "attributions": [],
            "commit_attributions": {},
            "workers": 1,
            "cached_commits": 0,
            "blamed_commits": 1,
            "timings_ms": {"total": 1.0, "blame": 0.0},
            "warnings": [],
        }

    monkeypatch.setattr(regexsafe, "_run_worker", fake_run_worker)
    monkeypatch.setattr(gitmeta, "szz_attributions_with_status", fake_szz)

    index_project(root, provider, full_rebuild=True, git_fix_regex=unsafe)
    gitorchestration.wait_for_szz_tasks(timeout=5)

    pdir = paths.project_dir(root, create=False)
    m = manifest.load_project(pdir)
    assert m is not None
    history = gitstore.load_history(m.logical_project_id)
    szz = gitstore.load_szz(m.logical_project_id)
    assert history is not None
    assert szz is not None
    assert m.git_fix_regex == gitmeta.DEFAULT_FIX_REGEX
    assert m.requested_git_fix_regex == unsafe
    assert history["fix_regex"] == gitmeta.DEFAULT_FIX_REGEX
    assert history["requested_fix_regex"] == unsafe
    assert any(unsafe in warning for warning in history["warnings"])
    assert szz["status"] == "ready"
    assert szz["fix_regex"] == gitmeta.DEFAULT_FIX_REGEX
    assert szz["requested_fix_regex"] == unsafe
    assert worker_patterns == [unsafe]
    assert szz_calls == 1

    worker_patterns.clear()
    index_project(root, provider, full_rebuild=True, git_fix_regex=unsafe)
    gitorchestration.wait_for_szz_tasks(timeout=5)
    assert worker_patterns == []
    assert szz_calls == 1

    first = server.do_project_map(str(root), include_files=True, include_git=True)
    second = server.do_project_map(str(root), include_files=True, include_git=True)
    assert first["git_analytics"]["szz"]["status"] == "ready"
    assert second["git_analytics"]["szz"]["status"] == "ready"
    assert worker_patterns == []


def test_reindex_file_preserves_no_git_analytics_policy(tmp_path, monkeypatch) -> None:
    from engram_mcp.pipeline import reindex_file

    root = tmp_path / "proj"
    root.mkdir()
    (root / "alpha.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    provider = _FakeProvider()
    monkeypatch.setattr(gitmeta, "repo_ref_fingerprint", lambda *_args, **_kwargs: _fingerprint())
    monkeypatch.setattr(
        gitmeta,
        "commit_log_with_status",
        lambda *_args, **_kwargs: {"status": "ready", "warning": "", "commits": _history_commits()},
    )

    index_project(root, provider, full_rebuild=True, git_analytics=False)
    pdir = paths.project_dir(root, create=False)
    m = manifest.load_project(pdir)
    assert m is not None
    (root / "alpha.py").write_text("def alpha():\n    return 2\n", encoding="utf-8")
    reindex_file(root, provider, "alpha.py")

    assert gitstore.load_history(m.logical_project_id) is None
    assert gitstore.load_szz(m.logical_project_id) is None


def test_full_rebuild_drops_superseded_catalog_sidecars(tmp_path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "alpha.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    provider = _FakeProvider()

    index_project(root, provider, full_rebuild=True, git_analytics=False)
    pdir = paths.project_dir(root, create=False)
    m1 = manifest.load_project(pdir)
    assert m1 is not None
    c1 = catalog.catalog_path(pdir, m1.generation)
    assert c1.exists()

    index_project(root, provider, full_rebuild=True, git_analytics=False)
    m2 = manifest.load_project(pdir)
    assert m2 is not None
    assert c1.exists()

    index_project(root, provider, full_rebuild=True, git_analytics=False)
    assert not c1.exists()
    assert catalog.catalog_path(pdir, m2.generation).exists()


def test_szz_background_sidecar_does_not_block_searchable_generation(tmp_path, monkeypatch) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "alpha.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    provider = _FakeProvider()
    started = threading.Event()
    release = threading.Event()

    monkeypatch.setattr(gitmeta, "repo_ref_fingerprint", lambda *_args, **_kwargs: _fingerprint())
    monkeypatch.setattr(
        gitmeta,
        "commit_log_with_status",
        lambda *_args, **_kwargs: {
            "status": "ready",
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
    gitorchestration.wait_for_szz_tasks(timeout=5)

    after = server.do_project_map(str(root), include_files=True, include_git=True)
    assert after["git_analytics"]["szz"]["status"] == "ready"
    assert after["files"][0]["git"]["defect_hotspot_score"] > 0


def test_index_resumes_nonterminal_szz_sidecar(tmp_path, monkeypatch) -> None:
    root, _provider = _project(tmp_path)
    logical_project_id = _save_history(root, _ready_history())
    resumed = threading.Event()
    monkeypatch.setattr(gitmeta, "repo_ref_fingerprint", lambda *_args, **_kwargs: _fingerprint())
    gitstore.save_szz(
        logical_project_id,
        {
            "schema_version": gitstore.SCHEMA_VERSION,
            "logical_project_id": logical_project_id,
            "status": "computing",
            "fix_regex": gitmeta.DEFAULT_FIX_REGEX,
            "fingerprint": _ready_history()["fingerprint"],
            "head_commit": "c2",
            "git_common_dir": _fingerprint()["common_dir"],
            "fix_commits": 1,
            "blamed_lines": 0,
            "attributions": [],
            "commit_attributions": {},
            "workers": 1,
            "cached_commits": 0,
            "blamed_commits": 0,
            "timings_ms": {"total": 0.0, "blame": 0.0},
            "warnings": [],
        },
    )

    def fake_szz(_root, _commits, **kwargs):
        assert (kwargs.get("previous") or {}).get("status") == "computing"
        resumed.set()
        return {
            "status": "ready",
            "warning": "",
            "fix_regex": gitmeta.DEFAULT_FIX_REGEX,
            "fix_commits": 1,
            "blamed_lines": 0,
            "attributions": [],
            "commit_attributions": {},
            "workers": 1,
            "cached_commits": 0,
            "blamed_commits": 0,
            "timings_ms": {"total": 1.0, "blame": 0.0},
            "warnings": [],
        }

    monkeypatch.setattr(gitmeta, "szz_attributions_with_status", fake_szz)
    gitorchestration.ensure_shared_git_analytics(
        root=root,
        logical_project_id=logical_project_id,
        checkout_kind="main",
        enabled=True,
        fix_regex=None,
    )
    gitorchestration.wait_for_szz_tasks(timeout=5)

    assert resumed.is_set()
    assert gitstore.load_szz(logical_project_id)["status"] == "ready"


def test_shared_git_store_concurrent_writers_do_not_corrupt(tmp_path, monkeypatch) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    logical_project_id = paths.project_id_for(root)
    barrier = threading.Barrier(2)
    calls = 0
    calls_lock = threading.Lock()

    monkeypatch.setattr(gitmeta, "repo_ref_fingerprint", lambda *_args, **_kwargs: _fingerprint())
    monkeypatch.setattr(
        gitmeta,
        "commit_log_with_status",
        lambda *_args, **_kwargs: {"status": "ready", "warning": "", "commits": _history_commits()},
    )

    def fake_szz(_root, commits, **_kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
        return {
            "status": "ready",
            "warning": "",
            "fix_regex": gitmeta.DEFAULT_FIX_REGEX,
            "fix_commits": len(commits),
            "blamed_lines": 0,
            "attributions": [],
            "commit_attributions": {},
            "workers": 1,
            "cached_commits": 0,
            "blamed_commits": 0,
            "timings_ms": {"total": 1.0, "blame": 0.0},
            "warnings": [],
        }

    monkeypatch.setattr(gitmeta, "szz_attributions_with_status", fake_szz)

    def writer() -> None:
        barrier.wait()
        gitorchestration.ensure_shared_git_analytics(
            root=root,
            logical_project_id=logical_project_id,
            checkout_kind="main",
            enabled=True,
            fix_regex=None,
        )

    threads = [threading.Thread(target=writer) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    gitorchestration.wait_for_szz_tasks(timeout=5)

    history = gitstore.load_history(logical_project_id)
    szz = gitstore.load_szz(logical_project_id)
    assert history is not None
    assert szz is not None
    assert history["fingerprint"]["hash"] == "fp-c2"
    assert szz["fingerprint"]["hash"] == "fp-c2"
    assert szz["status"] == "ready"
    assert calls == 1


def test_project_map_regex_worker_crash_degrades_to_warning(tmp_path, monkeypatch) -> None:
    root, _provider = _project(tmp_path)
    _save_history(root, _ready_history())
    monkeypatch.setattr(gitmeta, "repo_ref_fingerprint", lambda *_args, **_kwargs: _fingerprint())
    monkeypatch.setattr(regexsafe, "_extract_first_worker", _crash_regex_worker)

    out = server.do_project_map(
        str(root),
        include_files=True,
        include_git=True,
        group_by="ticket",
        ticket_regex=r"([A-Z]+-\d+)",
    )

    assert "error" not in out
    assert out["git_analytics"]["available"] is True
    assert out["git_analytics"]["warnings"]
    assert any("ticket_regex" in warning for warning in out["git_analytics"]["warnings"])


def test_grep_index_worker_crash_degrades_to_partial_warning(tmp_path, monkeypatch) -> None:
    root, _provider = _project(tmp_path)
    monkeypatch.setattr(pipeline, "grep_rows_worker", _crash_grep_worker)

    out = pipeline.grep_index(str(root), "alpha")

    assert "error" not in out
    assert out["status"] == "partial"
    assert out["results"] == []
    assert out["warnings"]


class _BoomMpContext:
    """A context whose Pipe() raises OSError, simulating pipe-allocation failure.

    Before the fix, ctx/Pipe()/Process() construction sat outside the guarded
    try/except in both regexsafe._run_worker and
    pipeline._grep_rows_with_timeout, so this OSError would escape as a server
    error instead of degrading like a timeout would.
    """

    def Pipe(self, duplex: bool = False):
        raise OSError("simulated pipe allocation failure")


def test_project_map_regex_pipe_allocation_failure_degrades_to_warning(tmp_path, monkeypatch) -> None:
    root, _provider = _project(tmp_path)
    _save_history(root, _ready_history())
    monkeypatch.setattr(gitmeta, "repo_ref_fingerprint", lambda *_args, **_kwargs: _fingerprint())
    monkeypatch.setattr(regexsafe.mp, "get_context", lambda name: _BoomMpContext())

    out = server.do_project_map(
        str(root),
        include_files=True,
        include_git=True,
        group_by="ticket",
        ticket_regex=r"([A-Z]+-\d+)",
    )

    assert "error" not in out
    assert out["git_analytics"]["available"] is True
    assert out["git_analytics"]["warnings"]


def test_grep_index_pipe_allocation_failure_degrades_to_partial_warning(tmp_path, monkeypatch) -> None:
    root, _provider = _project(tmp_path)
    monkeypatch.setattr(pipeline.mp, "get_context", lambda name: _BoomMpContext())

    out = pipeline.grep_index(str(root), "alpha")

    assert "error" not in out
    assert out["status"] == "partial"
    assert out["results"] == []
    assert out["warnings"]


class _FakeGrepProc:
    """A stand-in Process whose terminate() raises; start()/is_alive() never
    actually run the worker, so recv_conn.poll(timeout) times out and the
    cleanup finally-block in _grep_rows_with_timeout runs deterministically.
    """

    def __init__(self) -> None:
        self.terminate_called = False
        self.kill_called = False
        self.join_calls = 0
        self.closed = False

    def start(self) -> None:
        pass

    def is_alive(self) -> bool:
        return True

    def terminate(self) -> None:
        self.terminate_called = True
        raise RuntimeError("simulated terminate() failure")

    def kill(self) -> None:
        self.kill_called = True

    def join(self, timeout=None) -> None:
        self.join_calls += 1

    def close(self) -> None:
        self.closed = True


class _FakeGrepConn:
    """A stand-in Connection that never delivers data, so poll(timeout) always
    times out deterministically (a real OS pipe whose write end is closed
    without a live child on the other end behaves inconsistently across
    platforms — e.g. WinError 109 instead of a clean timeout).

    ``send()`` is a no-op: _grep_rows_with_timeout's feeder thread streams
    row batches to the (fake) child over this connection after start(), and
    since nothing here ever reads them back, a real Connection could block --
    a no-op keeps the feeder thread finishing quickly regardless.
    """

    def __init__(self) -> None:
        self.closed = False

    def poll(self, timeout=None) -> bool:
        return False

    def recv(self):
        raise EOFError("no data sent")

    def send(self, _obj) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _FakeGrepCtx:
    """Records every Pipe() pair requested, in order.

    _grep_rows_with_timeout allocates two independent pipes -- one for
    streaming row batches to the (fake) child, one for the child's result --
    so this records each pair rather than only the last one.
    """

    def __init__(self, proc: "_FakeGrepProc") -> None:
        self._proc = proc
        self.pipes: list[tuple[_FakeGrepConn, _FakeGrepConn]] = []

    def Pipe(self, duplex: bool = False):
        pair = (_FakeGrepConn(), _FakeGrepConn())
        self.pipes.append(pair)
        return pair

    def Process(self, target=None, args=()):
        return self._proc


def test_grep_rows_with_timeout_terminate_failure_still_runs_kill_and_closes_both_ends(
    tmp_path, monkeypatch
) -> None:
    fake_proc = _FakeGrepProc()
    fake_ctx = _FakeGrepCtx(fake_proc)
    monkeypatch.setattr(pipeline.mp, "get_context", lambda name: fake_ctx)

    result = pipeline._grep_rows_with_timeout(
        pattern="alpha",
        flags=0,
        rows=[{"rel_path": "alpha.py", "start_line": 1, "content": "alpha"}],
        include_lines=False,
        max_matches=10,
        timeout_sec=0.05,
    )

    assert result["stopped"] is True
    assert "timed out" in result["warning"]
    assert fake_proc.terminate_called
    assert fake_proc.kill_called
    assert fake_proc.join_calls >= 2
    assert fake_proc.closed
    assert len(fake_ctx.pipes) == 2
    for recv_conn, send_conn in fake_ctx.pipes:
        assert recv_conn.closed
        assert send_conn.closed


class _CapturingProc:
    def start(self) -> None:
        pass

    def is_alive(self) -> bool:
        return True

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass

    def join(self, timeout=None) -> None:
        pass

    def close(self) -> None:
        pass


class _ReadyConn:
    """A Connection whose poll()/recv() return a canned result immediately
    (no real child ever runs), and whose send() is a no-op."""

    def __init__(self, canned=None) -> None:
        self._canned = canned
        self.closed = False

    def poll(self, timeout=None) -> bool:
        return self._canned is not None

    def recv(self):
        return self._canned

    def send(self, _obj) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _CapturingCtx:
    """Records the exact args handed to Process(), so a test can assert the
    row corpus never appears there (see grepworker.py's module docstring)."""

    def __init__(self) -> None:
        self.captured_args: tuple | None = None
        self._pipe_calls = 0

    def Pipe(self, duplex: bool = False):
        self._pipe_calls += 1
        if self._pipe_calls == 1:
            # Data pipe: the parent's write-end is never read by this fake
            # child, so its recv-side stays not-ready (unused by the parent).
            return _ReadyConn(), _ReadyConn()
        # Result pipe: pre-loaded with a canned "ok" payload with
        # total_matches=0 -- distinguishable from any answer actually derived
        # from the real (never-scanned) row corpus.
        canned = ("ok", {"by_path": {}, "total_matches": 0, "stopped": False})
        return _ReadyConn(canned=canned), _ReadyConn()

    def Process(self, target=None, args=()):
        self.captured_args = args
        return _CapturingProc()


def test_grep_rows_with_timeout_does_not_copy_corpus_through_process_args() -> None:
    """Item 5 of the search-hot-path audit: the row corpus must never be
    handed to multiprocessing.Process(args=...) -- on spawn (the only start
    method on Windows), those args are serialized and copied into the child
    synchronously inside Process.start(), before the timeout below even
    starts being measured. This is non-vacuous against the pre-fix code: the
    old signature was args=(send_conn, pattern, flags, rows, include_lines,
    max_matches), which would fail the "no list in args" assertion below (and
    also does not match this fake context's two-pipe expectations at all).
    """
    fake_ctx = _CapturingCtx()
    real_get_context = pipeline.mp.get_context
    try:
        pipeline.mp.get_context = lambda name: fake_ctx

        # A corpus large enough (~2MB) that copying it would be obviously
        # expensive if it ended up in Process(args=...).
        big_rows = [
            {"rel_path": f"file_{i}.py", "start_line": 1, "content": "x" * 1000}
            for i in range(2000)
        ]

        result = pipeline._grep_rows_with_timeout(
            pattern="x",
            flags=0,
            rows=big_rows,
            include_lines=False,
            max_matches=10,
            timeout_sec=1.0,
        )
    finally:
        pipeline.mp.get_context = real_get_context

    # The canned result (not derived from big_rows) proves this really went
    # through the fake Process path rather than silently falling back to
    # something else.
    assert result == {"by_path": {}, "total_matches": 0, "stopped": False}

    assert fake_ctx.captured_args is not None
    assert big_rows not in fake_ctx.captured_args
    assert not any(isinstance(a, list) for a in fake_ctx.captured_args)
    # Everything handed to Process(args=...) other than the two Connection
    # objects must be small, fixed-size arguments -- not a serialized copy of
    # the ~2MB row corpus.
    small_args = [a for a in fake_ctx.captured_args if not hasattr(a, "poll")]
    assert len(repr(small_args)) < 500


def test_regex_worker_crash_on_later_pass_does_not_desync_szz_identity(tmp_path, monkeypatch) -> None:
    """Regression for the unsafe-custom-regex loop.

    Unlike test_unsafe_git_fix_regex_persists_effective_identity_and_reuses_szz,
    this does NOT monkeypatch gitmeta.szz_attributions_with_status — the real
    SZZ pass runs. A worker mock that succeeds once then fails proves the
    fix-commit identity computed once during history resolution is reused for
    both the "computing" count and the SZZ blame pass, rather than being
    re-resolved (and disagreeing with itself) each time.
    """

    root, provider = _project(tmp_path)
    custom = r"(?i)\bfix\b"
    monkeypatch.setattr(gitmeta, "repo_ref_fingerprint", lambda *_args, **_kwargs: _fingerprint())
    monkeypatch.setattr(
        gitmeta,
        "commit_log_with_status",
        lambda *_args, **_kwargs: {
            "status": "ready",
            "warning": "",
            "commits": [
                {
                    "commit": "c2",
                    "parents": ["c1"],
                    "ts": 200,
                    "author": "Ann",
                    "message": "fix alpha",
                    "paths": [{"path": "alpha.py", "status": "M", "added": 1, "deleted": 0}],
                },
                {
                    "commit": "c1",
                    "parents": ["c0"],
                    "ts": 100,
                    "author": "Bob",
                    "message": "add beta",
                    "paths": [{"path": "beta.py", "status": "M", "added": 1, "deleted": 0}],
                },
            ],
        },
    )

    calls: list[str] = []

    def fake_run_worker(_target, *, pattern, flags, texts, timeout_sec):
        del timeout_sec
        calls.append(pattern)
        if len(calls) == 1:
            rx = re.compile(pattern, flags)
            return True, [bool(rx.search(text)) for text in texts], ""
        # A real second/third invocation of a catastrophic pattern could crash
        # or time out; a passing test after the fix requires this branch to
        # never be reached for the index-time count/SZZ passes.
        return False, None, "simulated crash on later pass"

    monkeypatch.setattr(regexsafe, "_run_worker", fake_run_worker)

    index_project(root, provider, full_rebuild=True, git_fix_regex=custom)
    gitorchestration.wait_for_szz_tasks(timeout=5)

    pdir = paths.project_dir(root, create=False)
    m = manifest.load_project(pdir)
    assert m is not None
    history = gitstore.load_history(m.logical_project_id)
    szz = gitstore.load_szz(m.logical_project_id)
    assert history is not None
    assert szz is not None
    assert history["fix_regex"] == custom
    # Terminal status reached (never stuck at "computing"); blame itself may
    # be "partial" since `root` here is not a real git checkout — that is
    # orthogonal to the regex-identity bug this test targets.
    assert szz["status"] in {"ready", "partial"}
    assert szz["fix_regex"] == custom  # must not have desynced to the default
    assert calls == [custom]  # resolved exactly once; reused for count + SZZ

    first = server.do_project_map(str(root), include_files=True, include_git=True)
    assert "error" not in first
    assert first["git_analytics"]["szz"]["status"] in {"ready", "partial"}

    second = server.do_project_map(str(root), include_files=True, include_git=True)
    assert "error" not in second
    assert second["git_analytics"]["szz"]["status"] in {"ready", "partial"}

    # The persisted identity is untouched by read-time recomputation (e.g.
    # churn stats), so it never perpetually re-enters "computing".
    assert gitstore.load_history(m.logical_project_id)["fix_regex"] == custom
    assert gitstore.load_szz(m.logical_project_id)["fix_regex"] == custom


def test_project_map_reuses_bounded_fix_regex_pass_per_request(tmp_path, monkeypatch) -> None:
    root, _provider = _project(tmp_path)
    custom = r"(?i)\bfix\b"
    commits = [
        {
            "commit": "c2",
            "parents": ["c1"],
            "ts": 200,
            "author": "Ann",
            "message": "fix alpha",
            "paths": [{"path": "alpha.py", "status": "M", "added": 2, "deleted": 1}],
        },
        {
            "commit": "c1",
            "parents": ["c0"],
            "ts": 100,
            "author": "Bob",
            "message": "feature beta",
            "paths": [{"path": "beta.py", "status": "M", "added": 1, "deleted": 0}],
        },
    ]
    history = _ready_history(commits)
    history["fix_regex"] = custom
    history["requested_fix_regex"] = custom
    _save_history(root, history)
    monkeypatch.setattr(gitmeta, "repo_ref_fingerprint", lambda *_args, **_kwargs: _fingerprint())
    calls: Counter[tuple[str, tuple[str, ...]]] = Counter()

    def fake_run_worker(_target, *, pattern, flags, texts, timeout_sec):
        del timeout_sec
        corpus = tuple(str(text or "") for text in texts)
        calls[(pattern, corpus)] += 1
        hits = [bool(re.search(pattern, text, flags)) for text in corpus]
        return True, hits, ""

    monkeypatch.setattr(regexsafe, "_run_worker", fake_run_worker)

    out = server.do_project_map(str(root), include_files=True, include_git=True)

    assert out["git_analytics"]["status"] == "ready"
    assert calls[(custom, ("fix alpha", "feature beta"))] == 1


def test_project_map_disabled_policy_never_performs_live_git_walk(tmp_path, monkeypatch) -> None:
    """A project indexed with --no-git-analytics must report a `disabled`
    status from project_map, even when the caller explicitly asks for
    include_git=True -- it must never fall back to a live request-path git
    walk. Spy on the git runner: any call is a hard failure."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "alpha.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    provider = _FakeProvider()
    index_project(root, provider, full_rebuild=True, git_analytics=False)

    def boom_fingerprint(*_args, **_kwargs):
        raise AssertionError("git must not be probed for a project indexed with analytics disabled")

    def boom_log(*_args, **_kwargs):
        raise AssertionError("git log must not run for a project indexed with analytics disabled")

    monkeypatch.setattr(gitmeta, "repo_ref_fingerprint", boom_fingerprint)
    monkeypatch.setattr(gitmeta, "commit_log_with_status", boom_log)

    out = server.do_project_map(str(root), include_files=True, include_git=True)

    assert out["git_analytics"]["status"] == "disabled"
    assert out["git_analytics"]["available"] is False
    assert out["git_analytics"]["warnings"]


def test_project_map_include_git_default_follows_env_var(tmp_path, monkeypatch) -> None:
    """`include_git` omitted must defer to ENGRAM_GIT_ANALYTICS, not a
    hardcoded True."""
    root, _provider = _project(tmp_path)
    _save_history(root, _ready_history())
    monkeypatch.setattr(gitmeta, "repo_ref_fingerprint", lambda *_args, **_kwargs: _fingerprint())

    monkeypatch.setenv("ENGRAM_GIT_ANALYTICS", "0")
    out_default_off = server.do_project_map(str(root), include_files=True)
    assert "git_analytics" not in out_default_off

    monkeypatch.setenv("ENGRAM_GIT_ANALYTICS", "1")
    out_default_on = server.do_project_map(str(root), include_files=True)
    assert out_default_on["git_analytics"]["status"] == "ready"

    # An explicit include_git always overrides the env var regardless of value.
    monkeypatch.setenv("ENGRAM_GIT_ANALYTICS", "0")
    out_explicit_on = server.do_project_map(str(root), include_files=True, include_git=True)
    assert out_explicit_on["git_analytics"]["status"] == "ready"


def test_full_rebuild_publishes_manifest_before_shared_git_analytics(tmp_path, monkeypatch) -> None:
    """The generation/manifest pointer must be committed to disk BEFORE the
    heavy shared git-history/SZZ pass runs, so a concurrent reader already
    sees the freshly built generation while analytics is still working."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "alpha.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    provider = _FakeProvider()

    seen_generations: list[int | None] = []

    def spy_ensure_shared_git_analytics(*, root, logical_project_id, checkout_kind, enabled, fix_regex, git_max_commits=None):
        pdir = paths.project_dir(root, create=False)
        m = manifest.load_project(pdir)
        seen_generations.append(m.generation if m is not None else None)
        return {"fix_regex": gitmeta.DEFAULT_FIX_REGEX, "requested_fix_regex": gitmeta.DEFAULT_FIX_REGEX, "warnings": []}

    monkeypatch.setattr(gitorchestration, "ensure_shared_git_analytics", spy_ensure_shared_git_analytics)

    index_project(root, provider, full_rebuild=True)

    # A first-time full rebuild has no manifest on disk at all until the
    # publish step runs; if analytics ran before publish, the spy would have
    # observed `None` (no project.json yet) instead of the freshly written
    # generation 1.
    assert seen_generations == [1]


def test_incremental_publishes_manifest_before_shared_git_analytics(tmp_path, monkeypatch) -> None:
    root, provider = _project(tmp_path)
    pdir = paths.project_dir(root, create=False)
    before = manifest.load_project(pdir)
    assert before is not None

    (root / "alpha.py").write_text("def alpha():\n    return 2\n", encoding="utf-8")

    seen_indexed_at: list[float | None] = []

    def spy_ensure_shared_git_analytics(*, root, logical_project_id, checkout_kind, enabled, fix_regex, git_max_commits=None):
        m = manifest.load_project(pdir)
        seen_indexed_at.append(m.indexed_at if m is not None else None)
        return {"fix_regex": gitmeta.DEFAULT_FIX_REGEX, "requested_fix_regex": gitmeta.DEFAULT_FIX_REGEX, "warnings": []}

    monkeypatch.setattr(gitorchestration, "ensure_shared_git_analytics", spy_ensure_shared_git_analytics)

    index_project(root, provider)  # incremental

    after = manifest.load_project(pdir)
    assert after is not None
    # If analytics ran before the incremental publish, the spy would have
    # observed the *previous* indexed_at, not the freshly written one.
    assert seen_indexed_at == [after.indexed_at]
    assert after.indexed_at != before.indexed_at


def test_reindex_file_publishes_manifest_before_shared_git_analytics(tmp_path, monkeypatch) -> None:
    from engram_mcp.pipeline import reindex_file

    root, provider = _project(tmp_path)
    pdir = paths.project_dir(root, create=False)
    before = manifest.load_project(pdir)
    assert before is not None

    (root / "alpha.py").write_text("def alpha():\n    return 2\n", encoding="utf-8")

    seen_indexed_at: list[float | None] = []

    def spy_ensure_shared_git_analytics(*, root, logical_project_id, checkout_kind, enabled, fix_regex, git_max_commits=None):
        m = manifest.load_project(pdir)
        seen_indexed_at.append(m.indexed_at if m is not None else None)
        return {"fix_regex": gitmeta.DEFAULT_FIX_REGEX, "requested_fix_regex": gitmeta.DEFAULT_FIX_REGEX, "warnings": []}

    monkeypatch.setattr(gitorchestration, "ensure_shared_git_analytics", spy_ensure_shared_git_analytics)

    reindex_file(root, provider, "alpha.py")

    after = manifest.load_project(pdir)
    assert after is not None
    assert seen_indexed_at == [after.indexed_at]
    assert after.indexed_at != before.indexed_at


def test_git_analytics_failure_does_not_roll_back_published_index(tmp_path, monkeypatch) -> None:
    """A failure in the heavy shared git-analytics pass must degrade to a
    warning, not invalidate the already-published index -- the generation
    must stay fully searchable."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "alpha.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    provider = _FakeProvider()

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated shared git-analytics failure")

    monkeypatch.setattr(gitorchestration, "ensure_shared_git_analytics", boom)

    stats = index_project(root, provider, full_rebuild=True)  # must not raise
    assert stats.mode == "full"
    assert stats.files == 1

    pdir = paths.project_dir(root, create=False)
    m = manifest.load_project(pdir)
    assert m is not None
    assert m.generation == 1
    assert m.active_table == "chunks_g1"

    hits = search_project(root, provider, "alpha", k=5)
    assert any(h["rel_path"] == "alpha.py" for h in hits)
