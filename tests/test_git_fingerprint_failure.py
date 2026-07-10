"""A transient failure to compute the *current* repo-ref fingerprint (e.g. a
`git` timeout) must never discard a previously-``ready`` shared history/SZZ
payload.

Before this fix, both the write path (``gitorchestration.ensure_shared_git_
analytics``) and the read path (``gitorchestration.analytics_source``, via
``project_map``) treated "can't compute the current fingerprint" the same as
"there is nothing usable" -- overwriting good cached commits with an
``unavailable``/empty payload, or returning ``commits: []`` even though a
ready cache existed. Both now degrade to ``status="stale"`` with the failure
attached, keeping the last known-good data intact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from engram_mcp import gitmeta, gitorchestration, gitstore, manifest, paths, server
from engram_mcp.pipeline import index_project


class _FakeProvider:
    model_id = "test:fake-fp-fail"
    backend_id = model_id
    dim = 4

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 1.0] for _ in texts]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_passages(texts)

    def release_unused_cache(self) -> None:
        pass


def _ready_fingerprint(*, hash_value: str = "fp-c1", tip: str = "c1") -> dict:
    return {
        "status": "ready",
        "algorithm": "git-refs-sha256-v1",
        "hash": hash_value,
        "refs": 1,
        "max_commit_ts": 100,
        "tip_commit": tip,
        "common_dir": "T:/fake/.git",
    }


def _commits() -> list[dict]:
    return [
        {
            "commit": "c1",
            "parents": [],
            "ts": 100,
            "author": "Ann",
            "message": "initial",
            "paths": [{"path": "alpha.py", "status": "A", "added": 3, "deleted": 0}],
        }
    ]


def _project_with_ready_history(tmp_path: Path, monkeypatch) -> tuple[Path, manifest.ProjectManifest]:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "alpha.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")

    monkeypatch.setattr(gitmeta, "repo_ref_fingerprint", lambda *_a, **_k: _ready_fingerprint())
    monkeypatch.setattr(
        gitmeta,
        "commit_log_with_status",
        lambda *_a, **_k: {"status": "ready", "warning": "", "commits": _commits()},
    )
    index_project(root, _FakeProvider(), full_rebuild=True)
    gitorchestration.wait_for_szz_tasks(timeout=5)

    pdir = paths.project_dir(root, create=False)
    m = manifest.load_project(pdir)
    assert m is not None
    history = gitstore.load_history(m.logical_project_id)
    assert history is not None and history["status"] == "ready"
    return root, m


def test_ensure_shared_git_analytics_preserves_ready_history_on_fingerprint_failure(
    tmp_path, monkeypatch
) -> None:
    root, m = _project_with_ready_history(tmp_path, monkeypatch)
    before_history = gitstore.load_history(m.logical_project_id)
    before_szz = gitstore.load_szz(m.logical_project_id)

    monkeypatch.setattr(
        gitmeta,
        "repo_ref_fingerprint",
        lambda *_a, **_k: {"status": "unavailable", "warning": "git timeout"},
    )

    result = gitorchestration.ensure_shared_git_analytics(
        root=root,
        logical_project_id=m.logical_project_id,
        checkout_kind=m.checkout_kind,
        enabled=True,
        fix_regex=None,
    )

    after_history = gitstore.load_history(m.logical_project_id)
    after_szz = gitstore.load_szz(m.logical_project_id)

    assert after_history["status"] == "stale"
    assert after_history["commits"] == before_history["commits"]
    assert after_history["fingerprint"] == before_history["fingerprint"]
    assert "git timeout" in after_history["warning"]
    # The SZZ sidecar still matches the (unchanged) preserved history's
    # fingerprint/fix_regex, so this transient-failure path must not touch it.
    assert after_szz == before_szz
    assert "git timeout" in "; ".join(result.get("warnings") or [])


def test_ensure_shared_git_analytics_still_marks_unavailable_with_nothing_cached(
    tmp_path, monkeypatch
) -> None:
    """No ready history exists yet -- the original "nothing to lose" behavior
    (persist an unavailable marker) is unchanged."""
    root = tmp_path / "proj"
    root.mkdir()
    logical_id = paths.project_id_for(root)

    monkeypatch.setattr(
        gitmeta,
        "repo_ref_fingerprint",
        lambda *_a, **_k: {"status": "unavailable", "warning": "git timeout"},
    )

    gitorchestration.ensure_shared_git_analytics(
        root=root,
        logical_project_id=logical_id,
        checkout_kind="non_git",
        enabled=True,
        fix_regex=None,
    )

    history = gitstore.load_history(logical_id)
    assert history is not None
    assert history["status"] == "unavailable"
    assert history["commits"] == []


def test_project_map_reports_stale_not_unavailable_on_fingerprint_failure(
    tmp_path, monkeypatch
) -> None:
    root, _m = _project_with_ready_history(tmp_path, monkeypatch)

    monkeypatch.setattr(
        gitmeta,
        "repo_ref_fingerprint",
        lambda *_a, **_k: {"status": "unavailable", "warning": "git timeout"},
    )

    out = server.do_project_map(str(root), include_files=True, include_git=True)

    assert out["git_analytics"]["status"] == "stale"
    assert out["git_analytics"]["cached_commits"] == 1
    assert out["git_analytics"]["scanned_commits"] == 0
    assert out["git_analytics"]["analyzed_changes"] == 1
