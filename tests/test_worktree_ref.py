from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

import pytest

from engram_mcp import gitmeta, gitstore, manifest, paths, server
from engram_mcp.pipeline import index_project, search_project, wait_for_szz_tasks


class _BranchProvider:
    backend_id: str
    dim = 4

    def __init__(self, model_id: str = "test:branch-aware") -> None:
        self.model_id = model_id
        self.backend_id = model_id

    def _vec(self, text: str) -> list[float]:
        low = text.lower()
        return [
            1.0 if "main_marker" in low or "main marker" in low else 0.0,
            1.0 if "branch_marker" in low or "branch marker" in low else 0.0,
            1.0 if "plain_marker" in low or "plain marker" in low else 0.0,
            1.0,
        ]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(text) for text in texts]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(text) for text in texts]

    def release_unused_cache(self) -> None:
        pass


def _git(repo: Path, *args: str) -> None:
    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_PAGER": "cat",
        "GIT_ASKPASS": "",
    }
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )


def _tiny_worktree_repo(tmp_path: Path) -> tuple[Path, Path, str]:
    if shutil.which("git") is None:
        pytest.skip("git is not available")

    main = tmp_path / "repo"
    worktree = tmp_path / "repo_feature"
    branch = "feature/ref-resolution"
    main.mkdir()
    try:
        _git(main, "init")
        _git(main, "config", "user.email", "engram@example.test")
        _git(main, "config", "user.name", "Engram Test")
        (main / "marker.py").write_text(
            "def main_marker():\n    return 'main marker'\n",
            encoding="utf-8",
        )
        _git(main, "add", "marker.py")
        _git(main, "commit", "-m", "initial")
        _git(main, "branch", "-M", "main")
        _git(main, "checkout", "-b", branch)
        (main / "marker.py").write_text(
            "def branch_marker():\n    return 'branch marker'\n",
            encoding="utf-8",
        )
        _git(main, "add", "marker.py")
        _git(main, "commit", "-m", "branch marker")
        _git(main, "checkout", "main")
        _git(main, "worktree", "add", str(worktree), branch)
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git worktree setup failed: {exc}")
    return main, worktree, branch


def test_git_worktree_indexes_share_logical_project_id_and_ref_query(tmp_path, monkeypatch):
    main, worktree, branch = _tiny_worktree_repo(tmp_path)
    main_provider = _BranchProvider("test:main")
    branch_provider = _BranchProvider("test:branch")

    index_project(main, main_provider, full_rebuild=True)
    index_project(worktree, branch_provider, full_rebuild=True)

    main_manifest = manifest.load_project(paths.project_dir(main, create=False))
    worktree_manifest = manifest.load_project(paths.project_dir(worktree, create=False))
    assert main_manifest is not None
    assert worktree_manifest is not None
    assert main_manifest.logical_project_id == worktree_manifest.logical_project_id
    assert main_manifest.checkout_kind == "main"
    assert worktree_manifest.checkout_kind == "worktree"
    assert main_manifest.indexed_ref == "main"
    assert worktree_manifest.indexed_ref == branch

    providers = {
        main_provider.model_id: main_provider,
        branch_provider.model_id: branch_provider,
    }
    loaded: list[str] = []

    def provider_for(model_id: str):
        loaded.append(model_id)
        return providers[model_id]

    monkeypatch.setattr(server, "_provider_for_query_model", provider_for)

    branch_out = server.do_search(
        str(main),
        "branch marker",
        k=1,
        mode="vector",
        content="full",
        ref=branch,
    )
    assert loaded[-1] == branch_provider.model_id
    assert branch_out["project_id"] == worktree_manifest.project_id
    assert branch_out["logical_project_id"] == main_manifest.logical_project_id
    assert branch_out["indexed_ref"] == branch
    assert branch_out["project_path"] == str(worktree.resolve())
    assert "branch_marker" in branch_out["results"][0]["content"]

    default_out = server.do_search(
        str(main),
        "main marker",
        k=1,
        mode="vector",
        content="full",
    )
    assert loaded[-1] == main_provider.model_id
    assert default_out["project_id"] == main_manifest.project_id
    assert default_out["indexed_ref"] == "main"
    assert "main_marker" in default_out["results"][0]["content"]

    found = server.do_find_definition(str(main), "branch_marker", ref=branch)
    assert found["project_id"] == worktree_manifest.project_id
    assert found["indexed_ref"] == branch
    assert found["results"][0]["rel_path"] == "marker.py"

    missing = server.do_search(
        str(main),
        "main marker",
        k=1,
        mode="vector",
        content="none",
        ref="missing/ref",
    )
    assert loaded[-1] == main_provider.model_id
    assert missing["project_id"] == main_manifest.project_id
    assert missing["source_revision"]["indexed"]["ref"] == "main"
    assert any(
        "no index for ref 'missing/ref'" in warning and "searched 'main'" in warning
        for warning in missing["warnings"]
    )

    missing_def = server.do_find_definition(str(main), "branch_marker", ref="missing/ref")
    assert missing_def["project_id"] == main_manifest.project_id
    assert any("no index for ref 'missing/ref'" in warning for warning in missing_def["warnings"])


def test_git_worktree_reuses_shared_repo_analytics(tmp_path, monkeypatch):
    main, worktree, _branch = _tiny_worktree_repo(tmp_path)
    calls: list[dict] = []

    def fake_szz(root, commits, **kwargs):
        calls.append(
            {
                "root": str(root),
                "commits": [commit.get("commit") for commit in commits],
                "previous": kwargs.get("previous"),
            }
        )
        return {
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
        }

    monkeypatch.setattr(gitmeta, "szz_attributions_with_status", fake_szz)

    index_project(main, _BranchProvider("test:shared-main"), full_rebuild=True)
    wait_for_szz_tasks(timeout=5)
    main_manifest = manifest.load_project(paths.project_dir(main, create=False))
    assert main_manifest is not None
    history = gitstore.load_history(main_manifest.logical_project_id)
    szz = gitstore.load_szz(main_manifest.logical_project_id)
    assert history is not None
    assert szz is not None
    assert history["status"] == "ready"
    assert szz["status"] == "ready"
    assert len(calls) == 1

    index_project(worktree, _BranchProvider("test:shared-worktree"), full_rebuild=True)
    wait_for_szz_tasks(timeout=5)
    worktree_manifest = manifest.load_project(paths.project_dir(worktree, create=False))
    assert worktree_manifest is not None
    assert worktree_manifest.logical_project_id == main_manifest.logical_project_id
    assert len(calls) == 1

    main_map = server.do_project_map(str(main), include_files=True, include_git=True)
    worktree_map = server.do_project_map(str(worktree), include_files=True, include_git=True)
    assert main_map["git_analytics"]["cache_fingerprint"]
    assert main_map["git_analytics"]["cache_fingerprint"] == worktree_map["git_analytics"]["cache_fingerprint"]
    assert main_map["git_analytics"]["analyzed_changes"] == worktree_map["git_analytics"]["analyzed_changes"]

    (worktree / "extra.py").write_text("def extra_marker():\n    return 'extra marker'\n", encoding="utf-8")
    _git(worktree, "add", "extra.py")
    _git(worktree, "commit", "-m", "fix extra marker")
    index_project(worktree, _BranchProvider("test:shared-worktree-refresh"), full_rebuild=True)
    wait_for_szz_tasks(timeout=5)
    refreshed = gitstore.load_history(main_manifest.logical_project_id)
    assert refreshed is not None
    assert refreshed["fingerprint"]["hash"] != history["fingerprint"]["hash"]
    assert len(calls) == 2


def test_non_git_identity_is_path_based_and_query_unchanged(tmp_path):
    root = tmp_path / "plain"
    root.mkdir()
    (root / "plain.py").write_text(
        "def plain_marker():\n    return 'plain marker'\n",
        encoding="utf-8",
    )
    provider = _BranchProvider("test:plain")

    index_project(root, provider, full_rebuild=True)
    m = manifest.load_project(paths.project_dir(root, create=False))
    assert m is not None
    assert m.checkout_kind == "non_git"
    assert m.logical_project_id == paths.project_id_for(root)

    out = search_project(root, provider, "plain marker", k=1, mode="vector", return_meta=True)
    assert out["project_id"] == m.project_id
    assert out["logical_project_id"] == m.logical_project_id
    assert out["hits"][0]["rel_path"] == "plain.py"
