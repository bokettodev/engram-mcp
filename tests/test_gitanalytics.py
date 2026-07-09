from __future__ import annotations

import shutil

import pytest

from engram_mcp import gitanalytics, gitmeta


def _commit(commit: str, paths: list[str], *, ts: int = 1000, author: str = "A", message: str = "") -> dict:
    return {
        "commit": commit,
        "parents": ["p"],
        "ts": ts,
        "author": author,
        "message": message,
        "paths": [
            {"path": path, "status": "M", "added": 1, "deleted": 1}
            for path in paths
        ],
    }


def test_cochange_ranks_by_lift_not_raw_support() -> None:
    change_sets = [
        {"id": "1", "files": ["a.py", "specific.py", "barrel.py"]},
        {"id": "2", "files": ["a.py", "specific.py", "barrel.py"]},
        {"id": "3", "files": ["a.py", "barrel.py"]},
        {"id": "4", "files": ["a.py", "barrel.py"]},
        {"id": "5", "files": ["a.py", "barrel.py"]},
        {"id": "6", "files": ["x.py", "barrel.py"]},
        {"id": "7", "files": ["y.py", "barrel.py"]},
        {"id": "8", "files": ["z.py", "barrel.py"]},
        {"id": "9", "files": ["w.py", "barrel.py"]},
        {"id": "10", "files": ["q.py"]},
    ]

    coupled = gitanalytics.cochange(change_sets, limit=2)

    assert coupled["a.py"][0]["path"] == "specific.py"
    assert coupled["a.py"][0]["support"] == 2
    assert coupled["a.py"][0]["lift"] > coupled["a.py"][1]["lift"]
    assert coupled["a.py"][1]["path"] == "barrel.py"
    assert coupled["a.py"][1]["support"] == 5


def test_group_by_commit_vs_ticket_keeps_unmatched_commit() -> None:
    commits = [
        _commit("c1", ["a.py"], message="ABC-1 add a"),
        _commit("c2", ["b.py"], message="ABC-1 add b"),
        _commit("c3", ["c.py"], message="no ticket here"),
    ]

    by_commit = gitanalytics.group_changes(commits, group_by="commit")
    by_ticket = gitanalytics.group_changes(commits, group_by="ticket")

    assert len(by_commit) == 3
    assert len(by_ticket) == 2
    assert {tuple(change["files"]) for change in by_ticket} == {("a.py", "b.py"), ("c.py",)}


def test_noise_filter_and_merge_file_stats_are_skipped() -> None:
    noisy = _commit("big", [f"f{i}.py" for i in range(51)])
    merge = _commit("merge", ["merged.py"])
    merge["parents"] = ["p1", "p2"]

    grouped = gitanalytics.group_changes_result([noisy, merge], max_files_per_change=50)

    assert grouped["change_sets"] == []
    assert grouped["skipped_large_changes"] == 1
    assert grouped["skipped_merge_commits"] == 1
    assert grouped["skipped_changes"] == 2


def test_churn_recency_fix_density_and_hotspot_quadrant() -> None:
    recent = _commit("c1", ["hot.py"], ts=1000, author="Ann", message="fix hot path")
    old = _commit("c2", ["hot.py"], ts=1000 - 100 * 86400, author="Bob", message="feature")
    low = _commit("c3", ["low.py"], ts=900, author="Ann", message="feature")
    changes = gitanalytics.group_changes([recent, old, low])

    churn = gitanalytics.churn(changes, now_ts=1000, recent_days=90)
    hotspot = gitanalytics.hotspots(
        churn,
        [
            {"path": "hot.py", "chunks": 1, "symbols": [], "indent_complexity": 6.0},
            {"path": "low.py", "chunks": 100, "symbols": [{"name": "A"}] * 100, "indent_complexity": 0.5},
        ],
        limit=10,
    )

    assert churn["hot.py"]["changes"] == 2
    assert churn["hot.py"]["recent_changes"] == 1
    assert "authors_count" not in churn["hot.py"]
    assert churn["hot.py"]["fix_density"] == 0.5
    assert hotspot["files"]["hot.py"]["hotspot_quadrant"] == "high_churn_high_complexity"
    assert hotspot["files"]["hot.py"]["indent_complexity"] == 6.0
    assert hotspot["files"]["low.py"]["hotspot_quadrant"] == "low_churn_low_complexity"
    assert hotspot["hotspots"][0]["path"] == "hot.py"


def test_indentation_complexity_metric_on_synthetic_content() -> None:
    text = "\n".join(
        [
            "def outer():",
            "    if enabled:",
            "        return 1",
            "    # comment-only indentation ignored",
            "\treturn 2",
            "",
        ]
    )

    assert gitanalytics.indentation_complexity(text) == 4.0
    assert gitanalytics.indentation_complexity("# comment\nplain = 1\n") == 0.0
    assert gitanalytics.indentation_complexity("x\0y") == 0.0


def test_szz_defect_introductions_summarize_correct_introducer() -> None:
    summary = gitanalytics.defect_introductions(
        [
            {
                "fix_commit": "fix",
                "introducing_commit": "intro",
                "path": "bug.py",
                "lines": 2,
            }
        ]
    )

    assert summary["bug.py"]["introducing_commits"] == ["intro"]
    assert summary["bug.py"]["defect_introducing_commits"] == 1
    assert summary["bug.py"]["defect_introducing_lines"] == 2


def _git(repo, *args: str) -> str:
    out = gitmeta._git(repo, *args)
    if out is None:
        raise RuntimeError(f"git failed: {' '.join(args)}")
    return out


def test_commit_log_synthetic_repo_parses_stats_merge_and_rename(tmp_path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable unavailable")

    repo = tmp_path / "repo"
    repo.mkdir()
    try:
        _git(repo, "init")
        _git(repo, "config", "user.email", "test@example.com")
        _git(repo, "config", "user.name", "Test User")

        (repo / "a.txt").write_text("a1\n", encoding="utf-8")
        (repo / "b.txt").write_text("b1\n", encoding="utf-8")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "ABC-1 add A B")
        base_branch = _git(repo, "branch", "--show-current") or "master"

        (repo / "a.txt").write_text("a1\na2\n", encoding="utf-8")
        (repo / "b.txt").write_text("b1\nb2\n", encoding="utf-8")
        (repo / "test_a.txt").write_text("test\n", encoding="utf-8")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "fix ABC-1 update A B")

        for idx in range(55):
            (repo / f"noise_{idx}.txt").write_text(f"{idx}\n", encoding="utf-8")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "big generated import")

        _git(repo, "mv", "b.txt", "renamed_b.txt")
        _git(repo, "commit", "-m", "rename b")

        _git(repo, "checkout", "-b", "feature")
        (repo / "feature.txt").write_text("feature\n", encoding="utf-8")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "feature branch")
        _git(repo, "checkout", base_branch)
        _git(repo, "merge", "--no-ff", "feature", "-m", "merge feature")
    except RuntimeError as exc:
        pytest.skip(str(exc))

    commits = gitmeta.commit_log(repo, max_commits=20)

    assert commits
    assert any(len(commit["parents"]) > 1 for commit in commits)
    assert any(
        path.get("status") == "R"
        and path.get("old_path") == "b.txt"
        and path.get("path") == "renamed_b.txt"
        for commit in commits
        for path in commit["paths"]
    )
    fix_commit = next(commit for commit in commits if commit["message"].startswith("fix ABC-1"))
    assert {path["path"] for path in fix_commit["paths"]} >= {"a.txt", "renamed_b.txt", "test_a.txt"} or {
        path["path"] for path in fix_commit["paths"]
    } >= {"a.txt", "b.txt", "test_a.txt"}

    changes = gitanalytics.group_changes(commits, max_files_per_change=50)
    assert not any(len(change["files"]) > 50 for change in changes)


def test_szz_synthetic_repo_attributes_fix_to_introducing_commit(tmp_path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable unavailable")

    repo = tmp_path / "repo"
    repo.mkdir()
    try:
        _git(repo, "init")
        _git(repo, "config", "user.email", "test@example.com")
        _git(repo, "config", "user.name", "Test User")

        (repo / "bug.py").write_text("def answer():\n    return 0\n", encoding="utf-8")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "introduce bug")
        introducing = _git(repo, "rev-parse", "HEAD")

        (repo / "bug.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "fix bug in answer")
        fixing = _git(repo, "rev-parse", "HEAD")
    except RuntimeError as exc:
        pytest.skip(str(exc))

    commits = gitmeta.commit_log(repo, max_commits=None)
    szz = gitmeta.szz_attributions_with_status(repo, commits)

    assert szz["status"] in {"ready", "partial"}
    assert any(
        item["fix_commit"] == fixing
        and item["introducing_commit"] == introducing
        and item["path"] == "bug.py"
        for item in szz["attributions"]
    )


def test_parallel_szz_matches_sequential_on_synthetic_repo(tmp_path, monkeypatch) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable unavailable")

    repo = tmp_path / "repo"
    repo.mkdir()
    try:
        _git(repo, "init")
        _git(repo, "config", "user.email", "test@example.com")
        _git(repo, "config", "user.name", "Test User")

        (repo / "bug.py").write_text(
            "def answer():\n    return 0\n\ndef other():\n    return 10\n",
            encoding="utf-8",
        )
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "introduce bugs")

        (repo / "bug.py").write_text(
            "def answer():\n    return 1\n\ndef other():\n    return 10\n",
            encoding="utf-8",
        )
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "fix answer bug")

        (repo / "bug.py").write_text(
            "def answer():\n    return 1\n\ndef other():\n    return 11\n",
            encoding="utf-8",
        )
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "fix other bug")
    except RuntimeError as exc:
        pytest.skip(str(exc))

    commits = gitmeta.commit_log(repo, max_commits=None)
    monkeypatch.setenv("ENGRAM_SZZ_WORKERS", "1")
    sequential = gitmeta.szz_attributions_with_status(repo, commits)
    monkeypatch.setenv("ENGRAM_SZZ_WORKERS", "4")
    parallel = gitmeta.szz_attributions_with_status(repo, commits)

    assert sequential["attributions"] == parallel["attributions"]
    assert sequential["fix_commits"] == parallel["fix_commits"]
    assert parallel["workers"] == 4


def _fake_fix(commit: str, parent: str = "p") -> dict:
    return {
        "commit": commit,
        "parents": [parent],
        "ts": 1,
        "author": "A",
        "message": "fix bug",
        "paths": [{"path": "bug.py", "status": "M", "added": 1, "deleted": 1}],
    }


def test_szz_worker_env_one_is_sequential_and_higher_uses_pool(tmp_path, monkeypatch) -> None:
    commits = [_fake_fix("a" * 40), _fake_fix("b" * 40)]
    calls: list[str] = []

    def fake_blame(_root, commit):
        commit_id = commit["commit"]
        calls.append(commit_id)
        return {
            "fix_commit": commit_id,
            "status": "ready",
            "attributions": [
                {
                    "fix_commit": commit_id,
                    "introducing_commit": "c" * 40,
                    "path": "bug.py",
                    "lines": 1,
                }
            ],
            "blamed_lines": 1,
            "warnings": [],
        }

    monkeypatch.setattr(gitmeta, "_szz_attributions_for_fix_commit", fake_blame)
    monkeypatch.setenv("ENGRAM_SZZ_WORKERS", "1")
    sequential = gitmeta.szz_attributions_with_status(tmp_path, commits)

    assert sequential["workers"] == 1
    assert sequential["blamed_commits"] == 2
    assert calls == ["a" * 40, "b" * 40]

    constructed = 0
    original_pool = gitmeta.ThreadPoolExecutor

    class SpyPool(original_pool):
        def __init__(self, *args, **kwargs):
            nonlocal constructed
            constructed += 1
            super().__init__(*args, **kwargs)

    calls.clear()
    monkeypatch.setattr(gitmeta, "ThreadPoolExecutor", SpyPool)
    monkeypatch.setenv("ENGRAM_SZZ_WORKERS", "3")
    parallel = gitmeta.szz_attributions_with_status(tmp_path, commits)

    assert parallel["workers"] == 3
    assert parallel["blamed_commits"] == 2
    assert constructed == 1
    assert sorted(calls) == ["a" * 40, "b" * 40]


def test_szz_incremental_cache_blames_only_new_fix_commits(tmp_path, monkeypatch) -> None:
    old_commits = [_fake_fix("a" * 40), _fake_fix("b" * 40)]
    new_commits = [_fake_fix("d" * 40), *old_commits]
    blamed: list[str] = []

    def fake_blame(_root, commit):
        commit_id = commit["commit"]
        blamed.append(commit_id)
        return {
            "fix_commit": commit_id,
            "status": "ready",
            "attributions": [
                {
                    "fix_commit": commit_id,
                    "introducing_commit": "e" * 40,
                    "path": f"{commit_id[:1]}.py",
                    "lines": 2,
                }
            ],
            "blamed_lines": 2,
            "warnings": [],
        }

    monkeypatch.setattr(gitmeta, "_szz_attributions_for_fix_commit", fake_blame)
    monkeypatch.setenv("ENGRAM_SZZ_WORKERS", "1")

    first = gitmeta.szz_attributions_with_status(tmp_path, old_commits)
    assert blamed == ["a" * 40, "b" * 40]

    blamed.clear()
    second = gitmeta.szz_attributions_with_status(tmp_path, new_commits, previous=first)

    assert blamed == ["d" * 40]
    assert second["cached_commits"] == 2
    assert second["blamed_commits"] == 1
    assert len(second["commit_attributions"]) == 3
    assert {item["fix_commit"] for item in second["attributions"]} == {
        "a" * 40,
        "b" * 40,
        "d" * 40,
    }
