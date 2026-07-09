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
            {"path": "hot.py", "chunks": 5, "symbols": [{"name": "A"}] * 5},
            {"path": "low.py", "chunks": 1, "symbols": []},
        ],
        limit=10,
    )

    assert churn["hot.py"]["changes"] == 2
    assert churn["hot.py"]["recent_changes"] == 1
    assert churn["hot.py"]["authors_count"] == 2
    assert churn["hot.py"]["fix_density"] == 0.5
    assert hotspot["files"]["hot.py"]["hotspot_quadrant"] == "high_churn_high_complexity"
    assert hotspot["hotspots"][0]["path"] == "hot.py"


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
