from __future__ import annotations

import os
import shutil

import pytest

from engram_mcp import gitanalytics, gitmeta, regexsafe


def _crash_regex_worker(*_args) -> None:
    os._exit(1)


class _FakeProc:
    """A stand-in multiprocessing.Process whose terminate() always raises.

    start()/is_alive() never actually run the target, so the caller's
    recv_conn.poll(timeout) always times out and the cleanup finally-block
    runs deterministically without spawning a real subprocess.
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


class _FakeConn:
    """A stand-in Connection that never delivers data, so poll(timeout) always
    times out deterministically (a real OS pipe whose write end is closed
    without ever having a live child on the other end behaves inconsistently
    across platforms — e.g. WinError 109 instead of a clean timeout).
    """

    def __init__(self) -> None:
        self.closed = False

    def poll(self, timeout=None) -> bool:
        return False

    def recv(self):
        raise EOFError("no data sent")

    def close(self) -> None:
        self.closed = True


class _FakeCtx:
    def __init__(self, proc: _FakeProc) -> None:
        self._proc = proc
        self.recv_conn = None
        self.send_conn = None

    def Pipe(self, duplex: bool = False):
        self.recv_conn, self.send_conn = _FakeConn(), _FakeConn()
        return self.recv_conn, self.send_conn

    def Process(self, target=None, args=()):
        return self._proc


class _BoomCtx:
    """A context whose Pipe() raises, simulating OS-level allocation failure."""

    def Pipe(self, duplex: bool = False):
        raise OSError("simulated pipe allocation failure")


def test_regexsafe_run_worker_terminate_failure_still_runs_kill_and_closes_both_ends(
    monkeypatch,
) -> None:
    fake_proc = _FakeProc()
    fake_ctx = _FakeCtx(fake_proc)
    monkeypatch.setattr(regexsafe.mp, "get_context", lambda name: fake_ctx)

    ok, payload, reason = regexsafe._run_worker(
        regexsafe._search_many_worker,
        pattern="foo",
        flags=0,
        texts=["foo bar"],
        timeout_sec=0.05,
    )

    assert ok is False
    assert payload is None
    assert "timed out" in reason
    assert fake_proc.terminate_called
    assert fake_proc.kill_called
    assert fake_proc.join_calls >= 2
    assert fake_proc.closed
    assert fake_ctx.recv_conn is not None and fake_ctx.recv_conn.closed
    assert fake_ctx.send_conn is not None and fake_ctx.send_conn.closed


def test_regexsafe_run_worker_pipe_allocation_oserror_degrades(monkeypatch) -> None:
    monkeypatch.setattr(regexsafe.mp, "get_context", lambda name: _BoomCtx())

    ok, payload, reason = regexsafe._run_worker(
        regexsafe._search_many_worker,
        pattern="foo",
        flags=0,
        texts=["foo bar"],
        timeout_sec=1.0,
    )

    assert ok is False
    assert payload is None
    assert reason  # OSError message surfaced, not raised


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


def test_invalid_ticket_and_fix_regex_fall_back_to_defaults() -> None:
    commits = [
        _commit("c1", ["a.py"], message="ABC-1 add a"),
        _commit("c2", ["a.py"], message="fix bug"),
    ]

    grouped = gitanalytics.group_changes_result(
        commits,
        group_by="ticket",
        ticket_regex="[",
    )
    churn = gitanalytics.churn(grouped["change_sets"], fix_regex="[")

    assert grouped["regex_warnings"]
    assert len(grouped["change_sets"]) == 2
    assert churn["a.py"]["fix_density"] == 0.5


def test_pathological_ticket_regex_degrades_without_hanging(monkeypatch) -> None:
    monkeypatch.setenv("ENGRAM_USER_REGEX_TIMEOUT_SEC", "0.05")
    commits = [_commit("c1", ["a.py"], message="ABC-1 " + ("b" * 128) + "!")]

    grouped = gitanalytics.group_changes_result(
        commits,
        group_by="ticket",
        ticket_regex=r"(b+)+$",
    )

    assert grouped["regex_warnings"]
    assert grouped["change_sets"][0]["id"] == "ABC-1"


def test_pathological_fix_regex_degrades_without_hanging(monkeypatch) -> None:
    monkeypatch.setenv("ENGRAM_USER_REGEX_TIMEOUT_SEC", "0.05")
    changes = [
        {
            "id": "c1",
            "ts": 1000,
            "message": "fix bug " + ("b" * 128) + "!",
            "messages": ["fix bug " + ("b" * 128) + "!"],
            "files": ["a.py"],
            "paths": [{"path": "a.py", "status": "M", "added": 1, "deleted": 1}],
        }
    ]

    churn = gitanalytics.churn_result(changes, now_ts=1000, fix_regex=r"(b+)+$")

    assert churn["regex_warnings"]
    assert churn["fix_regex"] == gitanalytics.DEFAULT_FIX_REGEX
    assert churn["files"]["a.py"]["fix_density"] == 1.0


def test_fix_regex_worker_crash_degrades_with_warning(monkeypatch) -> None:
    monkeypatch.setattr("engram_mcp.regexsafe._search_many_worker", _crash_regex_worker)
    changes = [
        {
            "id": "c1",
            "ts": 1000,
            "message": "fix bug",
            "messages": ["fix bug"],
            "files": ["a.py"],
            "paths": [{"path": "a.py", "status": "M", "added": 1, "deleted": 1}],
        }
    ]

    churn = gitanalytics.churn_result(changes, now_ts=1000, fix_regex=r"(?i)\bfix\b")

    assert churn["regex_warnings"]
    assert churn["fix_regex"] == gitanalytics.DEFAULT_FIX_REGEX
    assert churn["files"]["a.py"]["fix_density"] == 1.0


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

    commits = gitmeta.commit_log_with_status(repo, max_commits=20)["commits"]

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

    commits = gitmeta.commit_log_with_status(repo, max_commits=None)["commits"]
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

    commits = gitmeta.commit_log_with_status(repo, max_commits=None)["commits"]
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


def test_blame_parser_accepts_sha256_commit_ids(tmp_path, monkeypatch) -> None:
    intro = "a" * 64
    excluded = "b" * 64
    boundary = "0" * 64

    def fake_git(*_args, **_kwargs):
        return "\n".join(
            [
                f"{intro} 1 1 1",
                "\told line",
                f"{excluded} 1 1 1",
                "\texcluded line",
                f"{boundary} 1 1 1",
                "\tboundary line",
            ]
        )

    monkeypatch.setattr(gitmeta, "_git", fake_git)

    blamed = gitmeta._blame_commits_for_range(
        tmp_path,
        revision="HEAD^",
        path="bug.py",
        start_line=1,
        line_count=3,
        excluded_commit=excluded,
    )

    assert blamed == {intro: 1}


def test_shared_history_uses_explicit_index_git_timeout(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ENGRAM_GIT_INDEX_TIMEOUT", "123")
    seen: dict[str, float | None] = {}

    def fake_log(*_args, **kwargs):
        seen["timeout"] = kwargs.get("timeout_sec")
        return {"status": "ready", "warning": "", "commits": []}

    monkeypatch.setattr(gitmeta, "commit_log_with_status", fake_log)
    out = gitmeta.history_for_shared_store(
        tmp_path,
        logical_project_id="logical",
        fingerprint={
            "status": "ready",
            "common_dir": str(tmp_path),
            "hash": "fp",
            "refs": 1,
            "max_commit_ts": 1,
            "tip_commit": "c",
        },
        timeout_sec=gitmeta.git_index_timeout_seconds(),
    )

    assert out["status"] == "ready"
    assert seen["timeout"] == 123.0


def test_szz_fix_regex_degrades_on_real_message_timeout(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ENGRAM_USER_REGEX_TIMEOUT_SEC", "0.05")
    monkeypatch.setenv("ENGRAM_SZZ_WORKERS", "1")
    blamed: list[str] = []

    def fake_blame(_root, commit):
        blamed.append(commit["commit"])
        return {
            "fix_commit": commit["commit"],
            "status": "ready",
            "attributions": [],
            "blamed_lines": 0,
            "warnings": [],
        }

    monkeypatch.setattr(gitmeta, "_szz_attributions_for_fix_commit", fake_blame)
    commits = [
        _commit(
            "a" * 40,
            ["bug.py"],
            message="fix bug " + ("b" * 128) + "!",
        )
    ]

    szz = gitmeta.szz_attributions_with_status(tmp_path, commits, fix_regex=r"(b+)+$")

    assert szz["fix_regex"] == gitmeta.DEFAULT_FIX_REGEX
    assert szz["warnings"]
    assert szz["fix_commits"] == 1
    assert blamed == ["a" * 40]


def test_snapshot_uses_two_git_spawns_for_normal_repo(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_git(_root, *args, **_kwargs):
        calls.append(tuple(args))
        if args[:2] == ("rev-parse", "--path-format=absolute"):
            return "\n".join([str(tmp_path), str(tmp_path / ".git"), str(tmp_path / ".git")])
        if args[:2] == ("status", "--porcelain=v2"):
            return "\n".join(
                [
                    "# branch.oid " + ("a" * 40),
                    "# branch.head main",
                    "1 .M N... 100644 100644 100644 x x a.py",
                ]
            )
        raise AssertionError(args)

    monkeypatch.setattr(gitmeta, "_git", fake_git)

    snap = gitmeta.snapshot(tmp_path)

    assert len(calls) == 2
    assert snap["indexed_ref"] == "main"
    assert snap["indexed_commit"] == "a" * 40
    assert snap["indexed_dirty"] is True


def test_snapshot_falls_back_when_git_lacks_path_format(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_git(_root, *args, **_kwargs):
        calls.append(tuple(args))
        if args[:2] == ("rev-parse", "--path-format=absolute"):
            return None
        if args[:2] == ("rev-parse", "--show-toplevel"):
            return "\n".join([".", ".git", ".git"])
        if args[:2] == ("status", "--porcelain=v2"):
            return "\n".join(
                [
                    "# branch.oid " + ("b" * 40),
                    "# branch.head main",
                ]
            )
        raise AssertionError(args)

    monkeypatch.setattr(gitmeta, "_git", fake_git)

    snap = gitmeta.snapshot(tmp_path)

    assert calls[0][:2] == ("rev-parse", "--path-format=absolute")
    assert calls[1][:2] == ("rev-parse", "--show-toplevel")
    assert snap["git_worktree_root"] == tmp_path.resolve().as_posix()
    assert snap["indexed_ref"] == "main"
    assert snap["indexed_commit"] == "b" * 40
