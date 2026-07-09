"""Read-only git metadata helpers.

The metadata is diagnostic only. Engram records commit/ref/dirty state so tools
can report staleness, but it never uses churn or recency as a relevance prior.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# Env that makes git non-interactive and non-blocking: never prompt for
# credentials, never wait on/ take .git/index.lock (git status), never talk to
# a long-running fsmonitor daemon. These are the real reasons `git status` /
# `rev-parse` hang on Windows — and when git spawns such a grandchild it also
# inherits the stdout pipe, which defeats subprocess timeout (the post-kill
# read blocks forever). Prevent the hang at the source, then hard-kill the tree.
_GIT_ENV = {
    **os.environ,
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_PAGER": "cat",
    "GIT_ASKPASS": "",
    "GIT_CONFIG_NOSYSTEM": "1",
}
_GIT_TIMEOUT_SEC = 3.0
_LOG_RECORD_SEP = "\x1e"
_LOG_FIELD_SEP = "\x1f"


def _staleness_disabled() -> bool:
    return os.environ.get("ENGRAM_GIT_STALENESS", "").strip().lower() in {"0", "false", "no", "off"}


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
            )
        else:  # pragma: no cover - posix
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.wait(timeout=3)
    except Exception:
        pass


def _git(root: Path, *args: str) -> str | None:
    """Run a read-only git command, guaranteed to return (never hang).

    Diagnostic only: any failure/timeout returns None, and the whole process
    tree is killed so a stuck git child can't wedge a search or doctor call.
    """
    if _staleness_disabled():
        return None
    cmd = [
        "git",
        "-C",
        str(root),
        "-c",
        "core.fsmonitor=",
        "-c",
        "gc.auto=0",
        "-c",
        "core.quotePath=false",
        *args,
    ]
    creation = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if sys.platform == "win32"
        else {"start_new_session": True}
    )
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_GIT_ENV,
            **creation,
        )
    except OSError:
        return None
    try:
        out, _ = proc.communicate(timeout=_GIT_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        return None
    except (OSError, ValueError, UnicodeError):
        _kill_tree(proc)
        return None
    if proc.returncode != 0:
        return None
    return out.strip()


def _normalize_git_path(value: str | None) -> str:
    text = (value or "").replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text.strip("/")


def _parse_int_stat(value: str) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _parse_raw_entry(line: str) -> dict[str, Any] | None:
    parts = line.split("\t")
    if len(parts) < 2:
        return None
    meta = parts[0].split()
    status_token = meta[-1] if meta else ""
    status = status_token[:1] or "M"
    if status in {"R", "C"} and len(parts) >= 3:
        old_path = _normalize_git_path(parts[1])
        path = _normalize_git_path(parts[2])
    else:
        old_path = ""
        path = _normalize_git_path(parts[-1])
    if not path:
        return None
    out: dict[str, Any] = {"path": path, "status": status}
    if old_path and old_path != path:
        out["old_path"] = old_path
    return out


def _parse_log_output(text: str) -> list[dict]:
    commits: list[dict] = []
    for record in text.split(_LOG_RECORD_SEP):
        record = record.strip("\r\n")
        if not record:
            continue
        lines = record.splitlines()
        if not lines:
            continue
        fields = lines[0].split(_LOG_FIELD_SEP, 4)
        if len(fields) != 5:
            continue
        commit, parents_raw, ts_raw, author, message = fields
        commit = commit.strip()
        if not commit:
            continue
        parents = [p for p in parents_raw.split() if p]
        try:
            ts = int(ts_raw)
        except ValueError:
            ts = 0
        raw_entries: list[dict[str, Any]] = []
        paths: list[dict[str, Any]] = []
        stat_index = 0
        for line in lines[1:]:
            if not line:
                continue
            if line.startswith(":"):
                raw_entry = _parse_raw_entry(line)
                if raw_entry is not None:
                    raw_entries.append(raw_entry)
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            added_raw, deleted_raw = parts[0], parts[1]
            if not (
                added_raw == "-"
                or deleted_raw == "-"
                or added_raw.isdigit()
                or deleted_raw.isdigit()
            ):
                continue
            base: dict[str, Any]
            if stat_index < len(raw_entries):
                base = dict(raw_entries[stat_index])
            else:
                base = {
                    "path": _normalize_git_path("\t".join(parts[2:])),
                    "status": "M",
                }
            stat_index += 1
            path = _normalize_git_path(base.get("path"))
            if not path:
                continue
            entry = {
                "path": path,
                "status": str(base.get("status") or "M"),
                "added": _parse_int_stat(added_raw),
                "deleted": _parse_int_stat(deleted_raw),
            }
            old_path = _normalize_git_path(base.get("old_path"))
            if old_path and old_path != path:
                entry["old_path"] = old_path
            paths.append(entry)
        commits.append(
            {
                "commit": commit,
                "parents": parents,
                "ts": ts,
                "author": author[:200],
                "message": message[:240],
                "paths": paths,
            }
        )
    return commits


def commit_log_with_status(
    root: str | Path,
    max_commits: int = 1000,
    rev_range: str | None = None,
) -> dict:
    """Return a compact parsed git history or an unavailable status.

    This helper uses the same prompt/lock/timeout-proof process wrapper as the
    staleness helpers. It never raises for git failures.
    """

    if _staleness_disabled():
        return {
            "status": "unavailable",
            "warning": "git disabled by ENGRAM_GIT_STALENESS=0",
            "commits": [],
        }
    try:
        path = Path(root).resolve()
    except OSError as exc:
        return {"status": "unavailable", "warning": str(exc), "commits": []}
    max_commits = max(1, min(int(max_commits), 100_000))
    if not (_git(path, "rev-parse", "--is-inside-work-tree") or "").strip():
        return {"status": "unavailable", "warning": "not a git worktree", "commits": []}
    fmt = f"{_LOG_RECORD_SEP}%H{_LOG_FIELD_SEP}%P{_LOG_FIELD_SEP}%ct{_LOG_FIELD_SEP}%an{_LOG_FIELD_SEP}%s"
    args = [
        "log",
        "--no-ext-diff",
        "--date-order",
        "-M",
        "--raw",
        "--numstat",
        f"--format={fmt}",
        "-n",
        str(max_commits),
    ]
    if rev_range:
        args.append(rev_range)
    args.append("--")
    out = _git(path, *args)
    if out is None:
        return {"status": "unavailable", "warning": "git log unavailable", "commits": []}
    return {"status": "ready", "warning": "", "commits": _parse_log_output(out)}


def commit_log(root: str | Path, max_commits: int = 1000) -> list[dict]:
    """Walk ``git log`` and return compact commit/file statistics.

    Failures return an empty list; callers that need to distinguish empty
    history from unavailable git should use ``commit_log_with_status``.
    """

    return list(commit_log_with_status(root, max_commits=max_commits).get("commits") or [])


def head_commit(root: str | Path) -> str:
    try:
        path = Path(root).resolve()
    except OSError:
        return ""
    return _git(path, "rev-parse", "HEAD") or ""


def is_ancestor(root: str | Path, ancestor: str, descendant: str) -> bool:
    if not ancestor or not descendant:
        return False
    try:
        path = Path(root).resolve()
    except OSError:
        return False
    out = _git(path, "merge-base", "--is-ancestor", ancestor, descendant)
    return out is not None


def history_for_catalog(
    root: str | Path,
    *,
    max_commits: int = 1000,
    previous: dict | None = None,
    head: str | None = None,
) -> dict:
    """Build/update the raw git-history block stored inside a catalog sidecar."""

    max_commits = max(1, min(int(max_commits), 100_000))
    current_head = head or head_commit(root)
    base = {
        "schema_version": 1,
        "status": "unavailable",
        "max_commits": max_commits,
        "head_commit": current_head or "",
        "commits": [],
    }
    if not current_head:
        return base | {"warning": "git head unavailable"}

    old_head = ""
    old_commits: list[dict] = []
    if isinstance(previous, dict) and previous.get("status") == "ready":
        old_head = str(previous.get("head_commit") or "")
        old_commits = [c for c in (previous.get("commits") or []) if isinstance(c, dict)]

    if old_head and old_head == current_head:
        return base | {
            "status": "ready",
            "commits": old_commits[:max_commits],
        }

    if old_head and is_ancestor(root, old_head, current_head):
        delta = commit_log_with_status(root, max_commits=max_commits, rev_range=f"{old_head}..{current_head}")
        if delta.get("status") != "ready":
            return base | {"warning": delta.get("warning") or "git log unavailable"}
        seen: set[str] = set()
        commits: list[dict] = []
        for item in list(delta.get("commits") or []) + old_commits:
            commit = str(item.get("commit") or "")
            if not commit or commit in seen:
                continue
            seen.add(commit)
            commits.append(item)
            if len(commits) >= max_commits:
                break
        return base | {
            "status": "ready",
            "commits": commits,
        }

    full = commit_log_with_status(root, max_commits=max_commits)
    if full.get("status") != "ready":
        return base | {"warning": full.get("warning") or "git log unavailable"}
    return base | {
        "status": "ready",
        "commits": list(full.get("commits") or [])[:max_commits],
    }


def snapshot(root: str | Path) -> dict:
    """Return git state for ``root`` without mutating the repository."""

    try:
        path = Path(root).resolve()
    except OSError:
        return {
            "git_worktree_root": "",
            "indexed_ref": "",
            "indexed_commit": "",
            "indexed_dirty": False,
        }
    worktree = _git(path, "rev-parse", "--show-toplevel")
    if not worktree:
        return {
            "git_worktree_root": "",
            "indexed_ref": "",
            "indexed_commit": "",
            "indexed_dirty": False,
        }
    commit = _git(path, "rev-parse", "HEAD") or ""
    ref = _git(path, "symbolic-ref", "--short", "-q", "HEAD")
    if not ref:
        ref = _git(path, "rev-parse", "--short", "HEAD") or ""
    status = _git(path, "status", "--porcelain")
    try:
        worktree_root = str(Path(worktree).resolve())
    except OSError:
        worktree_root = ""
    return {
        "git_worktree_root": worktree_root,
        "indexed_ref": ref,
        "indexed_commit": commit,
        "indexed_dirty": bool(status),
    }


def current_staleness(root: str | Path, indexed: dict) -> dict:
    """Compare current git state against manifest fields."""

    try:
        current = snapshot(root)
    except Exception as exc:
        return {
            "available": False,
            "git_stale": False,
            "reasons": [],
            "indexed": indexed,
            "current": {},
            "warning": f"git metadata unavailable: {exc}",
        }
    if not indexed.get("git_worktree_root") and not current.get("git_worktree_root"):
        return {
            "available": False,
            "git_stale": False,
            "reasons": [],
            "indexed": indexed,
            "current": current,
        }
    reasons: list[str] = []
    for key in ("git_worktree_root", "indexed_ref", "indexed_commit", "indexed_dirty"):
        if indexed.get(key) != current.get(key):
            reasons.append(key)
    return {
        "available": bool(current.get("git_worktree_root") or indexed.get("git_worktree_root")),
        "git_stale": bool(reasons),
        "reasons": reasons,
        "indexed": indexed,
        "current": current,
    }


def _snapshot_field(snapshot_data: dict, manifest_key: str, normalized_key: str, default=None):
    if manifest_key in snapshot_data:
        return snapshot_data.get(manifest_key)
    return snapshot_data.get(normalized_key, default)


def _normalize_revision_snapshot(snapshot_data: dict | None) -> dict:
    data = snapshot_data if isinstance(snapshot_data, dict) else {}
    dirty = _snapshot_field(data, "indexed_dirty", "dirty", False)
    return {
        "worktree_root": _snapshot_field(data, "git_worktree_root", "worktree_root", "") or "",
        "ref": _snapshot_field(data, "indexed_ref", "ref", "") or "",
        "commit": _snapshot_field(data, "indexed_commit", "commit", "") or "",
        "dirty": bool(dirty),
    }


def source_revision_from_staleness(git_status: dict | None) -> dict:
    """Normalize ``current_staleness`` output for search responses.

    This is a pure formatter: callers must pass an already-computed git status.
    It intentionally performs no repository inspection.
    """

    status = git_status if isinstance(git_status, dict) else {}
    available = bool(status.get("available"))
    indexed = _normalize_revision_snapshot(status.get("indexed"))
    current = _normalize_revision_snapshot(status.get("current"))
    reasons = [str(reason) for reason in (status.get("reasons") or [])]

    branch_mismatch = available and indexed["ref"] != current["ref"]
    commit_mismatch = available and indexed["commit"] != current["commit"]
    dirty_mismatch = available and indexed["dirty"] != current["dirty"]
    if not available:
        reasons = []
    else:
        for flag, reason in (
            (branch_mismatch, "indexed_ref"),
            (commit_mismatch, "indexed_commit"),
            (dirty_mismatch, "indexed_dirty"),
        ):
            if flag and reason not in reasons:
                reasons.append(reason)

    return {
        "available": available,
        "indexed": indexed,
        "current": current,
        "stale": available and bool(status.get("git_stale") or reasons),
        "branch_mismatch": bool(branch_mismatch),
        "commit_mismatch": bool(commit_mismatch),
        "dirty_mismatch": bool(dirty_mismatch),
        "reasons": reasons,
    }


def _revision_label(value: str, *, shorten: bool = False) -> str:
    text = value or "unknown"
    if shorten and len(text) > 12:
        return text[:12]
    return text


def source_revision_warning(
    source_revision: dict | None,
    *,
    include_commit_mismatch: bool = False,
) -> str | None:
    """Return a concise human/action warning for a formatted source revision."""

    revision = source_revision if isinstance(source_revision, dict) else {}
    if not revision.get("available"):
        return None
    indexed = revision.get("indexed") if isinstance(revision.get("indexed"), dict) else {}
    current = revision.get("current") if isinstance(revision.get("current"), dict) else {}
    if revision.get("branch_mismatch"):
        return (
            f"results are from indexed ref '{_revision_label(indexed.get('ref') or '')}', "
            f"not current ref '{_revision_label(current.get('ref') or '')}'; "
            "reindex the worktree to search it"
        )
    if include_commit_mismatch and revision.get("commit_mismatch"):
        return (
            f"results are from indexed commit "
            f"'{_revision_label(indexed.get('commit') or '', shorten=True)}', "
            f"not current commit '{_revision_label(current.get('commit') or '', shorten=True)}'; "
            "reindex the worktree to search it"
        )
    return None
