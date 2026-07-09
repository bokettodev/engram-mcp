"""Read-only git metadata helpers.

The metadata is diagnostic only. Engram records commit/ref/dirty state so tools
can report staleness, but it never uses churn or recency as a relevance prior.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

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
    cmd = ["git", "-C", str(root), "-c", "core.fsmonitor=", "-c", "gc.auto=0", *args]
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
