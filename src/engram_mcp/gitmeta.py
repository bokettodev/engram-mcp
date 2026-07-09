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
