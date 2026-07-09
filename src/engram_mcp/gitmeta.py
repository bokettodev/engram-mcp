"""Read-only git metadata helpers.

The metadata is diagnostic only. Engram records commit/ref/dirty state so tools
can report staleness, but it never uses churn or recency as a relevance prior.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def _git(root: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


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
