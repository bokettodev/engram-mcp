"""Resolve the on-disk data directory for indexes, caches, and manifests.

Defaults to ``%LOCALAPPDATA%\\engram`` on Windows (kept out of any
OneDrive-synced project tree on purpose). Override with ``ENGRAM_HOME``.
The directory stores indexed code content, so treat it as sensitive.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from filelock import FileLock


def _resolve_data_home() -> tuple[Path, str]:
    override = os.environ.get("ENGRAM_HOME")
    if override:
        return Path(override).expanduser(), "ENGRAM_HOME"
    local = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
    if local:
        return Path(local).expanduser() / "engram", (
            "LOCALAPPDATA" if os.environ.get("LOCALAPPDATA") else "XDG_DATA_HOME"
        )
    return Path.home() / ".engram", "default"


_DATA_HOME, _DATA_HOME_SOURCE = _resolve_data_home()


def data_home(*, create: bool = True) -> Path:
    """Return the process-fixed Engram data directory.

    `ENGRAM_HOME` is resolved once when this module is imported. Read-only
    inventory paths pass ``create=False`` so inspecting an empty host does not
    create directories.
    """

    if create:
        _DATA_HOME.mkdir(parents=True, exist_ok=True)
    return _DATA_HOME


def data_home_source() -> str:
    return _DATA_HOME_SOURCE


def _reset_data_home_for_tests() -> None:
    """Refresh the process-fixed data home after test env monkeypatching."""

    global _DATA_HOME, _DATA_HOME_SOURCE
    _DATA_HOME, _DATA_HOME_SOURCE = _resolve_data_home()


def global_cache_dir(*, create: bool = True) -> Path:
    d = data_home(create=create) / "global-cache"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def project_id_for(root: Path) -> str:
    """Stable id for a project root: slug of the name + short path hash."""
    root = Path(root).resolve()
    # normcase so the id is stable regardless of drive-letter / path casing
    # on case-insensitive filesystems (Windows).
    key = os.path.normcase(str(root))
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", root.name).strip("-").lower() or "project"
    return f"{slug}-{digest}"


def canonical_path(root: str | Path) -> str:
    """Return an absolute, resolved path string with forward slashes."""

    return Path(root).expanduser().resolve().as_posix()


def logical_project_id_for_common_dir(common_dir: str | Path) -> str:
    """Stable repo identity shared by linked worktrees of one git repository."""

    common = Path(common_dir).expanduser().resolve()
    # A normal non-bare repository's common dir is `<main-worktree>/.git`.
    # Use the parent name for a human-readable slug while hashing the common dir
    # itself so every linked worktree shares the same logical id.
    slug_source = common.parent if common.name.lower() == ".git" else common
    key = os.path.normcase(common.as_posix())
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", slug_source.name).strip("-").lower() or "project"
    return f"{slug}-{digest}"


def project_dir(root: Path, create: bool = True) -> Path:
    d = data_home(create=create) / "projects" / project_id_for(root)
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def project_lock(root: Path) -> FileLock:
    """Cross-process write lock for a project's index (one writer at a time).

    The lock file lives OUTSIDE the project data dir so a holder can delete the
    project dir (remove_project) while still holding the lock.
    """
    lock_dir = data_home() / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return FileLock(str(lock_dir / f"{project_id_for(root)}.lock"))


def gpu_index_lock() -> FileLock:
    """Machine-wide CUDA index admission lock.

    This intentionally exposes one blocking FileLock. ENGRAM_GPU_INDEX_SLOTS is
    reserved for a future multi-slot implementation; the default and current
    behavior is one CUDA indexing job per data home.
    """

    lock_dir = data_home() / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return FileLock(str(lock_dir / "gpu-index.lock"))
