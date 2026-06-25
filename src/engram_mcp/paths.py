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


def data_home() -> Path:
    override = os.environ.get("ENGRAM_HOME")
    if override:
        base = Path(override)
    else:
        local = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
        base = Path(local) / "engram" if local else Path.home() / ".engram"
    base.mkdir(parents=True, exist_ok=True)
    return base


def global_cache_dir() -> Path:
    d = data_home() / "global-cache"
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


def project_dir(root: Path, create: bool = True) -> Path:
    d = data_home() / "projects" / project_id_for(root)
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
