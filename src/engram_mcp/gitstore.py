"""Shared git-history/SZZ analytics store.

The store is keyed by ``logical_project_id`` so linked worktrees of the same
repository reuse one repo-wide git-history and SZZ payload.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from engram_mcp import paths

SCHEMA_VERSION = 2
HISTORY_FILE = "history.json"
SZZ_FILE = "szz.json"
SZZ_STATUSES = {"computing", "partial", "ready", "unavailable"}


def store_dir(logical_project_id: str, *, create: bool = True) -> Path:
    return paths.git_analytics_dir(logical_project_id, create=create)


def history_path(logical_project_id: str, *, create: bool = False) -> Path:
    return store_dir(logical_project_id, create=create) / HISTORY_FILE


def szz_path(logical_project_id: str, *, create: bool = False) -> Path:
    return store_dir(logical_project_id, create=create) / SZZ_FILE


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def save_history(logical_project_id: str, data: dict) -> None:
    payload = dict(data)
    payload["schema_version"] = SCHEMA_VERSION
    payload["logical_project_id"] = logical_project_id
    _atomic_write_json(history_path(logical_project_id, create=True), payload)


def load_history(logical_project_id: str) -> dict | None:
    data = _load_json(history_path(logical_project_id, create=False))
    if data is None or data.get("schema_version") != SCHEMA_VERSION:
        return None
    if str(data.get("logical_project_id") or "") != str(logical_project_id or ""):
        return None
    return data


def save_szz(logical_project_id: str, data: dict) -> None:
    payload = dict(data)
    payload["schema_version"] = SCHEMA_VERSION
    payload["logical_project_id"] = logical_project_id
    _atomic_write_json(szz_path(logical_project_id, create=True), payload)


def load_szz(logical_project_id: str) -> dict | None:
    data = _load_json(szz_path(logical_project_id, create=False))
    if data is None or data.get("schema_version") != SCHEMA_VERSION:
        return None
    if str(data.get("logical_project_id") or "") != str(logical_project_id or ""):
        return None
    if str(data.get("status") or "") not in SZZ_STATUSES:
        return None
    return data
