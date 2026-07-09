"""On-disk index inventory and orphan cleanup helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from engram_mcp import errors, paths


DEFAULT_LIST_LIMIT = 50
MAX_LIST_LIMIT = 500
_GIT_KEYS = (
    "git_worktree_root",
    "indexed_ref",
    "indexed_commit",
    "indexed_dirty",
)


def _projects_base() -> tuple[Path, Path]:
    home = paths.data_home(create=False)
    return home, home / "projects"


def _parse_limit(limit: int) -> int:
    if not isinstance(limit, int):
        raise ValueError("limit must be an integer")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    return min(limit, MAX_LIST_LIMIT)


def _parse_cursor(cursor: str | None) -> int:
    if cursor in (None, ""):
        return 0
    try:
        value = int(cursor)
    except (TypeError, ValueError) as exc:
        raise ValueError("cursor must be a cursor returned by list_indexed_projects") from exc
    if value < 0:
        raise ValueError("cursor must be non-negative")
    return value


def _empty_response(home: Path, *, home_exists: bool) -> dict:
    return {
        "data_home": str(home),
        "data_home_source": paths.data_home_source(),
        "home_exists": home_exists,
        "projects_empty": True,
        "projects": [],
        "cursor": None,
        "errors": [],
        "gc": {"pruned": [], "errors": []},
    }


def _project_dirs(base: Path) -> list[Path]:
    if not base.exists():
        return []
    return sorted(d for d in base.iterdir() if d.is_dir())


def _read_manifest(d: Path) -> tuple[dict | None, dict | None]:
    pj = d / "project.json"
    if not pj.is_file():
        return None, {
            "project_id": d.name,
            "manifest_path": str(pj),
            "code": errors.E_INDEX_INVALID,
            "error": "project manifest is missing",
        }
    try:
        raw = json.loads(pj.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, {
            "project_id": d.name,
            "manifest_path": str(pj),
            "code": errors.E_INDEX_INVALID,
            "error": f"invalid project manifest: {exc}",
        }
    if not isinstance(raw, dict):
        return None, {
            "project_id": d.name,
            "manifest_path": str(pj),
            "code": errors.E_INDEX_INVALID,
            "error": "project manifest must be a JSON object",
        }
    return raw, None


def _compact_project(d: Path, raw: dict) -> dict:
    root = raw.get("root_path") or ""
    item = {
        "project_id": raw.get("project_id") or d.name,
        "root_path": root,
        "root_exists": Path(root).exists() if root else False,
        "files": raw.get("files", 0),
        "chunks": raw.get("chunks", 0),
        "indexed_at": raw.get("indexed_at", 0.0),
        "embedder_id": raw.get("embedder_id", ""),
        "generation": raw.get("generation", 0),
    }
    for key in _GIT_KEYS:
        if raw.get(key) not in ("", None, False):
            item[key] = raw[key]
    return item


def _verbose_health(project: dict, d: Path, raw: dict) -> dict:
    root = raw.get("root_path")
    if not root:
        return project | {
            "valid": False,
            "health": {
                "code": errors.E_INDEX_INVALID,
                "error": "root_path is missing",
            },
        }
    try:
        from engram_mcp.pipeline import load_query_index

        qi = load_query_index(root)
        return project | {
            "valid": True,
            "active_table": qi.manifest.active_table,
            "table_rows": qi.count,
            "health": {"ok": True},
        }
    except errors.EngramError as exc:
        return project | {
            "valid": False,
            "health": {
                "code": exc.code,
                "error": str(exc),
                **({"hint": exc.hint} if exc.hint else {}),
            },
        }
    except Exception as exc:
        return project | {
            "valid": False,
            "health": {
                "code": errors.E_INDEX_INVALID,
                "error": str(exc),
            },
        }


def _prune_dir(d: Path, project_id: str, root_path: str) -> tuple[dict | None, dict | None]:
    item = {"project_id": project_id, "root_path": root_path, "index_path": str(d)}
    try:
        shutil.rmtree(d)
    except OSError as exc:
        return None, item | {"error": str(exc)}
    return item, None


def list_indexed_projects(
    *,
    limit: int = DEFAULT_LIST_LIMIT,
    cursor: str | None = None,
    verbose: bool = False,
    prune_orphans: bool = True,
) -> dict:
    """Return a paginated compact inventory.

    Compact mode reads manifests only. ``verbose=True`` validates table health
    and may open LanceDB for the current page.
    """

    limit = _parse_limit(limit)
    offset = _parse_cursor(cursor)
    home, base = _projects_base()
    if not home.exists():
        return _empty_response(home, home_exists=False)
    if not base.exists():
        return _empty_response(home, home_exists=True)

    dirs = _project_dirs(base)
    page = dirs[offset : offset + limit]
    next_offset = offset + limit
    out: list[dict] = []
    errs: list[dict] = []
    gc = {"pruned": [], "errors": []}

    for d in page:
        raw, err = _read_manifest(d)
        if err is not None:
            errs.append(err)
            continue
        assert raw is not None
        project = _compact_project(d, raw)
        if prune_orphans and project["root_path"] and not project["root_exists"]:
            pruned, prune_err = _prune_dir(d, project["project_id"], project["root_path"])
            if pruned is not None:
                gc["pruned"].append(pruned)
            if prune_err is not None:
                gc["errors"].append(prune_err)
            continue
        if verbose:
            project = _verbose_health(project, d, raw)
        out.append(project)

    return {
        "data_home": str(home),
        "data_home_source": paths.data_home_source(),
        "home_exists": True,
        "projects_empty": not out and not errs,
        "limit": limit,
        "cursor": str(next_offset) if next_offset < len(dirs) else None,
        "projects": out,
        "errors": errs,
        "gc": gc,
    }


def gc_orphans(*, prune: bool = False) -> dict:
    """Find or prune project index dirs whose recorded root_path is missing."""

    home, base = _projects_base()
    result = {
        "data_home": str(home),
        "data_home_source": paths.data_home_source(),
        "home_exists": home.exists(),
        "dry_run": not prune,
        "orphans": [],
        "pruned": [],
        "errors": [],
    }
    if not home.exists() or not base.exists():
        return result

    for d in _project_dirs(base):
        raw, err = _read_manifest(d)
        if err is not None:
            result["errors"].append(err)
            continue
        assert raw is not None
        project = _compact_project(d, raw)
        if not project["root_path"] or project["root_exists"]:
            continue
        orphan = {
            "project_id": project["project_id"],
            "root_path": project["root_path"],
            "index_path": str(d),
        }
        result["orphans"].append(orphan)
        if prune:
            pruned, prune_err = _prune_dir(
                d, project["project_id"], project["root_path"]
            )
            if pruned is not None:
                result["pruned"].append(pruned)
            if prune_err is not None:
                result["errors"].append(prune_err)
    return result
