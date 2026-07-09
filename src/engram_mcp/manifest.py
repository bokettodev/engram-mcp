"""Project + file manifests (v2) with atomic on-disk writes.

`project.json` holds the active LanceDB table pointer (swapped atomically on a
full rebuild) plus the embedder/chunker compatibility keys. `files.json` holds
the per-file content hashes + chunk ids that drive incremental indexing.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from engram_mcp import errors

SCHEMA_VERSION = 2


@dataclass
class ProjectManifest:
    project_id: str
    root_path: str
    active_table: str | None = None
    generation: int = 0
    embedder_id: str = ""
    dim: int = 0
    chunker_version: str = ""
    files: int = 0
    chunks: int = 0
    indexed_at: float = 0.0
    git_worktree_root: str = ""
    indexed_ref: str = ""
    indexed_commit: str = ""
    indexed_dirty: bool = False
    schema_version: int = SCHEMA_VERSION


def _atomic_write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)  # atomic on the same filesystem
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def load_project(pdir: Path) -> ProjectManifest | None:
    f = pdir / "project.json"
    if not f.is_file():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    # Tolerate a v1 manifest (no active_table) that used the fixed "chunks" table.
    if "active_table" not in data and (pdir / "lancedb").exists():
        data["active_table"] = "chunks"
    known = {fld.name for fld in dataclasses.fields(ProjectManifest)}
    return ProjectManifest(**{k: v for k, v in data.items() if k in known})


def load_project_strict(pdir: Path) -> ProjectManifest | None:
    """Load a project manifest, preserving parse/schema errors for callers."""

    f = pdir / "project.json"
    if not f.is_file():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except OSError as exc:
        raise errors.EngramError(
            f"could not read project manifest: {f}",
            errors.E_INDEX_INVALID,
            hint=str(exc),
        ) from exc
    except json.JSONDecodeError as exc:
        raise errors.EngramError(
            f"invalid project manifest JSON: {f}",
            errors.E_INDEX_INVALID,
            hint=str(exc),
        ) from exc
    if "active_table" not in data and (pdir / "lancedb").exists():
        data["active_table"] = "chunks"
    known = {fld.name for fld in dataclasses.fields(ProjectManifest)}
    try:
        return ProjectManifest(**{k: v for k, v in data.items() if k in known})
    except TypeError as exc:
        raise errors.EngramError(
            f"invalid project manifest schema: {f}",
            errors.E_INDEX_INVALID,
            hint=str(exc),
        ) from exc


def save_project(pdir: Path, manifest: ProjectManifest) -> None:
    _atomic_write_json(pdir / "project.json", asdict(manifest))


def load_files(pdir: Path) -> dict[str, dict]:
    f = pdir / "files.json"
    if not f.is_file():
        return {}
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_files(pdir: Path, files: dict[str, dict]) -> None:
    _atomic_write_json(pdir / "files.json", files)
