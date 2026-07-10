"""Project + file manifests (v3) with atomic on-disk writes.

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

SCHEMA_VERSION = 4
FILES_SCHEMA_VERSION = 1


@dataclass
class ProjectManifest:
    project_id: str
    root_path: str
    logical_project_id: str = ""
    checkout_kind: str = ""
    active_table: str | None = None
    generation: int = 0
    embedder_id: str = ""
    # Best-effort digest of the actual model artifact file(s) loaded (e.g. the
    # ONNX weight file's content-addressed blob name). Defense in depth *on
    # top of* embedder_id's revision pin: catches an artifact changing under
    # the same pinned revision (e.g. a corrupted/partial download) without
    # depending on it being always obtainable -- see
    # embeddings/hf_pin.py::blob_digest and index_repository.py's compat checks for
    # how an empty value on either side is treated as "unknown, skip check"
    # rather than a hard mismatch.
    embedder_artifact_digest: str = ""
    dim: int = 0
    chunker_version: str = ""
    files: int = 0
    chunks: int = 0
    indexed_at: float = 0.0
    git_worktree_root: str = ""
    indexed_ref: str = ""
    indexed_commit: str = ""
    indexed_dirty: bool = False
    git_analytics_enabled: bool = True
    git_max_commits: int | None = None
    git_fix_regex: str | None = None
    requested_git_fix_regex: str | None = None
    chunk_id_scheme: str = ""
    # O(1) read-time catalog validation token (see catalog.compute_catalog_token).
    # Written to project.json in the same locked section, immediately after the
    # matching catalog_g<generation>.json is written with the identical value,
    # so the two cannot drift. An empty/missing value (e.g. a pre-upgrade
    # manifest) is treated as "no valid catalog" -- not silently trusted.
    catalog_token: str = ""
    schema_version: int = 0


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
    if data.get("schema_version", 0) != SCHEMA_VERSION:
        raise errors.EngramError(
            f"unsupported project manifest schema_version {data.get('schema_version', 0)!r}: {f}",
            errors.E_INDEX_INVALID,
            hint="Rebuild the index with `engram index --rebuild <project_path>`.",
        )
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
    payload = asdict(manifest)
    payload["schema_version"] = SCHEMA_VERSION
    _atomic_write_json(pdir / "project.json", payload)


def load_files(pdir: Path) -> dict[str, dict]:
    """Tolerant read of ``files.json`` for non-authoritative diagnostics only.

    Returns ``{}`` on any parse/schema/provenance problem instead of raising.
    Indexing decisions (what changed, what to delete) MUST NOT be made from
    this — use `load_files_strict`, which fails loud instead of quietly
    treating a corrupt/mismatched file as "no files ever indexed".
    """
    f = pdir / "files.json"
    if not f.is_file():
        return {}
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    files = data.get("files")
    return files if isinstance(files, dict) else {}


def load_files_strict(pdir: Path, *, generation: int, active_table: str) -> dict[str, dict]:
    """Load ``files.json`` for indexing decisions (adds/changes/deletes).

    Raises `errors.EngramError` when the file is missing, corrupt, or was
    written for a different generation/active_table than the caller expects
    -- i.e. its provenance can no longer be established. Callers must treat
    that as "no reliable baseline" and force a full rebuild rather than
    compute deletions against an empty/wrong mapping.
    """
    f = pdir / "files.json"
    if not f.is_file():
        raise errors.EngramError(
            f"files manifest is missing: {f}",
            errors.E_INDEX_INVALID,
            hint="Run a full rebuild to reestablish the files manifest.",
        )
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except OSError as exc:
        raise errors.EngramError(
            f"could not read files manifest: {f}",
            errors.E_INDEX_INVALID,
            hint=str(exc),
        ) from exc
    except json.JSONDecodeError as exc:
        raise errors.EngramError(
            f"invalid files manifest JSON: {f}",
            errors.E_INDEX_INVALID,
            hint=str(exc),
        ) from exc
    if not isinstance(data, dict):
        raise errors.EngramError(
            f"invalid files manifest shape: {f}",
            errors.E_INDEX_INVALID,
        )
    if data.get("schema_version") != FILES_SCHEMA_VERSION:
        raise errors.EngramError(
            f"unsupported files manifest schema_version {data.get('schema_version')!r}: {f}",
            errors.E_INDEX_INVALID,
            hint="Run a full rebuild to reestablish the files manifest.",
        )
    if data.get("generation") != generation or data.get("active_table") != active_table:
        raise errors.EngramError(
            f"files manifest provenance mismatch: {f}",
            errors.E_INDEX_INVALID,
            hint=(
                f"files.json was written for generation={data.get('generation')!r} "
                f"active_table={data.get('active_table')!r}, but the project manifest "
                f"expects generation={generation!r} active_table={active_table!r}. "
                "Run a full rebuild."
            ),
        )
    files = data.get("files")
    if not isinstance(files, dict):
        raise errors.EngramError(
            f"invalid files manifest shape: {f}",
            errors.E_INDEX_INVALID,
        )
    return files


def save_files(pdir: Path, files: dict[str, dict], *, generation: int, active_table: str) -> None:
    payload = {
        "schema_version": FILES_SCHEMA_VERSION,
        "generation": generation,
        "active_table": active_table,
        "files": files,
    }
    _atomic_write_json(pdir / "files.json", payload)
