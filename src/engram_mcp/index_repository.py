"""On-disk index repository and transaction boundary.

This module owns validation and access to persisted index generations.  The
write-side transaction operations live here as well, keeping manifest,
``files.json``, catalog, and Lance table publication behind one lower-level
seam.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from engram_mcp import (
    catalog,
    config,
    errors,
    gitanalytics,
    gitmeta,
    gitorchestration,
    manifest,
    paths,
    regexsafe,
)
from engram_mcp.embeddings.base import EmbeddingProvider
from engram_mcp.embeddings.cache import EmbeddingCache
from engram_mcp.indexing.chunker import chunk_file
from engram_mcp.indexing.hash import embedding_input_hash, sha256_text
from engram_mcp.indexing.languages import detect_language
from engram_mcp.indexing.walker import looks_generated, walk
from engram_mcp.store.lancedb_store import LanceStore


class ProjectNotIndexedError(RuntimeError):
    """Raised when searching a project that has no index on disk yet."""


@dataclass(slots=True)
class QueryIndex:
    root: Path
    pdir: Path
    manifest: manifest.ProjectManifest
    store: LanceStore
    count: int
    requested_root: Path | None = None
    requested_ref: str | None = None
    resolution_warnings: tuple[str, ...] = ()


@dataclass(slots=True)
class IndexStats:
    files: int
    chunks: int
    embedded_unique: int
    reused_unique: int
    seconds: float
    chunks_per_sec: float
    mode: str = "full"
    added: int = 0
    changed: int = 0
    deleted: int = 0
    unchanged: int = 0


@dataclass(slots=True)
class IndexPlan:
    root: Path
    mode: str
    compatible: bool
    files: int
    chunks: int
    added: int
    changed: int
    deleted: int
    unchanged: int
    missing_unique_chunks: int


ProgressSink = Callable[[dict[str, Any]], None] | Callable[[int, int], None]

_CONFIG_NAMES = {
    ".env", ".env.example", ".gitignore", ".dockerignore",
    "package.json", "pyproject.toml", "setup.cfg", "tox.ini",
    "tsconfig.json", "vite.config.js", "webpack.config.js",
}
_CONFIG_EXTS = {".json", ".toml", ".yaml", ".yml", ".ini", ".cfg", ".lock"}
_TEMPLATE_EXTS = {".html", ".htm", ".jinja", ".j2", ".twig", ".hbs", ".mustache"}
_EXECUTABLE_KINDS = (
    "function", "method", "class", "constructor", "interface", "type_alias",
    "enum", "struct", "trait", "impl", "macro", "namespace", "module_declaration",
)
_CHECKOUT_KINDS = {"main", "worktree", "non_git"}


def _norm_path_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def _normalize_ref(ref: str | None) -> str | None:
    if ref is None:
        return None
    text = str(ref).strip()
    return text or None


def _indexed_project_dirs() -> list[Path]:
    base = paths.data_home(create=False) / "projects"
    if not base.exists():
        return []
    try:
        return sorted(d for d in base.iterdir() if d.is_dir())
    except OSError:
        return []


def _load_manifest_lenient(pdir: Path) -> manifest.ProjectManifest | None:
    try:
        project_manifest = manifest.load_project(pdir)
    except TypeError:
        return None
    if (
        project_manifest is not None
        and project_manifest.schema_version != manifest.SCHEMA_VERSION
    ):
        return None
    return project_manifest


def _query_logical_project_id(
    root: Path, own_manifest: manifest.ProjectManifest | None
) -> str:
    if own_manifest is not None and own_manifest.logical_project_id:
        return own_manifest.logical_project_id
    try:
        snap = gitmeta.snapshot(root)
    except Exception:
        snap = {}
    logical_project_id = str(snap.get("logical_project_id") or "")
    if logical_project_id:
        return logical_project_id
    try:
        return paths.project_id_for(root)
    except OSError:
        return ""


def _manifest_root_path(project_manifest: manifest.ProjectManifest) -> Path | None:
    if not project_manifest.root_path:
        return None
    try:
        return Path(project_manifest.root_path).expanduser().resolve()
    except OSError:
        return None


def _find_index_for_ref(
    *,
    requested_root: Path,
    requested_pdir: Path,
    requested_manifest: manifest.ProjectManifest | None,
    logical_project_id: str,
    ref: str,
) -> tuple[Path, Path, manifest.ProjectManifest] | None:
    if (
        requested_manifest is not None
        and requested_manifest.logical_project_id == logical_project_id
        and requested_manifest.indexed_ref == ref
    ):
        root = _manifest_root_path(requested_manifest) or requested_root
        return root, requested_pdir, requested_manifest

    candidates: list[tuple[float, str, Path, Path, manifest.ProjectManifest]] = []
    for pdir in _indexed_project_dirs():
        project_manifest = _load_manifest_lenient(pdir)
        if project_manifest is None:
            continue
        if (
            project_manifest.logical_project_id != logical_project_id
            or project_manifest.indexed_ref != ref
        ):
            continue
        root = _manifest_root_path(project_manifest)
        if root is None:
            continue
        candidates.append(
            (
                float(project_manifest.indexed_at or 0.0),
                project_manifest.project_id,
                root,
                pdir,
                project_manifest,
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    _, _, root, pdir, project_manifest = candidates[0]
    return root, pdir, project_manifest


def _resolve_query_index(
    root: Path,
    ref: str | None,
) -> tuple[Path, Path, tuple[str, ...]]:
    requested_root = root
    requested_pdir = paths.project_dir(requested_root, create=False)
    requested_ref = _normalize_ref(ref)
    if requested_ref is None:
        return requested_root, requested_pdir, ()

    requested_manifest = _load_manifest_lenient(requested_pdir)
    logical_project_id = _query_logical_project_id(requested_root, requested_manifest)
    match = _find_index_for_ref(
        requested_root=requested_root,
        requested_pdir=requested_pdir,
        requested_manifest=requested_manifest,
        logical_project_id=logical_project_id,
        ref=requested_ref,
    )
    if match is not None:
        resolved_root, resolved_pdir, _resolved_manifest = match
        return resolved_root, resolved_pdir, ()

    searched_ref = ""
    if requested_manifest is not None:
        searched_ref = requested_manifest.indexed_ref or ""
    label = searched_ref or "unknown"
    raise errors.EngramError(
        f"no index for ref '{requested_ref}' in this logical project",
        errors.E_REF_NOT_INDEXED,
        hint=(
            f"searched indexed ref '{label}'. "
            "Index that worktree/branch before searching it with ref."
        ),
    )


def load_query_index(root: str | Path, ref: str | None = None) -> QueryIndex:
    """Validate a project's readable index before any model/provider load."""

    requested_root = Path(root).expanduser().resolve()
    requested_ref = _normalize_ref(ref)
    root, pdir, resolution_warnings = _resolve_query_index(
        requested_root, requested_ref
    )
    if not pdir.exists():
        raise ProjectNotIndexedError(f"project not indexed: {requested_root}")
    project_manifest = manifest.load_project_strict(pdir)
    if project_manifest is None:
        raise ProjectNotIndexedError(f"project not indexed: {requested_root}")
    problems: list[str] = []
    if project_manifest.schema_version != manifest.SCHEMA_VERSION:
        problems.append(
            f"schema_version {project_manifest.schema_version!r} is incompatible with "
            f"{manifest.SCHEMA_VERSION!r}"
        )
    if not project_manifest.logical_project_id:
        problems.append("logical_project_id is missing")
    if project_manifest.checkout_kind not in _CHECKOUT_KINDS:
        problems.append("checkout_kind is missing or invalid")
    if not project_manifest.active_table:
        problems.append("active_table is missing")
    if not project_manifest.embedder_id:
        problems.append("embedder_id is missing")
    if not isinstance(project_manifest.dim, int) or project_manifest.dim <= 0:
        problems.append("dim must be > 0")
    if project_manifest.chunker_version != config.CHUNKER_VERSION:
        problems.append(
            f"chunker_version {project_manifest.chunker_version!r} is incompatible with "
            f"{config.CHUNKER_VERSION!r}"
        )
    if project_manifest.chunk_id_scheme != config.CHUNK_ID_SCHEME:
        problems.append(
            f"chunk_id_scheme {project_manifest.chunk_id_scheme!r} is incompatible with "
            f"{config.CHUNK_ID_SCHEME!r}"
        )
    if not project_manifest.root_path:
        problems.append("root_path is missing")
    else:
        try:
            if _norm_path_key(Path(project_manifest.root_path)) != _norm_path_key(root):
                problems.append(
                    f"root_path {project_manifest.root_path!r} does not match requested root"
                )
        except OSError:
            problems.append(
                f"root_path {project_manifest.root_path!r} could not be resolved"
            )
    if problems:
        raise errors.EngramError(
            "invalid index manifest: " + "; ".join(problems),
            errors.E_INDEX_INVALID,
            hint="Rebuild the index with `engram index --rebuild <project_path>`.",
        )

    db_dir = pdir / "lancedb"
    if not db_dir.exists():
        raise errors.EngramError(
            f"LanceDB directory is missing: {db_dir}",
            errors.E_INDEX_INVALID,
            hint="Rebuild the index with `engram index --rebuild <project_path>`.",
        )
    store = LanceStore(
        db_dir,
        project_manifest.dim,
        table=project_manifest.active_table or "chunks",
    )
    if not store.exists():
        raise errors.EngramError(
            f"active LanceDB table {project_manifest.active_table!r} is missing",
            errors.E_INDEX_INVALID,
            hint="Rebuild the index with `engram index --rebuild <project_path>`.",
        )
    try:
        count = store.count()
    except Exception as exc:
        raise errors.EngramError(
            f"active LanceDB table {project_manifest.active_table!r} could not be read",
            errors.E_INDEX_INVALID,
            hint=str(exc),
        ) from exc
    if count <= 0:
        raise errors.EngramError(
            f"active LanceDB table {project_manifest.active_table!r} is empty",
            errors.E_INDEX_INVALID,
            hint="Rebuild the index after adding indexable source files.",
        )
    return QueryIndex(
        root=root,
        pdir=pdir,
        manifest=project_manifest,
        store=store,
        count=count,
        requested_root=requested_root,
        requested_ref=requested_ref,
        resolution_warnings=resolution_warnings,
    )


def _catalog_chunk_ids(data: dict) -> set[str]:
    ids: set[str] = set()
    files = data.get("files") or []
    if not isinstance(files, list):
        return ids
    for file_entry in files:
        if not isinstance(file_entry, dict):
            continue
        refs = file_entry.get("chunk_refs") or []
        if not isinstance(refs, list):
            continue
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            chunk_id = ref.get("chunk_id")
            if chunk_id:
                ids.add(chunk_id)
    return ids


def catalog_ref_count(data: dict) -> int:
    """Return the number of chunk references in a loaded catalog."""

    files = data.get("files") or []
    if not isinstance(files, list):
        return -1
    total = 0
    for file_entry in files:
        if not isinstance(file_entry, dict):
            return -1
        refs = file_entry.get("chunk_refs") or []
        if not isinstance(refs, list):
            return -1
        total += len(refs)
    return total


# Internal compatibility name used by the unchanged transaction code below.
_catalog_ref_count = catalog_ref_count


def catalog_validation_error(data: dict, qi: QueryIndex) -> str | None:
    """Validate a catalog on the read path with an O(1) token comparison."""

    if data.get("active_table") != qi.manifest.active_table:
        return "catalog sidecar does not match the active table"
    manifest_token = str(getattr(qi.manifest, "catalog_token", "") or "")
    catalog_token = str(data.get("catalog_token") or "")
    if not manifest_token or not catalog_token:
        return "catalog token is missing; rebuild the index to regenerate it"
    if catalog_token != manifest_token:
        return "catalog token does not match the manifest; rebuild the index"
    return None


def catalog_deep_validation_error(data: dict, qi: QueryIndex) -> str | None:
    """Perform the diagnostic O(total chunks) catalog/table comparison."""

    reason = catalog_validation_error(data, qi)
    if reason is not None:
        return reason
    try:
        table_rows = qi.store.count()
    except Exception as exc:
        return f"active table row count unavailable while validating catalog: {exc}"
    totals = data.get("totals") or {}
    if not isinstance(totals, dict):
        return "catalog totals are malformed"
    try:
        total_chunks = int(totals.get("chunks", -1) or -1)
    except (TypeError, ValueError):
        return "catalog totals are malformed"
    if total_chunks != table_rows:
        return "catalog totals differ from active table row count"
    if catalog_ref_count(data) != table_rows:
        return "catalog chunk refs differ from active table row count"
    try:
        rows = qi.store.metadata_rows_strict(columns=("chunk_id",))
    except Exception as exc:
        return f"active table metadata unavailable while validating catalog: {exc}"
    table_ids = {row.get("chunk_id") for row in rows if row.get("chunk_id")}
    catalog_ids = _catalog_chunk_ids(data)
    if len(table_ids) != table_rows:
        return "active table chunk ids differ from active table row count"
    if len(catalog_ids) != table_rows:
        return "catalog chunk ids differ from active table row count"
    if catalog_ids != table_ids:
        return "catalog chunk refs disagree with active table rows"
    return None


def load_valid_catalog(qi: QueryIndex) -> tuple[dict | None, str | None]:
    data = catalog.load_catalog(qi.pdir, qi.manifest.generation)
    if data is None:
        return None, "catalog sidecar unavailable"
    reason = catalog_validation_error(data, qi)
    if reason is not None:
        return None, reason
    return data, None


def load_project_catalog(root: str | Path) -> tuple[QueryIndex, dict]:
    qi = load_query_index(root)
    data, reason = load_valid_catalog(qi)
    if data is None:
        raise errors.EngramError(
            f"catalog sidecar for generation {qi.manifest.generation} is missing or invalid",
            errors.E_INDEX_INVALID,
            hint=(reason or "Rebuild the index to regenerate the body-free catalog."),
        )
    return qi, data


def _search_text(c) -> str:
    """The text actually embedded + full-text indexed: a contextual header
    (path / symbol / language) followed by the raw chunk content.

    v3 retained lean path/symbol/language headers."""
    header = [f"path: {c.rel_path}"]
    if c.symbol:
        header.append(f"symbol: {c.symbol}")
    if c.language:
        header.append(f"language: {c.language}")
    return "\n".join(header) + "\n\n" + c.text


def digest_mismatch(manifest_digest: str, provider_digest: str) -> bool:
    """True only when BOTH sides have a digest and they disagree.

    An empty digest on either side means "unknown/not cheaply obtainable for
    this backend" (see ``embeddings/sentence_transformers_provider.py``'s
    module docstring) and must not be treated as a mismatch -- that would
    make e.g. every GPU-built index fail this check the moment it's queried
    via the CPU backend, even though the revision pin (checked separately via
    embedder_id) already establishes they're the same vector space.
    """
    return bool(manifest_digest and provider_digest and manifest_digest != provider_digest)


def _is_compatible(m: manifest.ProjectManifest | None, provider: EmbeddingProvider) -> bool:
    return bool(
        m is not None
        and m.schema_version == manifest.SCHEMA_VERSION
        and m.logical_project_id
        and m.checkout_kind in _CHECKOUT_KINDS
        and m.active_table
        and m.embedder_id == provider.model_id
        and not digest_mismatch(m.embedder_artifact_digest, getattr(provider, "artifact_digest", ""))
        and m.dim == provider.dim
        and m.chunker_version == config.CHUNKER_VERSION
        and m.chunk_id_scheme == config.CHUNK_ID_SCHEME
    )


def _load_files_for_indexing(pdir: Path, m: manifest.ProjectManifest) -> dict[str, dict] | None:
    """Strict `files.json` load gated on the manifest's generation/active_table.

    Returns ``None`` (never ``{}``) when the file's provenance can't be
    established (missing, corrupt, or written for a different
    generation/active_table). Callers must treat ``None`` as "no reliable
    baseline" and force a full rebuild rather than compute deletions against
    an empty mapping -- an empty mapping would make every previously indexed
    file look untracked, but a still-present one look "not deleted", which
    silently leaves stale rows searchable.
    """
    try:
        return manifest.load_files_strict(
            pdir, generation=m.generation, active_table=m.active_table or ""
        )
    except errors.EngramError:
        return None


def _strict_catalog_rows(store: LanceStore) -> list[dict]:
    rows = store.metadata_rows_strict()
    count = store.count()
    if len(rows) != count:
        raise RuntimeError(
            f"metadata scan read {len(rows)} rows from {store.table!r}, expected {count}"
        )
    return rows


def _save_catalog_from_rows(
    *,
    pdir: Path,
    project_id: str,
    root: Path,
    generation: int,
    active_table: str,
    files_meta: dict[str, dict],
    rows: list[dict],
    indexed_at: float,
    store: LanceStore,
) -> str:
    """Write the catalog sidecar and return its commit token.

    Callers must set the returned token on the ``ProjectManifest`` (
    ``m.catalog_token = token``) and save that manifest before releasing the
    project lock -- see ``catalog.compute_catalog_token`` for why the two
    writes must land together.
    """
    table_rows = store.count()
    if len(rows) != table_rows:
        raise RuntimeError(
            f"refusing to write catalog: metadata rows={len(rows)} table rows={table_rows}"
        )
    data = catalog.build_catalog(
        project_id=project_id,
        root_path=str(root),
        generation=generation,
        active_table=active_table,
        files_meta=files_meta,
        rows=rows,
        indexed_at=indexed_at,
    )
    if _catalog_ref_count(data) != table_rows:
        raise RuntimeError(
            f"refusing to write catalog: catalog refs={_catalog_ref_count(data)} table rows={table_rows}"
        )
    catalog.save_catalog(pdir, data)
    return str(data.get("catalog_token") or "")


def _mark_catalog_stale_for_update(
    *,
    pdir: Path,
    root: Path,
    m: manifest.ProjectManifest,
    reason: str,
) -> None:
    catalog.mark_catalog_stale(
        pdir,
        project_id=m.project_id,
        root_path=str(root),
        generation=m.generation,
        active_table=m.active_table or "chunks",
        reason=reason,
    )


def _emit_progress(
    progress: ProgressSink | None,
    stage: str,
    *,
    unit: str | None = None,
    done: int | None = None,
    total: int | None = None,
    files: int | None = None,
    chunks: int | None = None,
    embedded: int | None = None,
    reused: int | None = None,
    **extra,
) -> None:
    if progress is None:
        return
    event = {
        "stage": stage,
        "unit": unit,
        "done": done,
        "total": total,
    }
    if files is not None:
        event["files"] = files
    if chunks is not None:
        event["chunks"] = chunks
    if embedded is not None:
        event["embedded"] = embedded
    if reused is not None:
        event["reused"] = reused
    event.update(extra)
    try:
        progress(event)  # type: ignore[misc]
    except TypeError:
        # Back-compat for the original embedding-only progress(done, total)
        # callback. Non-embedding phases are intentionally invisible to it.
        if stage == "embedding" and done is not None and total is not None:
            progress(done, total)  # type: ignore[operator]


def _maybe_emit_scanning(
    progress: ProgressSink | None,
    *,
    scanned: int,
    files: int,
    chunks: int,
    force: bool = False,
) -> None:
    if force or scanned == 0 or scanned % 100 == 0:
        _emit_progress(
            progress,
            "scanning",
            unit="files",
            done=scanned,
            total=None,
            files=files,
            chunks=chunks,
        )


def _embed(texts, provider, cache, batch_size, progress):
    """Embed only texts whose hash is not cached. Returns (vec_by_hash, hashes, n_new, n_reused)."""
    artifact_digest = str(getattr(provider, "artifact_digest", "") or "")
    hashes = [
        embedding_input_hash(provider.model_id, config.CHUNKER_VERSION, t, artifact_digest)
        for t in texts
    ]
    cached = cache.get_many(set(hashes), dim=provider.dim)
    missing: dict[str, str] = {}
    for t, h in zip(texts, hashes):
        if h not in cached and h not in missing:
            missing[h] = t
    new_vecs: dict[str, list[float]] = {}
    items = list(missing.items())
    _emit_progress(
        progress,
        "embedding_plan",
        unit="embeddings",
        done=0,
        total=len(items),
        chunks=len(texts),
        embedded=0,
        reused=len(cached),
    )
    if not items:
        _emit_progress(
            progress,
            "embedding",
            unit="embeddings",
            done=0,
            total=0,
            chunks=len(texts),
            embedded=0,
            reused=len(cached),
        )
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        vecs = provider.embed_passages([t for _, t in batch])
        for (h, _), v in zip(batch, vecs):
            new_vecs[h] = v
        done = min(i + batch_size, len(items))
        _emit_progress(
            progress,
            "embedding",
            unit="embeddings",
            done=done,
            total=len(items),
            chunks=len(texts),
            embedded=done,
            reused=len(cached),
        )
    cache.put_many(new_vecs)
    return {**cached, **new_vecs}, hashes, len(new_vecs), len(cached)


def plan_index(
    root: str | Path,
    *,
    full_rebuild: bool = False,
    model_id: str,
    dim: int,
) -> IndexPlan:
    """Torch-free index plan used for delta-aware server routing.

    The plan walks/chunks/hashes source and checks the content-addressed
    embedding cache. It never constructs an embedding provider, so its cache
    lookups use the plain ``model_id``/chunker-version hash (no
    ``artifact_digest`` component -- that requires resolving the provider's
    local model snapshot, see embeddings/hf_pin.py). This makes the plan's
    ``missing_unique_chunks`` a conservative estimate when the real digest
    later turns out to matter (never wrong in a way that skips embedding
    something real indexing needs -- worst case it slightly undercounts
    cache hits it can't itself confirm).
    """

    root = Path(root).resolve()
    pdir = paths.project_dir(root)
    m = manifest.load_project(pdir)
    compatible = bool(
        m is not None
        and m.schema_version == manifest.SCHEMA_VERSION
        and m.logical_project_id
        and m.checkout_kind in _CHECKOUT_KINDS
        and m.active_table
        and m.embedder_id == model_id
        and m.dim == dim
        and m.chunker_version == config.CHUNKER_VERSION
        and m.chunk_id_scheme == config.CHUNK_ID_SCHEME
    )
    old_files = _load_files_for_indexing(pdir, m) if compatible and m is not None else None
    if old_files is None:
        old_files = {}
        compatible = False
    force_full = full_rebuild or not compatible

    new_files: dict[str, dict] = {}
    texts: list[str] = []
    added = changed = unchanged = 0

    for rec in walk(root):
        old = old_files.get(rec.rel_path)
        if not force_full and old and old.get("size") == rec.size and old.get("mtime_ns") == rec.mtime_ns:
            new_files[rec.rel_path] = old
            unchanged += 1
            continue
        text = read_text(rec.abs_path)
        if text is None or looks_generated(text, rec.language):
            continue
        fh = sha256_text(text)
        if not force_full and old and old.get("file_hash") == fh:
            entry = dict(old)
            entry["mtime_ns"] = rec.mtime_ns
            new_files[rec.rel_path] = entry
            unchanged += 1
            continue
        chunks = chunk_file(rec.rel_path, rec.language, text)
        new_files[rec.rel_path] = {
            "file_hash": fh,
            "mtime_ns": rec.mtime_ns,
            "size": rec.size,
            "language": rec.language or "",
            "chunks": len(chunks),
        }
        texts.extend(_search_text(c) for c in chunks)
        if force_full or not old:
            added += 1
        else:
            changed += 1

    deleted = 0 if force_full else sum(1 for p in old_files if p not in new_files)
    total_chunks = sum(int(meta.get("chunks", 0) or 0) for meta in new_files.values())
    hashes = {
        embedding_input_hash(model_id, config.CHUNKER_VERSION, text)
        for text in texts
    }
    if hashes:
        with EmbeddingCache(paths.global_cache_dir() / "embeddings.sqlite") as cache:
            cached = cache.get_many(hashes, dim=dim)
        missing = len(hashes - set(cached))
    else:
        missing = 0
    return IndexPlan(
        root=root,
        mode="full" if force_full else "incremental",
        compatible=compatible,
        files=len(new_files),
        chunks=total_chunks,
        added=added,
        changed=changed,
        deleted=deleted,
        unchanged=unchanged,
        missing_unique_chunks=missing,
    )


def _rows(chunks, search_texts, hashes, vec_by_hash, file_hash_by_path) -> list[dict]:
    """Build LanceDB rows, one per chunk.

    ``chunk_id`` must be a pure function of the chunk itself so it is
    byte-identical whether produced by a full rebuild or an incremental
    reindex of just its file, regardless of which other files share the
    batch. It is keyed on a per-file ordinal (the chunk's index within its
    own file's chunk list) rather than the batch's global enumeration index,
    which used to vary with what else was indexed alongside it. See
    ``config.CHUNK_ID_SCHEME``.
    """
    out = []
    file_ordinal: dict[str, int] = {}
    for c, st, h in zip(chunks, search_texts, hashes):
        idx = file_ordinal.get(c.rel_path, 0)
        file_ordinal[c.rel_path] = idx + 1
        chash = sha256_text(c.text)
        out.append(
            {
                "chunk_id": f"{sha256_text(c.rel_path)[:12]}:{idx}:{chash[:8]}",
                "rel_path": c.rel_path,
                "language": c.language or "",
                "symbol": c.symbol or "",
                "symbol_kind": c.symbol_kind or "",
                "chunk_role": derive_chunk_role(c.rel_path, c.language, c.symbol_kind),
                "start_line": c.start_line,
                "end_line": c.end_line,
                "content": c.text,
                "search_text": st,
                "file_hash": file_hash_by_path.get(c.rel_path, ""),
                "chunk_hash": chash,
                "vector": vec_by_hash[h],
            }
        )
    return out


def index_project(
    root: str | Path,
    provider: EmbeddingProvider,
    *,
    full_rebuild: bool = False,
    batch_size: int = config.EMBED_BATCH_SIZE,
    progress: ProgressSink | None = None,
    git_analytics: bool = True,
    git_max_commits: int | None = None,
    git_fix_regex: str | None = None,
) -> IndexStats:
    root = Path(root).resolve()
    pdir = paths.project_dir(root)
    _emit_progress(progress, "waiting_for_lock", unit="lock", done=0, total=None)
    with paths.project_lock(root):
        m = manifest.load_project(pdir)
        if (
            m is not None
            and m.schema_version != manifest.SCHEMA_VERSION
            and not full_rebuild
        ):
            raise errors.EngramError(
                f"unsupported index manifest schema_version {m.schema_version!r}",
                errors.E_INDEX_INVALID,
                hint="Run `engram index --rebuild <project_path>` to create a v3 manifest.",
            )
        # `old_files` provenance (schema/generation/active_table) must match
        # `m` before an incremental update trusts it as the deletion
        # baseline. A corrupt/mismatched/missing files.json means "no
        # reliable baseline" -> force a full rebuild instead of silently
        # treating everything as newly added (and nothing as deleted).
        old_files = _load_files_for_indexing(pdir, m) if _is_compatible(m, provider) else None
        compatible = old_files is not None
        if full_rebuild or not compatible:
            return _full_rebuild(
                root,
                pdir,
                provider,
                m,
                batch_size,
                progress,
                git_analytics=git_analytics,
                git_max_commits=git_max_commits,
                git_fix_regex=git_fix_regex,
            )
        return _incremental(
            root,
            pdir,
            provider,
            m,
            batch_size,
            progress,
            old_files,
            git_analytics=git_analytics,
            git_max_commits=git_max_commits,
            git_fix_regex=git_fix_regex,
        )


def _full_rebuild(
    root,
    pdir,
    provider,
    m,
    batch_size,
    progress,
    *,
    git_analytics: bool,
    git_max_commits: int | None,
    git_fix_regex: str | None,
) -> IndexStats:
    t0 = time.time()
    files_meta: dict[str, dict] = {}
    chunks = []
    scanned = 0
    _maybe_emit_scanning(progress, scanned=0, files=0, chunks=0, force=True)
    for rec in walk(root):
        scanned += 1
        text = read_text(rec.abs_path)
        if text is None or looks_generated(text, rec.language):
            _maybe_emit_scanning(
                progress,
                scanned=scanned,
                files=len(files_meta),
                chunks=len(chunks),
            )
            continue
        cs = chunk_file(rec.rel_path, rec.language, text)
        chunks.extend(cs)
        files_meta[rec.rel_path] = {
            "file_hash": sha256_text(text), "mtime_ns": rec.mtime_ns, "size": rec.size,
            "language": rec.language or "", "chunks": len(cs),
            "indent_complexity": gitanalytics.indentation_complexity(text),
        }
        _maybe_emit_scanning(
            progress,
            scanned=scanned,
            files=len(files_meta),
            chunks=len(chunks),
        )
    _maybe_emit_scanning(
        progress,
        scanned=scanned,
        files=len(files_meta),
        chunks=len(chunks),
        force=True,
    )
    file_hash_by_path = {p: meta["file_hash"] for p, meta in files_meta.items()}
    search_texts = [_search_text(c) for c in chunks]

    with EmbeddingCache(paths.global_cache_dir() / "embeddings.sqlite") as cache:
        vec_by_hash, hashes, embedded, reused = _embed(search_texts, provider, cache, batch_size, progress)
    rows = _rows(chunks, search_texts, hashes, vec_by_hash, file_hash_by_path)

    gen = (m.generation + 1) if m else 1
    new_table = f"chunks_g{gen}"
    db_dir = pdir / "lancedb"
    # GC tables left over from prior interrupted runs, but keep the current
    # active table so concurrent readers on the old pointer don't break.
    keep = {m.active_table} if m and m.active_table else set()
    dropped = LanceStore(db_dir, provider.dim).drop_stale_generations(keep)
    stale_catalog_generations = {
        generation
        for generation in (catalog.generation_for_table(name) for name in dropped)
        if generation is not None
    }
    catalog.drop_catalogs_for_generations(pdir, stale_catalog_generations)
    store = LanceStore(db_dir, provider.dim, table=new_table)
    _emit_progress(
        progress,
        "writing_table",
        unit="rows",
        done=0,
        total=len(rows),
        files=len(files_meta),
        chunks=len(chunks),
        embedded=embedded,
        reused=reused,
    )
    store.create(rows, refresh_fts=False)
    _emit_progress(
        progress,
        "writing_table",
        unit="rows",
        done=len(rows),
        total=len(rows),
        files=len(files_meta),
        chunks=len(chunks),
        embedded=embedded,
        reused=reused,
    )
    _emit_progress(
        progress,
        "writing_fts",
        unit="tables",
        done=0,
        total=1,
        files=len(files_meta),
        chunks=len(chunks),
        embedded=embedded,
        reused=reused,
    )
    store.refresh_fts()
    _emit_progress(
        progress,
        "writing_fts",
        unit="tables",
        done=1,
        total=1,
        files=len(files_meta),
        chunks=len(chunks),
        embedded=embedded,
        reused=reused,
    )

    indexed_at = time.time()
    git = gitmeta.snapshot(root)
    logical_project_id = str(git.get("logical_project_id") or paths.project_id_for(root))
    checkout_kind = str(git.get("checkout_kind") or "non_git")
    # Resolve the persisted fix-regex identity cheaply (pure regex validation,
    # no git I/O) so the manifest below can be published without waiting on
    # the heavy shared git-history/SZZ computation.
    fix_regex_value, requested_fix_regex_value = _resolve_git_fix_regex_identity(git_fix_regex)
    catalog_token = _save_catalog_from_rows(
        pdir=pdir,
        project_id=paths.project_id_for(root),
        root=root,
        generation=gen,
        active_table=new_table,
        files_meta=files_meta,
        rows=rows,
        indexed_at=indexed_at,
        store=store,
    )
    # Commit the pointer FIRST (it references the fully-built new table), then
    # the file manifest. A crash in between leaves the new table active but
    # files.json still bound to the previous generation/active_table; the
    # strict files loader then detects that provenance mismatch and forces
    # the next index run to do a full rebuild rather than silently
    # reconciling against a stale baseline.
    new_manifest = manifest.ProjectManifest(
        project_id=paths.project_id_for(root), root_path=str(root),
        active_table=new_table, generation=gen,
        embedder_id=provider.model_id, dim=provider.dim,
        embedder_artifact_digest=str(getattr(provider, "artifact_digest", "") or ""),
        chunker_version=config.CHUNKER_VERSION,
        chunk_id_scheme=config.CHUNK_ID_SCHEME,
        files=len(files_meta), chunks=len(chunks), indexed_at=indexed_at,
        git_analytics_enabled=bool(git_analytics),
        git_max_commits=gitorchestration.coerce_git_max_commits(git_max_commits),
        git_fix_regex=fix_regex_value,
        requested_git_fix_regex=requested_fix_regex_value,
        catalog_token=catalog_token,
        **git,
    )
    manifest.save_project(pdir, new_manifest)
    manifest.save_files(pdir, files_meta, generation=gen, active_table=new_table)
    # The previous active table is intentionally retained for in-flight readers;
    # it is GC'd at the start of the next rebuild.
    #
    # The index is now fully published and searchable (manifest generation
    # pointer + files baseline are on disk). Only now do we run the heavy
    # shared git-history/SZZ analytics -- on a large/ancient repo this can
    # synchronously walk `git log --all --raw --numstat` for up to
    # `gitmeta.git_index_timeout_seconds()`, and it must never delay the
    # moment the freshly built generation becomes searchable.
    git_regex = _run_shared_git_analytics_after_publish(
        root=root,
        logical_project_id=logical_project_id,
        checkout_kind=checkout_kind,
        enabled=git_analytics,
        fix_regex=git_fix_regex,
        git_max_commits=git_max_commits,
    )
    # A syntactically valid but unsafe/too-slow custom regex can only be
    # detected by actually running it against the real commit corpus -- the
    # heavy work just deferred above. Reconcile the provisional identity with
    # the one the heavy pass actually resolved, still under the project lock.
    _reconcile_git_fix_regex(
        pdir,
        new_manifest,
        git_regex=git_regex,
        provisional_fix_regex=fix_regex_value,
        provisional_requested_fix_regex=requested_fix_regex_value,
    )

    elapsed = time.time() - t0
    _emit_progress(
        progress,
        "done",
        unit="chunks",
        done=len(chunks),
        total=len(chunks),
        files=len(files_meta),
        chunks=len(chunks),
        embedded=embedded,
        reused=reused,
        mode="full",
    )
    return IndexStats(
        files=len(files_meta), chunks=len(chunks), embedded_unique=embedded,
        reused_unique=reused, seconds=elapsed,
        chunks_per_sec=(len(chunks) / elapsed if elapsed > 0 else 0.0),
        mode="full", added=len(files_meta),
    )


def _incremental(
    root,
    pdir,
    provider,
    m,
    batch_size,
    progress,
    old_files: dict[str, dict],
    *,
    git_analytics: bool,
    git_max_commits: int | None,
    git_fix_regex: str | None,
) -> IndexStats:
    """Mutate the active table in place.

    ``old_files`` must already be the strictly-loaded, provenance-checked
    baseline (see `_load_files_for_indexing`) -- this function trusts it
    verbatim to decide adds/changes/deletes, so callers must never pass a
    tolerant/empty-on-error read here.
    """
    t0 = time.time()
    new_files: dict[str, dict] = {}
    touched = []
    added = changed = unchanged = 0
    scanned = 0

    _maybe_emit_scanning(progress, scanned=0, files=0, chunks=0, force=True)
    for rec in walk(root):
        scanned += 1
        old = old_files.get(rec.rel_path)
        if old and old.get("size") == rec.size and old.get("mtime_ns") == rec.mtime_ns:
            entry = dict(old)
            if "indent_complexity" not in entry:
                text = read_text(rec.abs_path)
                if text is not None:
                    entry["indent_complexity"] = gitanalytics.indentation_complexity(text)
            new_files[rec.rel_path] = entry
            unchanged += 1
            _maybe_emit_scanning(
                progress,
                scanned=scanned,
                files=len(new_files),
                chunks=sum(int(meta.get("chunks", 0)) for meta in new_files.values()),
            )
            continue
        text = read_text(rec.abs_path)
        if text is None or looks_generated(text, rec.language):
            # Skipped (unreadable/generated): not added to new_files, so if it
            # was previously indexed it falls into deleted_paths below.
            _maybe_emit_scanning(
                progress,
                scanned=scanned,
                files=len(new_files),
                chunks=sum(int(meta.get("chunks", 0)) for meta in new_files.values()),
            )
            continue
        fh = sha256_text(text)
        if old and old.get("file_hash") == fh:
            entry = dict(old)
            entry["mtime_ns"] = rec.mtime_ns
            entry["indent_complexity"] = gitanalytics.indentation_complexity(text)
            new_files[rec.rel_path] = entry
            unchanged += 1
            _maybe_emit_scanning(
                progress,
                scanned=scanned,
                files=len(new_files),
                chunks=sum(int(meta.get("chunks", 0)) for meta in new_files.values()),
            )
            continue
        touched.append((rec, text, fh))
        changed += 1 if old else 0
        added += 0 if old else 1
        _maybe_emit_scanning(
            progress,
            scanned=scanned,
            files=len(new_files) + len(touched),
            chunks=sum(int(meta.get("chunks", 0)) for meta in new_files.values()),
        )

    chunks = []
    file_hash_by_path: dict[str, str] = {}
    for rec, text, fh in touched:
        cs = chunk_file(rec.rel_path, rec.language, text)
        chunks.extend(cs)
        file_hash_by_path[rec.rel_path] = fh
        new_files[rec.rel_path] = {
            "file_hash": fh, "mtime_ns": rec.mtime_ns, "size": rec.size,
            "language": rec.language or "", "chunks": len(cs),
            "indent_complexity": gitanalytics.indentation_complexity(text),
        }
    _maybe_emit_scanning(
        progress,
        scanned=scanned,
        files=len(new_files),
        chunks=sum(int(meta.get("chunks", 0)) for meta in new_files.values()),
        force=True,
    )

    # Anything previously indexed but no longer kept (deleted, generated, or
    # unreadable) gets its rows removed.
    deleted_paths = [p for p in old_files if p not in new_files]

    embedded = reused = 0
    rows = []
    if chunks:
        search_texts = [_search_text(c) for c in chunks]
        with EmbeddingCache(paths.global_cache_dir() / "embeddings.sqlite") as cache:
            vec_by_hash, hashes, embedded, reused = _embed(search_texts, provider, cache, batch_size, progress)
        rows = _rows(chunks, search_texts, hashes, vec_by_hash, file_hash_by_path)
    else:
        _emit_progress(
            progress,
            "embedding_plan",
            unit="embeddings",
            done=0,
            total=0,
            chunks=0,
            embedded=0,
            reused=0,
        )
        _emit_progress(
            progress,
            "embedding",
            unit="embeddings",
            done=0,
            total=0,
            chunks=0,
            embedded=0,
            reused=0,
        )

    store = LanceStore(pdir / "lancedb", provider.dim, table=m.active_table)
    row_work = len(rows) + len(touched) + len(deleted_paths)
    _emit_progress(
        progress,
        "writing_table",
        unit="rows",
        done=0,
        total=row_work,
        files=len(new_files),
        chunks=m.chunks,
        embedded=embedded,
        reused=reused,
    )
    if touched or deleted_paths:
        _mark_catalog_stale_for_update(
            pdir=pdir,
            root=root,
            m=m,
            reason="incremental update mutating active table",
        )
    store.delete_paths([rec.rel_path for rec, _, _ in touched] + deleted_paths)
    store.add(rows)
    _emit_progress(
        progress,
        "writing_table",
        unit="rows",
        done=row_work,
        total=row_work,
        files=len(new_files),
        chunks=m.chunks,
        embedded=embedded,
        reused=reused,
    )
    if touched or deleted_paths:
        _emit_progress(
            progress,
            "writing_fts",
            unit="tables",
            done=0,
            total=1,
            files=len(new_files),
            chunks=m.chunks,
            embedded=embedded,
            reused=reused,
        )
        store.refresh_fts()
        _emit_progress(
            progress,
            "writing_fts",
            unit="tables",
            done=1,
            total=1,
            files=len(new_files),
            chunks=m.chunks,
            embedded=embedded,
            reused=reused,
        )
    else:
        _emit_progress(
            progress,
            "writing_fts",
            unit="tables",
            done=0,
            total=0,
            files=len(new_files),
            chunks=m.chunks,
            embedded=embedded,
            reused=reused,
        )

    indexed_at = time.time()
    git = gitmeta.snapshot(root)
    catalog_rows = _strict_catalog_rows(store)
    requested_git_fix_regex = (
        git_fix_regex
        if git_fix_regex is not None
        else (getattr(m, "requested_git_fix_regex", None) or m.git_fix_regex)
    )
    logical_project_id = str(git.get("logical_project_id") or m.logical_project_id)
    checkout_kind = str(git.get("checkout_kind") or m.checkout_kind)
    # Resolve the persisted fix-regex identity cheaply (pure regex validation,
    # no git I/O) so the manifest below can be published without waiting on
    # the heavy shared git-history/SZZ computation.
    original_git_fix_regex = m.git_fix_regex
    reuse_previous_fix_regex = not git_analytics and git_fix_regex is None
    fix_regex_value, requested_fix_regex_value = _resolve_git_fix_regex_identity(
        requested_git_fix_regex,
        reuse_previous=reuse_previous_fix_regex,
        previous_fix_regex=original_git_fix_regex,
    )
    catalog_token = _save_catalog_from_rows(
        pdir=pdir,
        project_id=m.project_id,
        root=root,
        generation=m.generation,
        active_table=m.active_table or "chunks",
        files_meta=new_files,
        rows=catalog_rows,
        indexed_at=indexed_at,
        store=store,
    )
    manifest.save_files(pdir, new_files, generation=m.generation, active_table=m.active_table or "chunks")
    m.files = len(new_files)
    m.chunks = store.count()
    m.indexed_at = indexed_at
    m.catalog_token = catalog_token
    # _is_compatible already required the provider's digest to agree with (or
    # be blank alongside) the manifest's before an incremental update was
    # chosen -- this just lets a previously-unknown digest (e.g. an index
    # last touched via the GPU backend, which never sets one) get recorded
    # now that a real CPU-backend digest is available, strengthening future
    # checks instead of leaving it blank forever.
    provider_digest = str(getattr(provider, "artifact_digest", "") or "")
    if provider_digest:
        m.embedder_artifact_digest = provider_digest
    m.git_analytics_enabled = bool(git_analytics)
    m.git_max_commits = gitorchestration.coerce_git_max_commits(git_max_commits)
    m.git_fix_regex = fix_regex_value
    m.requested_git_fix_regex = requested_fix_regex_value
    m.chunk_id_scheme = config.CHUNK_ID_SCHEME
    for key, value in git.items():
        setattr(m, key, value)
    manifest.save_project(pdir, m)
    # The index is now fully published and searchable. Only now do we run the
    # heavy shared git-history/SZZ analytics -- it must never delay the
    # moment this incrementally updated generation becomes searchable.
    git_regex = _run_shared_git_analytics_after_publish(
        root=root,
        logical_project_id=logical_project_id,
        checkout_kind=checkout_kind,
        enabled=git_analytics,
        fix_regex=requested_git_fix_regex,
        git_max_commits=git_max_commits,
    )
    # A syntactically valid but unsafe/too-slow custom regex can only be
    # detected by actually running it against the real commit corpus -- the
    # heavy work just deferred above. Reconcile the provisional identity with
    # the one the heavy pass actually resolved, still under the project lock.
    _reconcile_git_fix_regex(
        pdir,
        m,
        git_regex=git_regex,
        provisional_fix_regex=fix_regex_value,
        provisional_requested_fix_regex=requested_fix_regex_value,
        reuse_previous=reuse_previous_fix_regex,
        previous_fix_regex=original_git_fix_regex,
    )

    elapsed = time.time() - t0
    _emit_progress(
        progress,
        "done",
        unit="chunks",
        done=m.chunks,
        total=m.chunks,
        files=len(new_files),
        chunks=m.chunks,
        embedded=embedded,
        reused=reused,
        mode="incremental",
    )
    return IndexStats(
        files=len(new_files), chunks=m.chunks, embedded_unique=embedded,
        reused_unique=reused, seconds=elapsed,
        chunks_per_sec=(len(chunks) / elapsed if elapsed > 0 else 0.0),
        mode="incremental", added=added, changed=changed,
        deleted=len(deleted_paths), unchanged=unchanged,
    )


def reindex_file(root: str | Path, provider: EmbeddingProvider, rel_path: str) -> dict:
    """Re-index (or drop) a single file on the active table — no full walk."""
    root = Path(root).resolve()
    pdir = paths.project_dir(root)
    with paths.project_lock(root):
        m = manifest.load_project(pdir)
        if not _is_compatible(m, provider):
            raise ProjectNotIndexedError(f"project not indexed (or incompatible): {root}")
        abs_path = (root / rel_path).resolve()
        try:
            rel = abs_path.relative_to(root).as_posix()
        except ValueError:
            raise ValueError(f"path escapes project root: {rel_path}")

        # A single-file reindex only ever rewrites this file's own entry, so
        # a corrupt/mismatched files.json must not be silently treated as an
        # empty baseline: that would truncate every other tracked file's
        # entry away, not just fail to notice a delete. Fail loud instead.
        old_files = _load_files_for_indexing(pdir, m)
        if old_files is None:
            raise errors.EngramError(
                "files manifest is missing or does not match the active index",
                errors.E_INDEX_INVALID,
                hint="Run `engram index --rebuild <project_path>` to reestablish the files manifest.",
            )
        store = LanceStore(pdir / "lancedb", provider.dim, table=m.active_table)
        _mark_catalog_stale_for_update(
            pdir=pdir,
            root=root,
            m=m,
            reason="single-file reindex mutating active table",
        )
        store.delete_paths([rel])
        chunks = []
        # Apply the same admission rules as a normal walk (skip symlink, binary,
        # oversized, unreadable, generated); a rejected file is just dropped.
        text = None
        lang = detect_language(abs_path.suffix)
        if abs_path.is_file() and not abs_path.is_symlink():
            ext = abs_path.suffix.lower()
            try:
                size = abs_path.stat().st_size
            except OSError:
                size = 0
            if ext not in config.BINARY_EXTS and 0 < size <= config.MAX_FILE_BYTES:
                candidate = read_text(abs_path)
                if candidate is not None and not looks_generated(candidate, lang):
                    text = candidate
        if text is not None:
            fh = sha256_text(text)
            chunks = chunk_file(rel, lang, text)
            if chunks:
                search_texts = [_search_text(c) for c in chunks]
                with EmbeddingCache(paths.global_cache_dir() / "embeddings.sqlite") as cache:
                    vec_by_hash, hashes, _, _ = _embed(search_texts, provider, cache, config.EMBED_BATCH_SIZE, None)
                store.add(_rows(chunks, search_texts, hashes, vec_by_hash, {rel: fh}))
            st = abs_path.stat()
            old_files[rel] = {
                "file_hash": fh, "mtime_ns": st.st_mtime_ns, "size": st.st_size,
                "language": lang or "", "chunks": len(chunks),
                "indent_complexity": gitanalytics.indentation_complexity(text),
            }
            action = "reindexed"
        else:
            old_files.pop(rel, None)
            action = "deleted"

        store.refresh_fts()
        indexed_at = time.time()
        catalog_rows = _strict_catalog_rows(store)
        git = gitmeta.snapshot(root)
        git_analytics_enabled = bool(getattr(m, "git_analytics_enabled", True))
        logical_project_id = str(git.get("logical_project_id") or m.logical_project_id)
        checkout_kind = str(git.get("checkout_kind") or m.checkout_kind)
        requested_git_fix_regex = getattr(m, "requested_git_fix_regex", None) or getattr(m, "git_fix_regex", None)
        # Resolve the persisted fix-regex identity cheaply (pure regex
        # validation, no git I/O) so the manifest below can be published
        # without waiting on the heavy shared git-history/SZZ computation.
        original_git_fix_regex = getattr(m, "git_fix_regex", None)
        reuse_previous_fix_regex = not git_analytics_enabled
        fix_regex_value, requested_fix_regex_value = _resolve_git_fix_regex_identity(
            requested_git_fix_regex,
            reuse_previous=reuse_previous_fix_regex,
            previous_fix_regex=original_git_fix_regex,
        )
        catalog_token = _save_catalog_from_rows(
            pdir=pdir,
            project_id=m.project_id,
            root=root,
            generation=m.generation,
            active_table=m.active_table or "chunks",
            files_meta=old_files,
            rows=catalog_rows,
            indexed_at=indexed_at,
            store=store,
        )
        manifest.save_files(pdir, old_files, generation=m.generation, active_table=m.active_table or "chunks")
        m.files = len(old_files)
        m.chunks = store.count()
        m.indexed_at = indexed_at
        m.catalog_token = catalog_token
        provider_digest = str(getattr(provider, "artifact_digest", "") or "")
        if provider_digest:
            m.embedder_artifact_digest = provider_digest
        m.git_fix_regex = fix_regex_value
        m.requested_git_fix_regex = requested_fix_regex_value
        m.chunk_id_scheme = config.CHUNK_ID_SCHEME
        for key, value in git.items():
            setattr(m, key, value)
        manifest.save_project(pdir, m)
        # The index is now fully published and searchable. Only now do we run
        # the heavy shared git-history/SZZ analytics -- it must never delay
        # the moment this single-file update becomes searchable.
        git_regex = _run_shared_git_analytics_after_publish(
            root=root,
            logical_project_id=logical_project_id,
            checkout_kind=checkout_kind,
            enabled=git_analytics_enabled,
            fix_regex=requested_git_fix_regex,
            git_max_commits=getattr(m, "git_max_commits", None),
        )
        # A syntactically valid but unsafe/too-slow custom regex can only be
        # detected by actually running it against the real commit corpus --
        # the heavy work just deferred above. Reconcile the provisional
        # identity with the one the heavy pass actually resolved, still
        # under the project lock.
        _reconcile_git_fix_regex(
            pdir,
            m,
            git_regex=git_regex,
            provisional_fix_regex=fix_regex_value,
            provisional_requested_fix_regex=requested_fix_regex_value,
            reuse_previous=reuse_previous_fix_regex,
            previous_fix_regex=original_git_fix_regex,
        )
        return {"rel_path": rel, "action": action, "chunks": len(chunks)}


def remove_project(root: str | Path) -> bool:
    root = Path(root).resolve()
    pdir = paths.project_dir(root, create=False)
    if not pdir.exists():
        return False
    # The lock lives outside the project dir, so we can hold it through the
    # delete; surface rmtree failures instead of silently "succeeding".
    with paths.project_lock(root):
        shutil.rmtree(pdir)
    return True


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None


def derive_chunk_role(
    rel_path: str | None, language: str | None = None, symbol_kind: str | None = None
) -> str:
    """Classify a chunk into a coarse ranking/display role."""

    rel = (rel_path or "").replace("\\", "/")
    lower = rel.lower()
    name = lower.rsplit("/", 1)[-1]
    suffix = Path(name).suffix
    lang = (language or "").lower()
    kind = (symbol_kind or "").lower()

    if name in _CONFIG_NAMES or suffix in _CONFIG_EXTS or "/.github/" in f"/{lower}":
        return "config"
    if (
        "/test/" in f"/{lower}/"
        or "/tests/" in f"/{lower}/"
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name.endswith(".test.js")
        or name.endswith(".test.ts")
        or name.endswith(".spec.js")
        or name.endswith(".spec.ts")
    ):
        return "test"
    if suffix in _TEMPLATE_EXTS or lang in {"html", "css"}:
        return "template"
    if kind in {"comment", "prose", "section", "file"} or lang in {"markdown", "text"}:
        return "comment"
    if any(token in kind for token in _EXECUTABLE_KINDS):
        return "executable"
    if kind == "module":
        return "config" if suffix in _CONFIG_EXTS else "comment"
    return "comment"


def _resolve_git_fix_regex_identity(
    requested_fix_regex: str | None,
    *,
    reuse_previous: bool = False,
    previous_fix_regex: str | None = None,
) -> tuple[str, str]:
    requested = gitmeta.requested_fix_regex_value(requested_fix_regex)
    if reuse_previous and previous_fix_regex:
        return str(previous_fix_regex), requested
    effective, _warnings = regexsafe.pattern_or_default(
        requested_fix_regex,
        gitmeta.DEFAULT_FIX_REGEX,
        label="git_fix_regex",
    )
    return effective, requested


def _run_shared_git_analytics_after_publish(
    *,
    root: Path,
    logical_project_id: str,
    checkout_kind: str,
    enabled: bool,
    fix_regex: str | None,
    git_max_commits: int | None,
) -> dict:
    try:
        return gitorchestration.ensure_shared_git_analytics(
            root=root,
            logical_project_id=logical_project_id,
            checkout_kind=checkout_kind,
            enabled=enabled,
            fix_regex=fix_regex,
            git_max_commits=git_max_commits,
        )
    except Exception as exc:
        effective, requested = _resolve_git_fix_regex_identity(fix_regex)
        return {
            "fix_regex": effective,
            "requested_fix_regex": requested,
            "warnings": [f"shared git analytics unavailable: {exc}"],
        }


def _reconcile_git_fix_regex(
    pdir: Path,
    project_manifest: manifest.ProjectManifest,
    *,
    git_regex: dict,
    provisional_fix_regex: str,
    provisional_requested_fix_regex: str,
    reuse_previous: bool = False,
    previous_fix_regex: str | None = None,
) -> None:
    effective = str(git_regex.get("fix_regex") or provisional_fix_regex)
    requested = str(
        git_regex.get("requested_fix_regex") or provisional_requested_fix_regex
    )
    if reuse_previous and not git_regex.get("fix_regex") and previous_fix_regex:
        effective = str(previous_fix_regex)
    if (
        project_manifest.git_fix_regex == effective
        and project_manifest.requested_git_fix_regex == requested
    ):
        return
    project_manifest.git_fix_regex = effective
    project_manifest.requested_git_fix_regex = requested
    manifest.save_project(pdir, project_manifest)
