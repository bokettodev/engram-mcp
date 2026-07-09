"""Index pipeline: atomic full rebuild + incremental update + hybrid search.

Full rebuild writes a fresh generation table (``chunks_g<N>``) and atomically
swaps the active pointer in ``project.json`` (old table dropped only after the
swap). Incremental updates touch only changed/added/deleted files. Embedding is
deduped via the global content-hash cache. Each chunk is embedded with a
contextual header (path/symbol/language) prepended; the raw content is kept
separately for display. All writers hold a per-project file lock.
"""

from __future__ import annotations

import os
import re
import shutil
import time
import multiprocessing as mp
import threading
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

from engram_mcp import catalog, config, errors, gitanalytics, gitmeta, manifest, paths, retrieval
from engram_mcp.embeddings.base import EmbeddingProvider
from engram_mcp.embeddings.cache import EmbeddingCache
from engram_mcp.indexing.chunker import chunk_file
from engram_mcp.indexing.hash import embedding_input_hash, sha256_text
from engram_mcp.indexing.languages import detect_language, is_valid_language
from engram_mcp.indexing.walker import looks_generated, walk
from engram_mcp.store.lancedb_store import LanceStore


class ProjectNotIndexedError(RuntimeError):
    """Raised when searching a project that has no index on disk yet."""


@dataclass(slots=True)
class IndexStats:
    files: int
    chunks: int
    embedded_unique: int
    reused_unique: int
    seconds: float
    chunks_per_sec: float
    mode: str = "full"  # "full" | "incremental"
    added: int = 0
    changed: int = 0
    deleted: int = 0
    unchanged: int = 0


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


MAX_SEARCH_K = 50
MAX_RERANK_CANDIDATES = 50
DEFAULT_RERANK_CANDIDATE_K = 20
VECTOR_ESTIMATE_RELATIVE_FRACTION = 0.85
DEFAULT_GREP_REGEX_TIMEOUT_SEC = 2.0
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


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None


def _norm_path_key(path: Path) -> str:
    import os

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
        return manifest.load_project(pdir)
    except TypeError:
        return None


def _query_logical_project_id(root: Path, own_manifest: manifest.ProjectManifest | None) -> str:
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


def _manifest_root_path(m: manifest.ProjectManifest) -> Path | None:
    if not m.root_path:
        return None
    try:
        return Path(m.root_path).expanduser().resolve()
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
        m = _load_manifest_lenient(pdir)
        if m is None:
            continue
        if m.logical_project_id != logical_project_id or m.indexed_ref != ref:
            continue
        root = _manifest_root_path(m)
        if root is None:
            continue
        candidates.append((float(m.indexed_at or 0.0), m.project_id, root, pdir, m))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    _, _, root, pdir, m = candidates[0]
    return root, pdir, m


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
    warning = (
        f"no index for ref '{requested_ref}'; searched '{label}'; "
        "index that worktree/branch to search it"
    )
    return requested_root, requested_pdir, (warning,)


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


def _ensure_chunk_role(row: dict) -> dict:
    if row.get("chunk_role"):
        return row
    row["chunk_role"] = derive_chunk_role(
        row.get("rel_path"), row.get("language"), row.get("symbol_kind")
    )
    return row


def _validate_search_k(k: int) -> int:
    if not isinstance(k, int) or not (1 <= k <= MAX_SEARCH_K):
        raise ValueError(f"k must be between 1 and {MAX_SEARCH_K}")
    return k


def rerank_enabled() -> bool:
    """Master switch for reranking. Off unless ENGRAM_RERANK_ENABLED is truthy.

    When off, a per-call ``rerank=true`` is ignored and NO reranker model is
    constructed or downloaded — a hard guarantee that the heavy ONNX cross-encoder
    never loads on the always-on server unless the operator opts in.
    """
    return os.environ.get("ENGRAM_RERANK_ENABLED", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def rerank_candidate_k_default() -> int:
    raw = os.environ.get("ENGRAM_RERANK_CANDIDATE_K", "").strip()
    if not raw:
        value = DEFAULT_RERANK_CANDIDATE_K
    else:
        try:
            value = int(raw)
        except ValueError:
            value = DEFAULT_RERANK_CANDIDATE_K
    return max(1, min(value, MAX_RERANK_CANDIDATES))


def load_query_index(root: str | Path, ref: str | None = None) -> QueryIndex:
    """Validate a project's readable index before any model/provider load."""

    requested_root = Path(root).expanduser().resolve()
    requested_ref = _normalize_ref(ref)
    root, pdir, resolution_warnings = _resolve_query_index(requested_root, requested_ref)
    if not pdir.exists():
        raise ProjectNotIndexedError(f"project not indexed: {requested_root}")
    m = manifest.load_project_strict(pdir)
    if m is None:
        raise ProjectNotIndexedError(f"project not indexed: {requested_root}")
    problems: list[str] = []
    if m.schema_version != manifest.SCHEMA_VERSION:
        problems.append(
            f"schema_version {m.schema_version!r} is incompatible with "
            f"{manifest.SCHEMA_VERSION!r}"
        )
    if not m.logical_project_id:
        problems.append("logical_project_id is missing")
    if m.checkout_kind not in _CHECKOUT_KINDS:
        problems.append("checkout_kind is missing or invalid")
    if not m.active_table:
        problems.append("active_table is missing")
    if not m.embedder_id:
        problems.append("embedder_id is missing")
    if not isinstance(m.dim, int) or m.dim <= 0:
        problems.append("dim must be > 0")
    if m.chunker_version != config.CHUNKER_VERSION:
        problems.append(
            f"chunker_version {m.chunker_version!r} is incompatible with "
            f"{config.CHUNKER_VERSION!r}"
        )
    if not m.root_path:
        problems.append("root_path is missing")
    else:
        try:
            if _norm_path_key(Path(m.root_path)) != _norm_path_key(root):
                problems.append(f"root_path {m.root_path!r} does not match requested root")
        except OSError:
            problems.append(f"root_path {m.root_path!r} could not be resolved")
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
    store = LanceStore(db_dir, m.dim, table=m.active_table or "chunks")
    if not store.exists():
        raise errors.EngramError(
            f"active LanceDB table {m.active_table!r} is missing",
            errors.E_INDEX_INVALID,
            hint="Rebuild the index with `engram index --rebuild <project_path>`.",
        )
    try:
        count = store.count()
    except Exception as exc:
        raise errors.EngramError(
            f"active LanceDB table {m.active_table!r} could not be read",
            errors.E_INDEX_INVALID,
            hint=str(exc),
        ) from exc
    if count <= 0:
        raise errors.EngramError(
            f"active LanceDB table {m.active_table!r} is empty",
            errors.E_INDEX_INVALID,
            hint="Rebuild the index after adding indexable source files.",
        )
    return QueryIndex(
        root=root,
        pdir=pdir,
        manifest=m,
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


def _catalog_ref_count(data: dict) -> int:
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


def _catalog_validation_error(data: dict, qi: QueryIndex) -> str | None:
    if data.get("active_table") != qi.manifest.active_table:
        return "catalog sidecar does not match the active table"
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
    if _catalog_ref_count(data) != table_rows:
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


def _load_valid_catalog(qi: QueryIndex) -> tuple[dict | None, str | None]:
    data = catalog.load_catalog(qi.pdir, qi.manifest.generation)
    if data is None:
        return None, "catalog sidecar unavailable"
    reason = _catalog_validation_error(data, qi)
    if reason is not None:
        return None, reason
    return data, None


def load_project_catalog(root: str | Path) -> tuple[QueryIndex, dict]:
    qi = load_query_index(root)
    data, reason = _load_valid_catalog(qi)
    if data is None:
        raise errors.EngramError(
            f"catalog sidecar for generation {qi.manifest.generation} is missing or invalid",
            errors.E_INDEX_INVALID,
            hint=(reason or "Rebuild the index to regenerate the body-free catalog."),
        )
    return qi, data


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


def _unavailable_git_analytics(
    *,
    group_by: str,
    source_counts: dict | None = None,
    log_ms: float,
    total_start: float,
    warning: str | None = None,
    cache_head: str = "",
    current_head: str = "",
    freshened_commits: int = 0,
) -> dict:
    warnings = [warning] if warning else []
    return {
        "available": False,
        "status": "unavailable",
        "group_by": group_by,
        "analyzed_changes": 0,
        "skipped_changes": 0,
        "cache_head": cache_head,
        "current_head": current_head,
        "freshened_commits": freshened_commits,
        **(source_counts or {}),
        "timings_ms": {
            "log": log_ms,
            "group": 0.0,
            "cochange": 0.0,
            "churn": 0.0,
            "total": _ms_since(total_start),
        },
        "warnings": warnings,
        "szz": _szz_summary(_unavailable_szz_payload(None, warning or "git unavailable")),
        "hotspots": [],
    }


def _coerce_git_max_commits(value: int | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return min(parsed, 1_000_000)


def _limit_commits(commits: list[dict], max_commits: int | None) -> list[dict]:
    limit = _coerce_git_max_commits(max_commits)
    if limit is None:
        return list(commits)
    return list(commits)[:limit]


_SZZ_TASKS: set[threading.Thread] = set()
_SZZ_TASKS_LOCK = threading.Lock()
_SZZ_DEFECT_KEYS = {
    "defect_introducing_commits",
    "defect_introducing_lines",
    "defect_hotspot_score",
}


def _fix_commit_count(commits: list[dict], fix_regex: str) -> int:
    try:
        fix_rx = re.compile(fix_regex or gitmeta.DEFAULT_FIX_REGEX)
    except re.error:
        fix_rx = re.compile(gitmeta.DEFAULT_FIX_REGEX)
    total = 0
    seen: set[str] = set()
    for commit in commits:
        if not isinstance(commit, dict):
            continue
        commit_id = str(commit.get("commit") or "")
        if not commit_id or commit_id in seen:
            continue
        parents = [str(parent) for parent in (commit.get("parents") or []) if str(parent)]
        if len(parents) != 1:
            continue
        if not fix_rx.search(str(commit.get("message") or "")):
            continue
        seen.add(commit_id)
        total += 1
    return total


def _legacy_szz_from_history(history: dict | None) -> dict | None:
    if not isinstance(history, dict):
        return None
    szz = history.get("szz")
    return szz if isinstance(szz, dict) else None


def _szz_sidecar_matches(sidecar: dict, *, history: dict, fix_regex: str) -> bool:
    if str(sidecar.get("head_commit") or "") != str(history.get("head_commit") or ""):
        return False
    if _coerce_git_max_commits(sidecar.get("max_commits")) != _coerce_git_max_commits(
        history.get("max_commits")
    ):
        return False
    return str(sidecar.get("fix_regex") or gitmeta.DEFAULT_FIX_REGEX) == fix_regex


def _wrap_szz_sidecar(
    *,
    root: Path,
    generation: int,
    history: dict,
    szz: dict,
) -> dict:
    payload = dict(szz)
    payload.update(
        {
            "schema_version": catalog.SCHEMA_VERSION,
            "generation": int(generation),
            "root_path": str(root),
            "head_commit": str(history.get("head_commit") or ""),
            "max_commits": _coerce_git_max_commits(history.get("max_commits")),
            "fix_regex": str(payload.get("fix_regex") or history.get("fix_regex") or gitmeta.DEFAULT_FIX_REGEX),
        }
    )
    return payload


def _write_szz_sidecar_if_current(
    *,
    pdir: Path,
    generation: int,
    expected_history: dict,
    payload: dict,
) -> bool:
    current = catalog.load_catalog(pdir, generation)
    history = current.get("git_history") if isinstance(current, dict) else None
    if not isinstance(history, dict) or history.get("status") != "ready":
        return False
    fix_regex = str(expected_history.get("fix_regex") or gitmeta.DEFAULT_FIX_REGEX)
    expected = _wrap_szz_sidecar(
        root=Path(str(current.get("root_path") or "")),
        generation=generation,
        history=expected_history,
        szz=payload,
    )
    if not _szz_sidecar_matches(expected, history=history, fix_regex=fix_regex):
        return False
    catalog.save_szz(pdir, expected)
    return True


def _computing_szz_payload(history: dict, previous_szz: dict | None) -> dict:
    base = dict(previous_szz) if isinstance(previous_szz, dict) else {}
    warnings = [str(w) for w in (base.get("warnings") or []) if str(w)]
    return {
        **base,
        "status": "computing",
        "warning": "",
        "fix_regex": str(history.get("fix_regex") or base.get("fix_regex") or gitmeta.DEFAULT_FIX_REGEX),
        "fix_commits": int(base.get("fix_commits", 0) or 0),
        "blamed_lines": int(base.get("blamed_lines", 0) or 0),
        "attributions": list(base.get("attributions") or []),
        "commit_attributions": dict(base.get("commit_attributions") or {}),
        "workers": gitmeta.szz_worker_count(),
        "cached_commits": int(base.get("cached_commits", 0) or 0),
        "blamed_commits": 0,
        "timings_ms": {"total": 0.0, "blame": 0.0},
        "warnings": warnings[:50],
    }


def _unavailable_szz_payload(history: dict | None, warning: str) -> dict:
    return {
        "status": "unavailable",
        "warning": warning,
        "fix_regex": str((history or {}).get("fix_regex") or gitmeta.DEFAULT_FIX_REGEX),
        "fix_commits": 0,
        "blamed_lines": 0,
        "attributions": [],
        "commit_attributions": {},
        "workers": gitmeta.szz_worker_count(),
        "cached_commits": 0,
        "blamed_commits": 0,
        "timings_ms": {"total": 0.0, "blame": 0.0},
        "warnings": [warning] if warning else [],
    }


def _szz_summary(szz: dict, *, commits: list[dict] | None = None) -> dict:
    fix_regex = str(szz.get("fix_regex") or gitmeta.DEFAULT_FIX_REGEX)
    fix_commits = int(szz.get("fix_commits", 0) or 0)
    if not fix_commits and commits:
        fix_commits = _fix_commit_count(commits, fix_regex)
    attributions = szz.get("attributions") or []
    return {
        "status": str(szz.get("status") or "unavailable"),
        "workers": int(szz.get("workers", gitmeta.szz_worker_count()) or gitmeta.szz_worker_count()),
        "cached_commits": int(szz.get("cached_commits", 0) or 0),
        "blamed_commits": int(szz.get("blamed_commits", 0) or 0),
        "fix_commits": fix_commits,
        "blamed_lines": int(szz.get("blamed_lines", 0) or 0),
        "attributions": len(attributions),
        "timings_ms": dict(szz.get("timings_ms") or {}),
        "warnings": list(szz.get("warnings") or []),
    }


def _szz_for_source(
    *,
    pdir: Path,
    generation: int,
    source: dict,
    history: dict | None,
    commits: list[dict],
) -> dict:
    fix_regex = str(
        source.get("fix_regex")
        or (history or {}).get("fix_regex")
        or gitmeta.DEFAULT_FIX_REGEX
    )
    expected_history = {
        "status": "ready",
        "head_commit": str(source.get("current_head") or source.get("cache_head") or ""),
        "max_commits": source.get("max_commits"),
        "fix_regex": fix_regex,
    }
    sidecar = catalog.load_szz(pdir, generation)
    if isinstance(sidecar, dict) and _szz_sidecar_matches(
        sidecar,
        history=expected_history,
        fix_regex=fix_regex,
    ):
        return sidecar
    legacy = _legacy_szz_from_history(history)
    if isinstance(legacy, dict) and str(source.get("status") or "") == "ready":
        return legacy
    if str(source.get("status") or "") == "unavailable":
        return _unavailable_szz_payload(history, str(source.get("warning") or "git unavailable"))
    return {
        "status": "computing",
        "warning": "",
        "fix_regex": fix_regex,
        "fix_commits": _fix_commit_count(commits, fix_regex),
        "blamed_lines": 0,
        "attributions": [],
        "commit_attributions": {},
        "workers": gitmeta.szz_worker_count(),
        "cached_commits": 0,
        "blamed_commits": 0,
        "timings_ms": {"total": 0.0, "blame": 0.0},
        "warnings": [],
    }


def _strip_defect_signal(hotspot_result: dict) -> dict:
    files = hotspot_result.get("files")
    if isinstance(files, dict):
        for row in files.values():
            if isinstance(row, dict):
                for key in _SZZ_DEFECT_KEYS:
                    row.pop(key, None)
    for item in hotspot_result.get("hotspots") or []:
        if isinstance(item, dict):
            for key in _SZZ_DEFECT_KEYS:
                item.pop(key, None)
    return hotspot_result


def _run_szz_sidecar_task(
    *,
    root: Path,
    pdir: Path,
    generation: int,
    history: dict,
    previous_szz: dict | None,
) -> None:
    commits = [c for c in (history.get("commits") or []) if isinstance(c, dict)]
    fix_regex = str(history.get("fix_regex") or gitmeta.DEFAULT_FIX_REGEX)

    def write_progress(payload: dict) -> None:
        _write_szz_sidecar_if_current(
            pdir=pdir,
            generation=generation,
            expected_history=history,
            payload=payload,
        )

    try:
        szz = gitmeta.szz_attributions_with_status(
            root,
            commits,
            fix_regex=fix_regex,
            previous=previous_szz,
            progress=write_progress,
        )
    except Exception as exc:
        szz = _unavailable_szz_payload(history, f"szz unavailable: {exc}")
    _write_szz_sidecar_if_current(
        pdir=pdir,
        generation=generation,
        expected_history=history,
        payload=szz,
    )


def _schedule_szz_sidecar(
    *,
    root: Path,
    pdir: Path,
    generation: int,
    git_history: dict | None,
    previous_szz: dict | None,
) -> None:
    if not isinstance(git_history, dict) or git_history.get("status") != "ready":
        return
    if not git_history.get("head_commit"):
        return
    computing = _computing_szz_payload(git_history, previous_szz)
    computing["fix_commits"] = _fix_commit_count(
        [c for c in (git_history.get("commits") or []) if isinstance(c, dict)],
        str(computing.get("fix_regex") or gitmeta.DEFAULT_FIX_REGEX),
    )
    _write_szz_sidecar_if_current(
        pdir=pdir,
        generation=generation,
        expected_history=git_history,
        payload=computing,
    )

    def runner() -> None:
        thread = threading.current_thread()
        try:
            _run_szz_sidecar_task(
                root=root,
                pdir=pdir,
                generation=generation,
                history=git_history,
                previous_szz=previous_szz,
            )
        finally:
            with _SZZ_TASKS_LOCK:
                _SZZ_TASKS.discard(thread)

    thread = threading.Thread(target=runner, name=f"engram-szz-g{generation}", daemon=True)
    with _SZZ_TASKS_LOCK:
        _SZZ_TASKS.add(thread)
    thread.start()


def wait_for_szz_tasks(timeout: float | None = None) -> None:
    """Wait for currently scheduled background SZZ tasks.

    Used by tests and measurement scripts; normal indexing does not call this.
    """

    deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
    while True:
        with _SZZ_TASKS_LOCK:
            tasks = [task for task in _SZZ_TASKS if task.is_alive()]
        if not tasks:
            return
        for task in tasks:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining is not None and remaining <= 0:
                return
            task.join(remaining)


def _merge_commit_lists(newer: list[dict], older: list[dict], max_commits: int | None) -> list[dict]:
    seen: set[str] = set()
    merged: list[dict] = []
    for item in list(newer) + list(older):
        if not isinstance(item, dict):
            continue
        commit = str(item.get("commit") or "")
        if not commit or commit in seen:
            continue
        seen.add(commit)
        merged.append(item)
        if max_commits is not None and len(merged) >= max_commits:
            break
    return merged


def _merge_szz_payloads(old_szz: dict | None, new_szz: dict | None) -> dict:
    old_szz = old_szz if isinstance(old_szz, dict) else {}
    new_szz = new_szz if isinstance(new_szz, dict) else {}
    counts: Counter[tuple[str, str, str]] = Counter()
    for payload in (new_szz, old_szz):
        for item in payload.get("attributions") or []:
            if not isinstance(item, dict):
                continue
            fix_commit = str(item.get("fix_commit") or "")
            introducing_commit = str(item.get("introducing_commit") or "")
            path = str(item.get("path") or "").replace("\\", "/")
            if not fix_commit or not introducing_commit or not path:
                continue
            try:
                lines = max(1, int(item.get("lines", 1) or 1))
            except (TypeError, ValueError):
                lines = 1
            counts[(fix_commit, introducing_commit, path)] += lines
    attributions = [
        {
            "fix_commit": fix_commit,
            "introducing_commit": introducing_commit,
            "path": path,
            "lines": lines,
        }
        for (fix_commit, introducing_commit, path), lines in sorted(counts.items())
    ]
    warnings = list(new_szz.get("warnings") or []) + list(old_szz.get("warnings") or [])
    statuses = {str(old_szz.get("status") or ""), str(new_szz.get("status") or "")}
    status = "partial" if "partial" in statuses else "ready"
    if not old_szz and not new_szz:
        status = "unavailable"
    return {
        "status": status,
        "warning": "; ".join(str(w) for w in warnings[:5]),
        "fix_regex": new_szz.get("fix_regex") or old_szz.get("fix_regex") or gitmeta.DEFAULT_FIX_REGEX,
        "fix_commits": len({item["fix_commit"] for item in attributions}),
        "blamed_lines": sum(int(item.get("lines", 0) or 0) for item in attributions),
        "attributions": attributions,
        "warnings": [str(w) for w in warnings[:50]],
    }


def _analytics_source(
    root: Path,
    data: dict,
    *,
    git_max_commits: int | None,
) -> dict:
    t0 = time.perf_counter()
    max_commits = _coerce_git_max_commits(git_max_commits)
    history = data.get("git_history")
    cached_window = (
        _coerce_git_max_commits(history.get("max_commits")) if isinstance(history, dict) else None
    )
    cached_ready_raw = isinstance(history, dict) and history.get("status") == "ready"
    cached_ready = cached_ready_raw and cached_window == max_commits
    cached_commits = [c for c in ((history or {}).get("commits") or []) if isinstance(c, dict)]
    cache_head = str((history or {}).get("head_commit") or "") if isinstance(history, dict) else ""
    current_head = gitmeta.head_commit(root)
    base = {
        "commits": [],
        "status": "unavailable",
        "warning": "",
        "cache_head": cache_head,
        "current_head": current_head,
        "max_commits": max_commits,
        "fix_regex": str((history or {}).get("fix_regex") or gitmeta.DEFAULT_FIX_REGEX),
        "freshened_commits": 0,
        "source_counts": {
            "cached_commits": len(cached_commits) if isinstance(history, dict) else 0,
            "scanned_commits": 0,
        },
        "log_ms": 0.0,
    }
    if not current_head:
        base["warning"] = (
            str(history.get("warning") or "git head unavailable")
            if isinstance(history, dict)
            else "git head unavailable"
        )
        base["log_ms"] = _ms_since(t0)
        return base

    if cached_ready:
        if cache_head == current_head:
            base.update(
                {
                    "commits": _limit_commits(cached_commits, max_commits),
                    "status": "ready",
                    "warning": "",
                    "source_counts": {"cached_commits": len(cached_commits)},
                    "log_ms": _ms_since(t0),
                }
            )
            return base
        if cache_head and gitmeta.is_ancestor(root, cache_head, current_head):
            delta = gitmeta.commit_log_with_status(
                root,
                max_commits=max_commits,
                rev_range=f"{cache_head}..{current_head}",
            )
            if delta.get("status") != "ready":
                base["warning"] = str(delta.get("warning") or "git log unavailable")
                base["log_ms"] = _ms_since(t0)
                return base
            delta_commits = [c for c in (delta.get("commits") or []) if isinstance(c, dict)]
            base.update(
                {
                    "commits": _merge_commit_lists(delta_commits, cached_commits, max_commits),
                    "status": "freshened",
                    "warning": "",
                    "freshened_commits": len(delta_commits),
                    "source_counts": {
                        "cached_commits": len(cached_commits),
                        "scanned_commits": len(delta_commits),
                    },
                    "log_ms": _ms_since(t0),
                }
            )
            return base

        full = gitmeta.commit_log_with_status(root, max_commits=max_commits)
        if full.get("status") != "ready":
            base["warning"] = str(full.get("warning") or "git log unavailable")
            base["log_ms"] = _ms_since(t0)
            return base
        commits = [c for c in (full.get("commits") or []) if isinstance(c, dict)]
        base.update(
            {
                "commits": commits,
                "status": "freshened",
                "warning": "cached git head is not an ancestor of current HEAD",
                "freshened_commits": len(commits),
                "source_counts": {
                    "cached_commits": len(cached_commits),
                    "scanned_commits": len(commits),
                },
                "log_ms": _ms_since(t0),
            }
        )
        return base

    if isinstance(history, dict) and not cached_ready_raw:
        base["warning"] = str(history.get("warning") or "cached git history unavailable")
        base["log_ms"] = _ms_since(t0)
        return base

    status = gitmeta.commit_log_with_status(root, max_commits=max_commits)
    commits = [c for c in (status.get("commits") or []) if isinstance(c, dict)]
    if status.get("status") != "ready":
        base["warning"] = str(status.get("warning") or "git log unavailable")
        base["log_ms"] = _ms_since(t0)
        return base
    base.update(
        {
            "commits": commits,
            "status": "uncached",
            "warning": (
                "cached git history window differs from requested window"
                if cached_ready_raw
                else ""
            ),
            "source_counts": {"scanned_commits": len(commits)},
            "log_ms": _ms_since(t0),
        }
    )
    return base


def _file_git_payload(
    path: str,
    *,
    churn_by_file: dict[str, dict],
    cochanges_by_file: dict[str, list[dict]],
    hotspot_by_file: dict[str, dict],
    include_defects: bool,
) -> dict:
    churn_row = churn_by_file.get(path) or {}
    hotspot = hotspot_by_file.get(path) or {}
    payload = {
        "changes": int(churn_row.get("changes", 0) or 0),
        "churn_lines": int(churn_row.get("churn_lines", 0) or 0),
        "last_touched_ts": int(churn_row.get("last_touched_ts", 0) or 0),
        "fix_density": float(churn_row.get("fix_density", 0.0) or 0.0),
        "complexity": float(hotspot.get("complexity", 0.0) or 0.0),
        "indent_complexity": float(hotspot.get("indent_complexity", 0.0) or 0.0),
        "hotspot_quadrant": hotspot.get("hotspot_quadrant") or "low_churn_low_complexity",
        "cochanges": list(cochanges_by_file.get(path) or []),
    }
    if include_defects:
        payload.update(
            {
                "defect_introducing_commits": int(
                    hotspot.get("defect_introducing_commits", 0) or 0
                ),
                "defect_introducing_lines": int(hotspot.get("defect_introducing_lines", 0) or 0),
                "defect_hotspot_score": int(hotspot.get("defect_hotspot_score", 0) or 0),
            }
        )
    return payload


def _attach_git_analytics(
    out: dict,
    *,
    root: Path,
    pdir: Path,
    data: dict,
    group_by: str,
    ticket_regex: str | None,
    window_hours: float,
    git_max_commits: int | None,
    recent_days: int,
    max_files_per_change: int,
    cochange_limit: int,
    hotspots_limit: int,
) -> dict:
    total_start = time.perf_counter()
    warnings: list[str] = []
    source = _analytics_source(
        root,
        data,
        git_max_commits=git_max_commits,
    )
    source_warning = str(source.get("warning") or "")
    if source.get("status") == "unavailable":
        out["git_analytics"] = _unavailable_git_analytics(
            group_by=group_by,
            source_counts=source.get("source_counts") or {},
            log_ms=float(source.get("log_ms", 0.0) or 0.0),
            total_start=total_start,
            warning=source_warning,
            cache_head=str(source.get("cache_head") or ""),
            current_head=str(source.get("current_head") or ""),
            freshened_commits=int(source.get("freshened_commits", 0) or 0),
        )
        return out
    if source_warning:
        warnings.append(source_warning)
    commits = [c for c in (source.get("commits") or []) if isinstance(c, dict)]
    history = data.get("git_history") if isinstance(data.get("git_history"), dict) else None
    szz = _szz_for_source(
        pdir=pdir,
        generation=int(data.get("generation", 0) or 0),
        source=source,
        history=history,
        commits=commits,
    )
    szz_status = str(szz.get("status") or "unavailable")
    szz_ready = szz_status == "ready"
    fix_regex = str(szz.get("fix_regex") or source.get("fix_regex") or gitmeta.DEFAULT_FIX_REGEX)

    t_group = time.perf_counter()
    grouped = gitanalytics.group_changes_result(
        commits,
        group_by=group_by,
        ticket_regex=ticket_regex,
        window_hours=window_hours,
        max_files_per_change=max_files_per_change,
    )
    change_sets = grouped["change_sets"]
    group_ms = _ms_since(t_group)

    t_cochange = time.perf_counter()
    cochanges_by_file = gitanalytics.cochange(change_sets, limit=cochange_limit)
    cochange_ms = _ms_since(t_cochange)

    t_churn = time.perf_counter()
    churn_by_file = gitanalytics.churn(
        change_sets,
        now_ts=int(time.time()),
        recent_days=recent_days,
        fix_regex=fix_regex,
    )
    defect_by_file = (
        gitanalytics.defect_introductions(szz.get("attributions") or []) if szz_ready else {}
    )
    hotspot_result = gitanalytics.hotspots(
        churn_by_file,
        data.get("files") or [],
        limit=hotspots_limit,
        defect_by_file=defect_by_file,
    )
    if not szz_ready:
        hotspot_result = _strip_defect_signal(hotspot_result)
    churn_ms = _ms_since(t_churn)
    hotspot_by_file = hotspot_result.get("files") or {}

    for row in out.get("files") or []:
        path = (row.get("path") or "").replace("\\", "/")
        row["git"] = _file_git_payload(
            path,
            churn_by_file=churn_by_file,
            cochanges_by_file=cochanges_by_file,
            hotspot_by_file=hotspot_by_file,
            include_defects=szz_ready,
        )

    out["git_analytics"] = {
        "available": True,
        "status": source.get("status") or "ready",
        "group_by": grouped.get("group_by") or group_by,
        "analyzed_changes": len(change_sets),
        "skipped_changes": int(grouped.get("skipped_changes", 0) or 0),
        "cache_head": str(source.get("cache_head") or ""),
        "current_head": str(source.get("current_head") or ""),
        "freshened_commits": int(source.get("freshened_commits", 0) or 0),
        **(source.get("source_counts") or {}),
        "szz": _szz_summary(szz, commits=commits),
        "timings_ms": {
            "log": float(source.get("log_ms", 0.0) or 0.0),
            "group": group_ms,
            "cochange": cochange_ms,
            "churn": churn_ms,
            "total": _ms_since(total_start),
        },
        "warnings": warnings,
        "hotspots": list(hotspot_result.get("hotspots") or []),
    }
    return out


def project_map(
    root: str | Path,
    depth: int = 2,
    sort: str = "path",
    dirs_limit: int | None = 200,
    dirs_offset: int = 0,
    include_files: bool = False,
    files_limit: int | None = 50,
    files_offset: int = 0,
    include_symbols: bool = False,
    symbols_limit: int | None = 20,
    code_only: bool = False,
    languages: list[str] | None = None,
    chunk_roles: list[str] | None = None,
    kinds: list[str] | None = None,
    path_prefix: str | None = None,
    path_glob: str | None = None,
    symbol_kinds: list[str] | None = None,
    min_symbols: int = 0,
    non_empty: bool = True,
    include_git: bool = True,
    group_by: str = "commit",
    ticket_regex: str | None = None,
    window_hours: float = 2.0,
    git_max_commits: int | None = None,
    recent_days: int = 90,
    max_files_per_change: int = 50,
    cochange_limit: int = 5,
    hotspots_limit: int = 25,
) -> dict:
    """Return a body-free project map from the catalog sidecar."""

    qi, data = load_project_catalog(root)
    out = catalog.project_map(
        data,
        depth=depth,
        sort=sort,
        dirs_limit=dirs_limit,
        dirs_offset=dirs_offset,
        include_files=include_files,
        files_limit=files_limit,
        files_offset=files_offset,
        include_symbols=include_symbols,
        symbols_limit=symbols_limit,
        code_only=code_only,
        languages=languages,
        chunk_roles=chunk_roles,
        kinds=kinds,
        path_prefix=path_prefix,
        path_glob=path_glob,
        symbol_kinds=symbol_kinds,
        min_symbols=min_symbols,
        non_empty=non_empty,
    )
    if not include_git:
        return out
    return _attach_git_analytics(
        out,
        root=qi.root,
        pdir=qi.pdir,
        data=data,
        group_by=group_by,
        ticket_regex=ticket_regex,
        window_hours=window_hours,
        git_max_commits=git_max_commits,
        recent_days=recent_days,
        max_files_per_change=max_files_per_change,
        cochange_limit=cochange_limit,
        hotspots_limit=hotspots_limit,
    )


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


def _is_compatible(m: manifest.ProjectManifest | None, provider: EmbeddingProvider) -> bool:
    return bool(
        m is not None
        and m.schema_version == manifest.SCHEMA_VERSION
        and m.logical_project_id
        and m.checkout_kind in _CHECKOUT_KINDS
        and m.active_table
        and m.embedder_id == provider.model_id
        and m.dim == provider.dim
        and m.chunker_version == config.CHUNKER_VERSION
    )


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
    git_history: dict | None = None,
) -> None:
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
        git_history=git_history,
    )
    if _catalog_ref_count(data) != table_rows:
        raise RuntimeError(
            f"refusing to write catalog: catalog refs={_catalog_ref_count(data)} table rows={table_rows}"
        )
    catalog.save_catalog(pdir, data)


def _maybe_git_history_for_catalog(
    root: Path,
    *,
    enabled: bool,
    max_commits: int | None,
    previous: dict | None = None,
    head: str | None = None,
    fix_regex: str | None = None,
) -> dict | None:
    if not enabled:
        return None
    try:
        return gitmeta.history_for_catalog(
            root,
            max_commits=max_commits,
            previous=previous,
            head=head,
            fix_regex=fix_regex,
        )
    except Exception as exc:
        return {
            "schema_version": 1,
            "status": "unavailable",
            "max_commits": _coerce_git_max_commits(max_commits),
            "head_commit": head or "",
            "fix_regex": fix_regex or gitmeta.DEFAULT_FIX_REGEX,
            "commits": [],
            "warning": f"git history unavailable: {exc}",
        }


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
    hashes = [embedding_input_hash(provider.model_id, config.CHUNKER_VERSION, t) for t in texts]
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
    embedding cache. It never constructs an embedding provider.
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
        and (pdir / "files.json").is_file()
    )
    old_files = manifest.load_files(pdir) if compatible else {}
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
        text = _read_text(rec.abs_path)
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
    out = []
    for idx, (c, st, h) in enumerate(zip(chunks, search_texts, hashes)):
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
        compatible = _is_compatible(m, provider) and (pdir / "files.json").is_file()
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
        text = _read_text(rec.abs_path)
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
    previous_szz = None
    if git_analytics and m is not None:
        previous_szz = catalog.load_szz(pdir, m.generation)
        if previous_szz is None:
            old_catalog = catalog.load_catalog(pdir, m.generation)
            old_history = old_catalog.get("git_history") if isinstance(old_catalog, dict) else None
            previous_szz = _legacy_szz_from_history(old_history)
    new_table = f"chunks_g{gen}"
    db_dir = pdir / "lancedb"
    # GC tables left over from prior interrupted runs, but keep the current
    # active table so concurrent readers on the old pointer don't break.
    keep = {m.active_table} if m and m.active_table else set()
    LanceStore(db_dir, provider.dim).drop_stale_generations(keep)
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
    git_history = _maybe_git_history_for_catalog(
        root,
        enabled=git_analytics,
        max_commits=git_max_commits,
        previous=None,
        head=git.get("indexed_commit") or None,
        fix_regex=git_fix_regex,
    )
    _save_catalog_from_rows(
        pdir=pdir,
        project_id=paths.project_id_for(root),
        root=root,
        generation=gen,
        active_table=new_table,
        files_meta=files_meta,
        rows=rows,
        indexed_at=indexed_at,
        store=store,
        git_history=git_history,
    )
    # Commit the pointer FIRST (it references the fully-built new table), then
    # the file manifest. A crash in between leaves the new table active + a
    # stale files manifest, which the next incremental reconciles cheaply.
    manifest.save_project(
        pdir,
        manifest.ProjectManifest(
            project_id=paths.project_id_for(root), root_path=str(root),
            active_table=new_table, generation=gen,
            embedder_id=provider.model_id, dim=provider.dim,
            chunker_version=config.CHUNKER_VERSION,
            files=len(files_meta), chunks=len(chunks), indexed_at=indexed_at,
            **git,
        ),
    )
    manifest.save_files(pdir, files_meta)
    _schedule_szz_sidecar(
        root=root,
        pdir=pdir,
        generation=gen,
        git_history=git_history if git_analytics else None,
        previous_szz=previous_szz,
    )
    # The previous active table is intentionally retained for in-flight readers;
    # it is GC'd at the start of the next rebuild.

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
    *,
    git_analytics: bool,
    git_max_commits: int | None,
    git_fix_regex: str | None,
) -> IndexStats:
    t0 = time.time()
    old_catalog = catalog.load_catalog(pdir, m.generation) if git_analytics else None
    old_git_history = old_catalog.get("git_history") if isinstance(old_catalog, dict) else None
    previous_szz = catalog.load_szz(pdir, m.generation) if git_analytics else None
    if previous_szz is None:
        previous_szz = _legacy_szz_from_history(old_git_history)
    old_files = manifest.load_files(pdir)
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
                text = _read_text(rec.abs_path)
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
        text = _read_text(rec.abs_path)
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
    git_history = _maybe_git_history_for_catalog(
        root,
        enabled=git_analytics,
        max_commits=git_max_commits,
        previous=old_git_history if isinstance(old_git_history, dict) else None,
        head=git.get("indexed_commit") or None,
        fix_regex=git_fix_regex,
    )
    catalog_rows = _strict_catalog_rows(store)
    _save_catalog_from_rows(
        pdir=pdir,
        project_id=m.project_id,
        root=root,
        generation=m.generation,
        active_table=m.active_table or "chunks",
        files_meta=new_files,
        rows=catalog_rows,
        indexed_at=indexed_at,
        store=store,
        git_history=git_history,
    )
    manifest.save_files(pdir, new_files)
    m.files = len(new_files)
    m.chunks = store.count()
    m.indexed_at = indexed_at
    for key, value in git.items():
        setattr(m, key, value)
    manifest.save_project(pdir, m)
    _schedule_szz_sidecar(
        root=root,
        pdir=pdir,
        generation=m.generation,
        git_history=git_history if git_analytics else None,
        previous_szz=previous_szz,
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

        old_files = manifest.load_files(pdir)
        old_catalog = catalog.load_catalog(pdir, m.generation)
        old_git_history = old_catalog.get("git_history") if isinstance(old_catalog, dict) else None
        previous_szz = catalog.load_szz(pdir, m.generation)
        if previous_szz is None:
            previous_szz = _legacy_szz_from_history(old_git_history)
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
                candidate = _read_text(abs_path)
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
        git_history = _maybe_git_history_for_catalog(
            root,
            enabled=True,
            max_commits=(old_git_history or {}).get("max_commits") if isinstance(old_git_history, dict) else None,
            previous=old_git_history if isinstance(old_git_history, dict) else None,
            head=git.get("indexed_commit") or None,
            fix_regex=(
                str(old_git_history.get("fix_regex") or "")
                if isinstance(old_git_history, dict)
                else None
            ),
        )
        _save_catalog_from_rows(
            pdir=pdir,
            project_id=m.project_id,
            root=root,
            generation=m.generation,
            active_table=m.active_table or "chunks",
            files_meta=old_files,
            rows=catalog_rows,
            indexed_at=indexed_at,
            store=store,
            git_history=git_history,
        )
        manifest.save_files(pdir, old_files)
        m.files = len(old_files)
        m.chunks = store.count()
        m.indexed_at = indexed_at
        for key, value in git.items():
            setattr(m, key, value)
        manifest.save_project(pdir, m)
        _schedule_szz_sidecar(
            root=root,
            pdir=pdir,
            generation=m.generation,
            git_history=git_history,
            previous_szz=previous_szz,
        )
        return {"rel_path": rel, "action": action, "chunks": len(chunks)}


def _freshness_for_hit(root: Path, files_meta: dict[str, dict], hit: dict) -> tuple[bool, str]:
    rel = hit.get("rel_path") or ""
    meta = files_meta.get(rel)
    if not meta:
        return True, "file is not present in files.json"
    abs_path = root / rel
    try:
        st = abs_path.stat()
    except OSError:
        return True, "file is missing from the working tree"
    if st.st_size == meta.get("size") and st.st_mtime_ns == meta.get("mtime_ns"):
        return False, "mtime and size match the indexed file"
    if st.st_size <= config.MAX_FILE_BYTES:
        text = _read_text(abs_path)
        if text is not None and sha256_text(text) == meta.get("file_hash"):
            return False, "content hash matches despite metadata drift"
    return True, "working-tree file differs from the indexed file"


def _annotate_hits(
    root: Path,
    pdir: Path,
    hits: list[dict],
    query: str,
    mode_used: str,
    min_relevance: str | None,
) -> tuple[list[dict], dict, dict]:
    files_meta = manifest.load_files(pdir)
    annotated: list[dict] = []
    stale_paths: set[str] = set()
    for rank, hit in enumerate(hits, 1):
        h = _ensure_chunk_role(dict(hit))
        raw = float(h.get("score", 0.0))
        normalized = retrieval.normalize_score(
            raw, mode_used, rank, reranked=bool(h.get("reranked"))
        )
        relevance = retrieval.relevance_bucket(normalized)
        h["raw_score"] = raw
        h["score"] = raw
        h["score_normalized"] = normalized
        h["relevance"] = relevance
        h["matched"] = relevance in {"high", "medium"}
        h["match_reason"] = retrieval.match_reason(h, query, mode_used)
        stale, reason = _freshness_for_hit(root, files_meta, h)
        h["stale"] = stale
        h["index_stale"] = stale
        h["freshness_reason"] = reason
        if stale:
            stale_paths.add(h.get("rel_path") or "")
        if retrieval.relevance_at_least(relevance, min_relevance):
            annotated.append(h)

    dirty = {
        "stale": bool(stale_paths),
        "stale_results": len(stale_paths),
        "stale_paths": sorted(p for p in stale_paths if p),
    }
    tail = {
        "tail_weak": any(
            h.get("relevance") in {"low", "uncertain"} for h in annotated[3:]
        ),
        "tail_weak_after_rank": 3 if len(annotated) > 3 else None,
    }
    if not tail["tail_weak"]:
        tail["tail_weak_after_rank"] = None
    return annotated, dirty, tail


def _parse_facets(facets: list[str] | tuple[str, ...] | None) -> list[str]:
    if not facets:
        return []
    out: list[str] = []
    for item in facets:
        if item not in catalog.SUPPORTED_FACETS:
            raise ValueError(
                "facets must be drawn from: " + ", ".join(sorted(catalog.SUPPORTED_FACETS))
            )
        if item not in out:
            out.append(item)
    return out


def _vector_similarity(hit: dict) -> float:
    if "_distance" in hit:
        distance = max(0.0, float(hit.get("_distance") or 0.0))
        return 1.0 / (1.0 + distance)
    raw = float(hit.get("score", 0.0) or 0.0)
    return max(0.0, raw)


def _vector_candidate_estimate(candidates: list[dict]) -> dict:
    if not candidates:
        return {
            "count": 0,
            "exact": False,
            "scope": "candidate_pool",
            "candidate_count": 0,
            "relative_score_fraction": VECTOR_ESTIMATE_RELATIVE_FRACTION,
        }
    sims = [_vector_similarity(h) for h in candidates]
    top = max(sims)
    threshold = top * VECTOR_ESTIMATE_RELATIVE_FRACTION
    return {
        "count": sum(1 for s in sims if s >= threshold),
        "exact": False,
        "scope": "candidate_pool",
        "candidate_count": len(candidates),
        "relative_score_fraction": VECTOR_ESTIMATE_RELATIVE_FRACTION,
        "top_similarity": round(top, 6),
        "threshold": round(threshold, 6),
    }


def _metadata_facet_counts(rows: list[dict], catalog_data: dict | None, requested: list[str]) -> dict:
    if not requested:
        return {}
    counts: dict[str, dict[str, int]] = {f: {} for f in requested}
    by_path = catalog.file_by_path(catalog_data) if catalog_data else {}

    def inc(field: str, value: str, n: int = 1) -> None:
        if field not in counts:
            return
        bucket = value or "(none)"
        counts[field][bucket] = counts[field].get(bucket, 0) + n

    seen_paths: set[str] = set()
    for row in rows:
        rel = row.get("rel_path") or ""
        file_meta = by_path.get(rel, {})
        if "chunk_role" in counts:
            inc("chunk_role", row.get("chunk_role") or "(none)")
        if rel in seen_paths:
            continue
        seen_paths.add(rel)
        if "dir" in counts:
            inc("dir", file_meta.get("dir") or catalog._dir_for(rel))
        if "language" in counts:
            inc("language", row.get("language") or file_meta.get("language") or "(none)")
        if "kind" in counts:
            for kind in file_meta.get("kinds") or catalog.derive_file_kinds(rel, row.get("language")):
                value = kind.get("kind") or ""
                if value:
                    inc("kind", value)
    return {field: dict(sorted(values.items())) for field, values in counts.items()}


def _search_count_metadata(
    qi: QueryIndex,
    query: str,
    where: str | None,
    mode_used: str,
    vector_candidates: list[dict],
    requested_facets: list[str],
    catalog_data: dict | None,
) -> tuple[dict, dict | None, list[str]]:
    warnings: list[str] = []
    total = {
        "fts_exact": {
            "available": False,
            "count": None,
            "exact": True,
            "capped": False,
            "method": "lancedb_0_33_fts_metadata_scan",
        },
        "vector_estimate": _vector_candidate_estimate(vector_candidates),
    }
    facet_payload: dict | None = None
    fts_rows: list[dict] = []
    fts_exact = False
    if mode_used == "hybrid":
        fts_rows, warning, fts_meta = qi.store.fts_metadata(
            query,
            columns=("chunk_id", "rel_path", "language", "chunk_role"),
            where=where,
        )
        if warning:
            warnings.append(warning)
        else:
            capped = bool(fts_meta.get("capped"))
            fts_exact = not capped
            total["fts_exact"] = {
                "available": True,
                "count": len(fts_rows),
                "exact": fts_exact,
                "capped": capped,
                "method": "lancedb_0_33_fts_metadata_scan",
            }
            if capped:
                cap = fts_meta.get("cap", len(fts_rows))
                warnings.append(
                    f"FTS metadata scan hit ENGRAM_FTS_COUNT_MAX_SCAN={cap}; "
                    "total_matches.fts_exact.count is a lower bound."
                )
    if requested_facets:
        if mode_used == "hybrid" and total["fts_exact"]["available"]:
            facet_payload = {
                "scope": "fts_exact" if fts_exact else "fts_capped_lower_bound",
                "exact": fts_exact,
                "fields": _metadata_facet_counts(fts_rows, catalog_data, requested_facets),
            }
        else:
            estimated_rows = [
                row for row in vector_candidates
                if _vector_similarity(row) >= (
                    (total["vector_estimate"].get("threshold") or 0.0)
                )
            ]
            facet_payload = {
                "scope": "vector_candidate_estimate",
                "exact": False,
                "fields": _metadata_facet_counts(estimated_rows, catalog_data, requested_facets),
            }
    return total, facet_payload, warnings


def search_project(
    root: str | Path,
    provider: EmbeddingProvider,
    query: str,
    k: int = 8,
    language: str | None = None,
    mode: str = "auto",
    candidate_k: int | None = None,
    rerank: bool = False,
    facets: list[str] | tuple[str, ...] | None = None,
    min_relevance: str | None = None,
    return_meta: bool = False,
    ref: str | None = None,
    _query_index: QueryIndex | None = None,
) -> list[dict] | dict:
    requested_root = Path(root).resolve()
    k = _validate_search_k(k)
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if language is not None and not is_valid_language(language):
        raise ValueError(f"unknown language filter: {language!r}")
    if mode not in ("auto", "hybrid", "vector"):
        raise ValueError(f"unknown search mode: {mode!r}")
    requested_facets = _parse_facets(facets)
    if min_relevance is not None and min_relevance not in retrieval.RELEVANCE_ORDER:
        retrieval.relevance_at_least("low", min_relevance)  # raises the canonical message
    resolved_mode = retrieval.classify_query(query) if mode == "auto" else mode
    qi = _query_index or load_query_index(requested_root, ref=ref)
    root = qi.root
    m = qi.manifest
    if m.embedder_id and m.embedder_id != provider.model_id:
        raise ValueError(
            f"index built with a different embedder ({m.embedder_id}); rebuild the index"
        )
    if m.dim != provider.dim:
        raise errors.EngramError(
            f"index dimension {m.dim} does not match provider dimension {provider.dim}",
            errors.E_INDEX_INVALID,
            hint="Rebuild the index with the recorded embedder.",
        )

    where = f"language = '{language}'" if language else None  # language is whitelisted above
    qv = provider.embed_queries([query])[0]
    if candidate_k is None:
        candidate_k = rerank_candidate_k_default()
    if not isinstance(candidate_k, int):
        raise ValueError("candidate_k must be an integer")
    candidate_k = max(k, min(candidate_k, MAX_RERANK_CANDIDATES))
    n = candidate_k
    warnings: list[str] = list(qi.resolution_warnings)
    mode_used = resolved_mode
    vector_candidates: list[dict] = []
    if resolved_mode == "hybrid":
        hits, meta = retrieval.hybrid_search(
            qi.store, query, qv, k=n, where=where, candidate_k=candidate_k, return_meta=True
        )
        warnings.extend(meta["warnings"])
        mode_used = meta["mode_used"]
        vector_candidates = [dict(h) for h in meta.get("vector_candidates", [])]
    else:
        hits = qi.store.search(qv, k=n, where=where)
        vector_candidates = [dict(h) for h in hits]
        for h in hits:
            _ensure_chunk_role(h)
            h["score"] = -float(h.get("_distance", 0.0)) + retrieval._role_boost(
                h.get("chunk_role")
            )
        hits = sorted(hits, key=lambda h: h.get("score", 0.0), reverse=True)
    rerank_applied = False
    rerank_model = None
    rerank_latency_ms = None
    rerank_skipped_reason = None
    if rerank and not rerank_enabled():
        # Master switch: reranking must be explicitly enabled by the operator.
        # Off by default so a stray rerank=true never loads the ONNX model.
        rerank_skipped_reason = "reranking disabled (set ENGRAM_RERANK_ENABLED=1 to enable)"
        warnings.append(f"rerank skipped: {rerank_skipped_reason}")
        hits = hits[:k]
    elif rerank and mode_used != "vector":
        # Skip hybrid rerank because measured quality regresses for exact
        # identifier/literal queries; callers can force mode="vector".
        rerank_skipped_reason = f"mode_used={mode_used} (rerank applies to vector mode only)"
        warnings.append(f"rerank skipped: {rerank_skipped_reason}")
        hits = hits[:k]
    elif rerank:
        t_rerank = time.time()
        try:
            from engram_mcp.rerankers import get_reranker

            reranker = get_reranker(backend="fastembed")
            hits = reranker.rerank(query, hits, top_k=k)
            rerank_applied = True
            rerank_model = getattr(reranker, "model_id", None)
        except Exception as exc:
            # Rerank is best-effort: on ANY failure (missing extra, model
            # download race, onnxruntime error) degrade to the base ranking
            # instead of failing the whole search.
            warnings.append(f"rerank unavailable, returning base ranking: {exc}")
            hits = hits[:k]
        finally:
            rerank_latency_ms = round((time.time() - t_rerank) * 1000, 3)
    else:
        hits = hits[:k]

    indexed_git = {
        "git_worktree_root": m.git_worktree_root,
        "indexed_ref": m.indexed_ref,
        "indexed_commit": m.indexed_commit,
        "indexed_dirty": m.indexed_dirty,
    }
    try:
        git_status = gitmeta.current_staleness(root, indexed_git)
    except Exception as exc:
        git_status = {
            "available": False,
            "git_stale": False,
            "reasons": [],
            "indexed": indexed_git,
            "current": {},
            "warning": f"git metadata unavailable: {exc}",
        }
        warnings.append(f"git metadata unavailable: {exc}")
    source_revision = gitmeta.source_revision_from_staleness(git_status)
    revision_warning = gitmeta.source_revision_warning(source_revision)
    if revision_warning:
        warnings.append(revision_warning)
    annotated, dirty, tail = _annotate_hits(
        root, qi.pdir, hits[:k], query, mode_used, min_relevance,
    )
    catalog_data, catalog_reason = _load_valid_catalog(qi)
    if catalog_data is None:
        if catalog_reason == "catalog sidecar unavailable":
            warnings.append("catalog sidecar unavailable; project_map/facets may require rebuild")
        else:
            warnings.append(
                f"catalog sidecar unavailable; project_map/facets may require rebuild ({catalog_reason})"
            )
    total_matches, facet_payload, count_warnings = _search_count_metadata(
        qi,
        query,
        where,
        mode_used,
        vector_candidates,
        requested_facets,
        catalog_data,
    )
    warnings.extend(count_warnings)
    if not return_meta:
        return annotated
    return {
        "query": query,
        "project_path": str(root),
        "requested_project_path": str(qi.requested_root or requested_root),
        "project_id": m.project_id,
        "logical_project_id": m.logical_project_id,
        "checkout_kind": m.checkout_kind,
        "indexed_ref": m.indexed_ref,
        "requested_ref": qi.requested_ref,
        "index_generation": m.generation,
        "embedder_id": m.embedder_id,
        "source_type": "static_indexed_source",
        "mode_requested": mode,
        "mode_used": mode_used,
        "warnings": warnings,
        "rerank_requested": rerank,
        "rerank_applied": rerank_applied,
        "rerank_skipped_reason": rerank_skipped_reason,
        "rerank_model": rerank_model,
        "rerank_latency_ms": rerank_latency_ms,
        "candidate_k": candidate_k,
        "facets_requested": requested_facets,
        "facets": facet_payload,
        "total_matches": total_matches,
        "dirty": dirty,
        "index_stale": dirty["stale"],
        "source_revision": source_revision,
        **tail,
        "hits": annotated,
    }


def _symbol_suggestions(store: LanceStore, name: str, limit: int = 8) -> list[dict]:
    needle = name.lower()
    seen: set[str] = set()
    suggestions: list[tuple[float, dict]] = []
    for row in store.symbol_inventory():
        sym = row.get("symbol") or ""
        if not sym or sym in seen:
            continue
        seen.add(sym)
        low = sym.lower()
        leaf = low.rsplit(".", 1)[-1]
        score = SequenceMatcher(None, needle, low).ratio()
        if leaf.startswith(needle) or low.startswith(needle):
            score += 0.45
        elif needle in low:
            score += 0.30
        elif needle in leaf:
            score += 0.20
        if score < 0.45:
            continue
        item = dict(row)
        item["score"] = round(min(score, 1.0), 3)
        suggestions.append((score, item))
    suggestions.sort(key=lambda x: (-x[0], x[1].get("symbol", "")))
    return [s for _, s in suggestions[:limit]]


def find_definition(
    root: str | Path,
    name: str,
    k: int = 20,
    include_suggestions: bool = False,
    ref: str | None = None,
) -> list[dict] | dict:
    """Exact symbol lookup (no embedding): definitions named `name` or `Parent.name`.

    Returns whole-symbol chunks (path + line range + content), preferring real
    definitions over module-level chunks.
    """
    if not name or not name.strip():
        raise ValueError("symbol must not be empty")
    qi = load_query_index(root, ref=ref)
    rows = qi.store.by_symbol(name, k=k)
    for row in rows:
        _ensure_chunk_role(row)
    defs = [r for r in rows if r.get("symbol_kind") not in ("module", "file")]
    results = defs or rows
    if not include_suggestions:
        return results
    return {
        "symbol": name,
        "project_path": str(qi.root),
        "requested_project_path": str(qi.requested_root or qi.root),
        "project_id": qi.manifest.project_id,
        "logical_project_id": qi.manifest.logical_project_id,
        "checkout_kind": qi.manifest.checkout_kind,
        "indexed_ref": qi.manifest.indexed_ref,
        "requested_ref": qi.requested_ref,
        "source_type": "static_indexed_source",
        "warnings": list(qi.resolution_warnings),
        "count": len(results),
        "results": results,
        "suggestions": [] if results else _symbol_suggestions(qi.store, name),
    }


def get_chunk(
    root: str | Path,
    chunk_id: str,
    *,
    include_neighbors: bool = False,
    neighbor_window: int = 1,
    include_parent: bool = False,
) -> dict:
    """Fetch the full stored content for one chunk id."""

    if not chunk_id or not chunk_id.strip():
        raise ValueError("chunk_id must not be empty")
    qi = load_query_index(root)
    row = qi.store.by_chunk_id(chunk_id)
    if row is None:
        raise ValueError(f"unknown chunk_id: {chunk_id}")
    _ensure_chunk_role(row)
    stale, reason = _freshness_for_hit(qi.root, manifest.load_files(qi.pdir), row)
    row["stale"] = stale
    row["freshness_reason"] = reason
    row["project_path"] = str(qi.root)
    row["project_id"] = qi.manifest.project_id
    row["index_generation"] = qi.manifest.generation
    row["source_type"] = "static_indexed_source"
    if include_neighbors or include_parent:
        data, reason = _load_valid_catalog(qi)
        if data is None:
            detail = f": {reason}" if reason else ""
            row["warnings"] = [f"catalog sidecar unavailable; cannot expand neighborhood{detail}"]
            return row
        lookup = catalog.chunk_lookup(data)
        current = lookup.get(chunk_id)
        if current is None:
            row["warnings"] = ["chunk not found in catalog sidecar; cannot expand neighborhood"]
            return row
        file_entry, chunk_entry, idx = current
        refs = file_entry.get("chunk_refs") or []
        neighbor_window = max(0, min(int(neighbor_window), 5))
        if include_neighbors and neighbor_window:
            start = max(0, idx - neighbor_window)
            end = min(len(refs), idx + neighbor_window + 1)
            neighbors = []
            for nidx in range(start, end):
                ref = refs[nidx]
                cid = ref.get("chunk_id")
                if not cid or cid == chunk_id:
                    continue
                body = qi.store.by_chunk_id(cid)
                if body is None:
                    continue
                _ensure_chunk_role(body)
                body["relative_position"] = nidx - idx
                neighbors.append(body)
            row["neighbors"] = neighbors
        if include_parent:
            symbol = chunk_entry.get("symbol") or ""
            parent_symbol = symbol.rsplit(".", 1)[0] if "." in symbol else ""
            parent = None
            if parent_symbol:
                for ref in refs:
                    if ref.get("symbol") == parent_symbol and ref.get("chunk_id") != chunk_id:
                        parent = qi.store.by_chunk_id(ref.get("chunk_id") or "")
                        if parent is not None:
                            _ensure_chunk_role(parent)
                        break
            row["parent"] = parent
    return row


def doctor_project(root: str | Path, *, check_git: bool = True) -> dict:
    """Read-only index health check. Does not load the embedding model."""

    root = Path(root).expanduser().resolve()
    pdir = paths.project_dir(root, create=False)
    issues: list[dict] = []
    if not pdir.exists():
        raise ProjectNotIndexedError(f"project not indexed: {root}")

    def issue(code: str, severity: str, message: str, hint: str | None = None) -> None:
        item = {"code": code, "severity": severity, "message": message}
        if hint:
            item["hint"] = hint
        issues.append(item)

    try:
        m = manifest.load_project_strict(pdir)
    except errors.EngramError as exc:
        issue(exc.code, "error", str(exc), exc.hint)
        return {
            "project_path": str(root),
            "source_type": "static_indexed_source",
            "ok": False,
            "summary": {"issues": len(issues), "errors": 1, "warnings": 0},
            "issues": issues,
        }
    if m is None:
        raise ProjectNotIndexedError(f"project not indexed: {root}")

    if m.root_path and not Path(m.root_path).exists():
        issue("root_missing", "error", "manifest root_path no longer exists")
    if m.embedder_id != "fastembed:ibm-granite/granite-embedding-97m-multilingual-r2":
        issue("model_drift", "error", f"manifest embedder_id is {m.embedder_id!r}")
    if m.chunker_version != config.CHUNKER_VERSION:
        issue("chunker_drift", "error", f"manifest chunker_version is {m.chunker_version!r}")
    if m.schema_version != manifest.SCHEMA_VERSION:
        issue("manifest_schema", "error", f"manifest schema_version is {m.schema_version!r}")

    files_meta = manifest.load_files(pdir)
    if len(files_meta) != m.files:
        issue("files_manifest_mismatch", "warning", "files.json count differs from project manifest")

    table_rows = None
    expected_schema = set(LanceStore(pdir / "lancedb", max(1, m.dim or 1), table=m.active_table or "chunks")._schema.names)
    store = LanceStore(pdir / "lancedb", max(1, m.dim or 1), table=m.active_table or "chunks")
    if not store.exists():
        issue("table_missing", "error", f"active table {m.active_table!r} is missing")
    else:
        try:
            table_rows = store.count()
            if table_rows != m.chunks:
                issue("table_count_mismatch", "error", "active table row count differs from manifest chunks")
        except Exception as exc:
            issue("table_unreadable", "error", str(exc))
        schema_names = set(store.schema_names())
        missing_cols = sorted(expected_schema - schema_names)
        if missing_cols:
            issue("table_schema_mismatch", "error", "active table is missing columns: " + ", ".join(missing_cols))
        _rows, fts_warning = store.search_text_with_status("engram", k=1)
        if fts_warning:
            issue("fts_unavailable", "warning", fts_warning)

    cat = catalog.load_catalog(pdir, m.generation)
    if cat is None:
        issue("catalog_missing", "error", f"catalog_g{m.generation}.json is missing or invalid")
    else:
        catalog_qi = QueryIndex(root=root, pdir=pdir, manifest=m, store=store, count=table_rows or 0)
        catalog_problem = _catalog_validation_error(cat, catalog_qi)
        if catalog_problem:
            issue("catalog_count_mismatch", "error", catalog_problem)
        else:
            totals = cat.get("totals") or {}
            if totals.get("files") != m.files or totals.get("chunks") != m.chunks:
                issue("catalog_count_mismatch", "error", "catalog totals differ from manifest")

    stale_files = []
    for rel in files_meta:
        stale, reason = _freshness_for_hit(root, files_meta, {"rel_path": rel})
        if stale:
            stale_files.append({"path": rel, "reason": reason})
    if stale_files:
        issue("index_stale", "warning", f"{len(stale_files)} indexed files differ from the working tree")

    git_status = None
    if check_git:
        indexed_git = {
            "git_worktree_root": m.git_worktree_root,
            "indexed_ref": m.indexed_ref,
            "indexed_commit": m.indexed_commit,
            "indexed_dirty": m.indexed_dirty,
        }
        try:
            git_status = gitmeta.current_staleness(root, indexed_git)
        except Exception as exc:
            git_status = {
                "available": False,
                "git_stale": False,
                "reasons": [],
                "indexed": indexed_git,
                "current": {},
                "warning": f"git metadata unavailable: {exc}",
            }
            issue("git_unavailable", "warning", f"git metadata unavailable: {exc}")
        if git_status.get("git_stale"):
            issue("git_stale", "warning", "current git state differs from indexed git state")

    errors_count = sum(1 for i in issues if i["severity"] == "error")
    warnings_count = sum(1 for i in issues if i["severity"] == "warning")
    return {
        "project_path": str(root),
        "project_id": m.project_id,
        "logical_project_id": m.logical_project_id,
        "checkout_kind": m.checkout_kind,
        "indexed_ref": m.indexed_ref,
        "index_generation": m.generation,
        "source_type": "static_indexed_source",
        "ok": errors_count == 0,
        "summary": {
            "issues": len(issues),
            "errors": errors_count,
            "warnings": warnings_count,
            "manifest_files": m.files,
            "manifest_chunks": m.chunks,
            "table_rows": table_rows,
            "stale_files": len(stale_files),
        },
        "git": git_status,
        "issues": issues,
    }


def grep_regex_timeout_seconds() -> float:
    raw = os.environ.get("ENGRAM_GREP_REGEX_TIMEOUT_SEC", "").strip()
    if not raw:
        value = DEFAULT_GREP_REGEX_TIMEOUT_SEC
    else:
        try:
            value = float(raw)
        except ValueError:
            value = DEFAULT_GREP_REGEX_TIMEOUT_SEC
    return max(0.05, min(value, 30.0))


def _grep_rows_worker(conn, pattern, flags, rows, include_lines, max_matches) -> None:
    try:
        rx = re.compile(pattern, flags)
        by_path: dict[str, dict] = {}
        total_matches = 0
        stopped = False
        for row in rows:
            rel = row.get("rel_path") or ""
            content = row.get("content") or ""
            start = int(row.get("start_line") or 1)
            line_cache = None
            for match in rx.finditer(content):
                item = by_path.setdefault(
                    rel,
                    {"path": rel, "match_count": 0, "line_numbers": set(), "lines": []},
                )
                total_matches += 1
                item["match_count"] += 1
                line_no = start + content[: match.start()].count("\n")
                item["line_numbers"].add(line_no)
                if include_lines:
                    if line_cache is None:
                        line_cache = content.splitlines()
                    idx = max(0, min(line_no - start, len(line_cache) - 1))
                    line_text = line_cache[idx] if line_cache else ""
                    item["lines"].append({"line": line_no, "text": line_text[:300]})
                if total_matches >= max_matches:
                    stopped = True
                    break
            if stopped:
                break
        conn.send(("ok", {"by_path": by_path, "total_matches": total_matches, "stopped": stopped}))
    except Exception as exc:
        conn.send(("error", str(exc) or repr(exc)))
    finally:
        conn.close()


def _grep_rows_with_timeout(
    *,
    pattern: str,
    flags: int,
    rows: list[dict],
    include_lines: bool,
    max_matches: int,
    timeout_sec: float,
) -> dict:
    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_grep_rows_worker,
        args=(send_conn, pattern, flags, rows, include_lines, max_matches),
    )
    proc.start()
    send_conn.close()
    try:
        if recv_conn.poll(timeout_sec):
            status, payload = recv_conn.recv()
            proc.join(timeout=1)
            if status == "ok":
                return payload
            raise ValueError(f"regex execution failed: {payload}")
        proc.terminate()
        proc.join(timeout=1)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=1)
        raise ValueError(f"regex execution timed out after {timeout_sec:.2f}s")
    finally:
        recv_conn.close()


def grep_index(
    root: str | Path,
    pattern: str,
    *,
    ignore_case: bool = False,
    limit: int = 50,
    offset: int = 0,
    max_matches: int = 500,
    max_scan_chunks: int = 10000,
    include_lines: bool = False,
) -> dict:
    """Bounded regex/count probe over indexed chunk text."""

    if not pattern or not pattern.strip():
        raise ValueError("pattern must not be empty")
    if len(pattern) > 500:
        raise ValueError("pattern must be at most 500 characters")
    flags = re.IGNORECASE if ignore_case else 0
    try:
        re.compile(pattern, flags)
    except re.error as exc:
        raise ValueError(f"invalid regex: {exc}") from exc
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    max_matches = max(1, min(int(max_matches), 5000))
    max_scan_chunks = max(1, min(int(max_scan_chunks), 100000))

    qi = load_query_index(root)
    rows = qi.store.metadata_rows(
        columns=("rel_path", "start_line", "content"),
        limit=min(qi.count, max_scan_chunks),
    )
    regex_result = _grep_rows_with_timeout(
        pattern=pattern,
        flags=flags,
        rows=rows,
        include_lines=include_lines,
        max_matches=max_matches,
        timeout_sec=grep_regex_timeout_seconds(),
    )
    by_path = regex_result["by_path"]
    total_matches = int(regex_result["total_matches"])
    stopped = bool(regex_result["stopped"])
    items = []
    for item in by_path.values():
        item["line_numbers"] = sorted(item["line_numbers"])
        if not include_lines:
            item.pop("lines", None)
        items.append(item)
    items.sort(key=lambda r: (-r["match_count"], r["path"]))
    page = items[offset : offset + limit]
    return {
        "project_path": str(qi.root),
        "project_id": qi.manifest.project_id,
        "index_generation": qi.manifest.generation,
        "source_type": "static_indexed_source",
        "pattern": pattern,
        "ignore_case": ignore_case,
        "limit": limit,
        "offset": offset,
        "count": len(page),
        "total_paths": len(items),
        "total_matches": total_matches,
        "max_matches": max_matches,
        "scanned_chunks": len(rows),
        "max_scan_chunks": max_scan_chunks,
        "truncated": stopped or len(rows) >= max_scan_chunks,
        "has_more": offset + limit < len(items),
        "results": page,
    }


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
