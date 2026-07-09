"""Body-free project catalog sidecars.

The catalog is derived from the active Lance generation and is rebuildable. It
contains metadata only: paths, counts, symbols, chunk spans, and narrow inferred
file kinds. Raw bodies stay in LanceDB.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from fnmatch import fnmatchcase
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

SCHEMA_VERSION = 1
SUPPORTED_FACETS = {"dir", "language", "chunk_role", "kind"}
SZZ_STATUSES = {"computing", "partial", "ready", "unavailable"}

_CONFIG_NAMES = {
    ".env",
    ".env.example",
    ".gitignore",
    ".dockerignore",
    "package.json",
    "pyproject.toml",
    "setup.cfg",
    "tox.ini",
    "tsconfig.json",
}
_CONFIG_EXTS = {".json", ".toml", ".yaml", ".yml", ".ini", ".cfg"}
_DOC_EXTS = {".md", ".markdown", ".rst", ".txt", ".adoc"}
_TEST_RE = re.compile(r"(^test_|_test$|\.test\.|\.spec\.)")


def catalog_path(pdir: Path, generation: int) -> Path:
    return pdir / f"catalog_g{generation}.json"


def szz_path(pdir: Path, generation: int) -> Path:
    return pdir / f"szz_g{generation}.json"


def _atomic_write_json(path: Path, data) -> None:
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


def save_catalog(pdir: Path, data: dict) -> None:
    _atomic_write_json(catalog_path(pdir, int(data.get("generation", 0))), data)


def save_szz(pdir: Path, data: dict) -> None:
    _atomic_write_json(szz_path(pdir, int(data.get("generation", 0))), data)


def load_szz(pdir: Path, generation: int) -> dict | None:
    f = szz_path(pdir, generation)
    if not f.is_file():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("schema_version") != SCHEMA_VERSION:
        return None
    try:
        if int(data.get("generation", -1)) != int(generation):
            return None
    except (TypeError, ValueError):
        return None
    if str(data.get("status") or "") not in SZZ_STATUSES:
        return None
    return data


def mark_catalog_stale(
    pdir: Path,
    *,
    project_id: str,
    root_path: str,
    generation: int,
    active_table: str,
    reason: str,
) -> None:
    """Atomically replace a catalog with an invalidating marker.

    Incremental writers call this before mutating a same-generation Lance table.
    A crash after the table mutation then leaves an unavailable catalog, never a
    stale one that still passes generation/table checks.
    """

    _atomic_write_json(
        catalog_path(pdir, int(generation)),
        {
            "schema_version": SCHEMA_VERSION,
            "project_id": project_id,
            "root_path": root_path,
            "generation": int(generation),
            "active_table": active_table,
            "status": "stale",
            "reason": reason,
            "totals": {"files": 0, "chunks": 0, "symbols": 0},
            "files": [],
        },
    )


def load_catalog(pdir: Path, generation: int) -> dict | None:
    f = catalog_path(pdir, generation)
    if not f.is_file():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("schema_version") != SCHEMA_VERSION:
        return None
    if data.get("status") not in (None, "ready"):
        return None
    try:
        if int(data.get("generation", -1)) != int(generation):
            return None
    except (TypeError, ValueError):
        return None
    return data


def _dir_for(rel_path: str) -> str:
    parent = Path(rel_path.replace("\\", "/")).parent.as_posix()
    return "." if parent == "." else parent


def derive_file_kinds(rel_path: str, language: str | None = None) -> list[dict]:
    """High-precision structural file tags.

    These are intentionally narrow and marked inferred because they come from
    path/name/language heuristics, not semantic classification.
    """

    rel = rel_path.replace("\\", "/")
    lower = rel.lower()
    name = lower.rsplit("/", 1)[-1]
    suffixes = "".join(Path(name).suffixes)
    suffix = Path(name).suffix
    kinds: list[dict] = []

    def add(kind: str, reason: str) -> None:
        if not any(k["kind"] == kind for k in kinds):
            kinds.append({"kind": kind, "source": "inferred", "reason": reason})

    segments = [s for s in lower.split("/") if s]
    if (
        "test" in segments
        or "tests" in segments
        or name.startswith("test_")
        or name.endswith("_test.py")
        or _TEST_RE.search(name)
    ):
        add("test", "test path or filename")
    if "migration" in segments or "migrations" in segments:
        add("migration", "migration path segment")
    if name in _CONFIG_NAMES or suffix in _CONFIG_EXTS or ".config." in name or suffixes.endswith(".config.js"):
        add("config", "config filename or extension")
    if suffix in _DOC_EXTS or (language or "").lower() in {"markdown", "text"}:
        add("doc", "documentation extension or language")
    return kinds


def _symbol_entry(row: dict) -> dict | None:
    symbol = row.get("symbol") or ""
    if not symbol:
        return None
    return {
        "name": symbol,
        "symbol_kind": row.get("symbol_kind") or "",
        "chunk_id": row.get("chunk_id") or "",
        "start_line": row.get("start_line"),
        "end_line": row.get("end_line"),
    }


def _chunk_entry(row: dict) -> dict:
    return {
        "chunk_id": row.get("chunk_id") or "",
        "start_line": row.get("start_line"),
        "end_line": row.get("end_line"),
        "symbol": row.get("symbol") or "",
        "symbol_kind": row.get("symbol_kind") or "",
        "chunk_role": row.get("chunk_role") or "",
    }


def build_catalog(
    *,
    project_id: str,
    root_path: str,
    generation: int,
    active_table: str,
    files_meta: dict[str, dict],
    rows: Iterable[dict],
    indexed_at: float,
    git_history: dict | None = None,
) -> dict:
    by_file: dict[str, dict] = {}
    role_counts: dict[str, Counter] = defaultdict(Counter)
    symbols: dict[str, list[dict]] = defaultdict(list)
    chunks: dict[str, list[dict]] = defaultdict(list)
    seen_symbols: dict[str, set[tuple[str, str, int | None, int | None]]] = defaultdict(set)

    for rel_path, meta in files_meta.items():
        lang = meta.get("language") or ""
        by_file[rel_path] = {
            "path": rel_path,
            "dir": _dir_for(rel_path),
            "language": lang,
            "chunks": int(meta.get("chunks", 0) or 0),
            "indent_complexity": float(meta.get("indent_complexity", 0.0) or 0.0),
            "chunk_roles": {},
            "symbols": [],
            "chunk_refs": [],
            "kinds": derive_file_kinds(rel_path, lang),
        }

    for row in rows:
        rel = row.get("rel_path") or ""
        if not rel:
            continue
        if rel not in by_file:
            lang = row.get("language") or ""
            by_file[rel] = {
                "path": rel,
                "dir": _dir_for(rel),
                "language": lang,
                "chunks": 0,
                "indent_complexity": 0.0,
                "chunk_roles": {},
                "symbols": [],
                "chunk_refs": [],
                "kinds": derive_file_kinds(rel, lang),
            }
        role = row.get("chunk_role") or ""
        if role:
            role_counts[rel][role] += 1
        sym = _symbol_entry(row)
        if sym is not None:
            key = (sym["name"], sym["symbol_kind"], sym["start_line"], sym["end_line"])
            if key not in seen_symbols[rel]:
                seen_symbols[rel].add(key)
                symbols[rel].append(sym)
        chunks[rel].append(_chunk_entry(row))

    for rel, entry in by_file.items():
        refs = sorted(chunks.get(rel, []), key=lambda c: (c.get("start_line") or 0, c.get("chunk_id") or ""))
        entry["chunk_refs"] = refs
        entry["chunk_roles"] = dict(sorted(role_counts.get(rel, Counter()).items()))
        entry["symbols"] = sorted(
            symbols.get(rel, []), key=lambda s: (s.get("start_line") or 0, s.get("name") or "")
        )
        if refs:
            entry["chunks"] = len(refs)

    files = sorted(by_file.values(), key=lambda f: f["path"])
    totals = {
        "files": len(files),
        "chunks": sum(int(f.get("chunks", 0) or 0) for f in files),
        "symbols": sum(len(f.get("symbols", [])) for f in files),
    }
    data = {
        "schema_version": SCHEMA_VERSION,
        "project_id": project_id,
        "root_path": root_path,
        "generation": generation,
        "active_table": active_table,
        "indexed_at": indexed_at,
        "totals": totals,
        "files": files,
    }
    if isinstance(git_history, dict):
        data["git_history"] = git_history
    return data


def file_by_path(data: dict) -> dict[str, dict]:
    return {f.get("path", ""): f for f in data.get("files", []) if f.get("path")}


def chunk_lookup(data: dict) -> dict[str, tuple[dict, dict, int]]:
    out: dict[str, tuple[dict, dict, int]] = {}
    for f in data.get("files", []):
        for idx, ch in enumerate(f.get("chunk_refs", [])):
            cid = ch.get("chunk_id")
            if cid:
                out[cid] = (f, ch, idx)
    return out


_MAX_MAP_LIMIT = 1000


def _coerce_limit(value: int | None, *, default: int) -> int:
    if value is None:
        value = default
    return max(0, min(int(value), _MAX_MAP_LIMIT))


def _coerce_offset(value: int) -> int:
    return max(0, int(value))


def _normalize_values(values: Iterable[str] | str | None) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        values = [values]
    return {str(value) for value in values if str(value)}


def _normalize_rel_filter(value: str | None) -> str:
    if not value:
        return ""
    rel = str(value).replace("\\", "/").strip()
    while rel.startswith("./"):
        rel = rel[2:]
    return rel.strip("/")


def _matches_prefix(path: str, prefix: str) -> bool:
    if not prefix or prefix == ".":
        return True
    return path == prefix or path.startswith(prefix + "/")


def _kind_values(file_row: dict) -> set[str]:
    out: set[str] = set()
    for kind in file_row.get("kinds") or []:
        value = kind.get("kind") if isinstance(kind, dict) else kind
        if value:
            out.add(str(value))
    return out


def _symbol_kind_values(file_row: dict) -> set[str]:
    out: set[str] = set()
    for symbol in file_row.get("symbols") or []:
        value = symbol.get("symbol_kind") if isinstance(symbol, dict) else ""
        if value:
            out.add(str(value))
    return out


def _symbols_count(file_row: dict) -> int:
    return len(file_row.get("symbols") or [])


def _filtered_totals(files: list[dict]) -> dict:
    return {
        "files": len(files),
        "chunks": sum(int(f.get("chunks", 0) or 0) for f in files),
        "symbols": sum(_symbols_count(f) for f in files),
    }


def _page(items: list[dict], *, offset: int, limit: int) -> tuple[list[dict], dict]:
    total = len(items)
    selected = items[offset : offset + limit]
    return selected, {
        "offset": offset,
        "limit": limit,
        "count": len(selected),
        "total": total,
        "has_more": offset + len(selected) < total,
    }


def _compact_file_row(file_row: dict, *, include_symbols: bool, symbols_limit: int) -> dict:
    symbols = file_row.get("symbols") or []
    row = {
        "path": file_row.get("path") or "",
        "dir": file_row.get("dir") or ".",
        "language": file_row.get("language") or "",
        "chunks": int(file_row.get("chunks", 0) or 0),
        "indent_complexity": float(file_row.get("indent_complexity", 0.0) or 0.0),
        "symbols_count": len(symbols),
        "chunk_roles": dict(sorted((file_row.get("chunk_roles") or {}).items())),
        "kinds": list(file_row.get("kinds") or []),
    }
    if include_symbols:
        row["symbols"] = list(symbols[:symbols_limit])
        row["symbols_has_more"] = len(symbols) > symbols_limit
    return row


def _filter_files(
    files: Iterable[dict],
    *,
    code_only: bool,
    languages: Iterable[str] | str | None,
    chunk_roles: Iterable[str] | str | None,
    kinds: Iterable[str] | str | None,
    path_prefix: str | None,
    path_glob: str | None,
    symbol_kinds: Iterable[str] | str | None,
    min_symbols: int,
    non_empty: bool,
) -> list[dict]:
    language_filter = _normalize_values(languages)
    role_filter = _normalize_values(chunk_roles)
    kind_filter = _normalize_values(kinds)
    symbol_kind_filter = _normalize_values(symbol_kinds)
    prefix = _normalize_rel_filter(path_prefix)
    glob = _normalize_rel_filter(path_glob)
    min_symbols = max(0, int(min_symbols))

    out: list[dict] = []
    for file_row in files:
        path = (file_row.get("path") or "").replace("\\", "/")
        roles = {
            str(role)
            for role, count in (file_row.get("chunk_roles") or {}).items()
            if role and int(count or 0) > 0
        }
        symbol_count = _symbols_count(file_row)

        if non_empty and int(file_row.get("chunks", 0) or 0) <= 0:
            continue
        if code_only and not roles.intersection({"executable", "test"}):
            continue
        if language_filter and (file_row.get("language") or "") not in language_filter:
            continue
        if role_filter and not roles.intersection(role_filter):
            continue
        if kind_filter and not _kind_values(file_row).intersection(kind_filter):
            continue
        if prefix and not _matches_prefix(path, prefix):
            continue
        if glob and not fnmatchcase(path, glob):
            continue
        if symbol_kind_filter and not _symbol_kind_values(file_row).intersection(symbol_kind_filter):
            continue
        if symbol_count < min_symbols:
            continue
        out.append(file_row)
    return out


def project_map(
    data: dict,
    *,
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
    languages: Iterable[str] | str | None = None,
    chunk_roles: Iterable[str] | str | None = None,
    kinds: Iterable[str] | str | None = None,
    path_prefix: str | None = None,
    path_glob: str | None = None,
    symbol_kinds: Iterable[str] | str | None = None,
    min_symbols: int = 0,
    non_empty: bool = True,
) -> dict:
    depth = max(0, min(int(depth), 20))
    dirs_limit_value = _coerce_limit(dirs_limit, default=200)
    dirs_offset_value = _coerce_offset(dirs_offset)
    files_limit_value = _coerce_limit(files_limit, default=50)
    files_offset_value = _coerce_offset(files_offset)
    symbols_limit_value = _coerce_limit(symbols_limit, default=20)
    sort = sort if sort in {"path", "files", "chunks", "symbols"} else "path"

    filtered_files = _filter_files(
        data.get("files", []),
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

    dirs: dict[str, dict] = {}
    for f in filtered_files:
        path = f.get("path") or ""
        parts = path.split("/")[:-1]
        if depth == 0 or not parts:
            key = "."
        else:
            key = "/".join(parts[:depth])
        entry = dirs.setdefault(
            key,
            {
                "dir": key,
                "files": 0,
                "chunks": 0,
                "symbols": 0,
                "languages": Counter(),
                "chunk_roles": Counter(),
                "kinds": Counter(),
            },
        )
        entry["files"] += 1
        entry["chunks"] += int(f.get("chunks", 0) or 0)
        entry["symbols"] += len(f.get("symbols", []))
        lang = f.get("language") or ""
        if lang:
            entry["languages"][lang] += 1
        for role, count in (f.get("chunk_roles") or {}).items():
            entry["chunk_roles"][role] += int(count or 0)
        for kind in f.get("kinds") or []:
            entry["kinds"][kind.get("kind") or ""] += 1

    rows = list(dirs.values())
    if sort == "path":
        rows.sort(key=lambda r: r["dir"])
    elif sort == "files":
        rows.sort(key=lambda r: (-r["files"], r["dir"]))
    elif sort == "chunks":
        rows.sort(key=lambda r: (-r["chunks"], r["dir"]))
    else:
        rows.sort(key=lambda r: (-r["symbols"], r["dir"]))

    for row in rows:
        row["languages"] = dict(sorted(row["languages"].items()))
        row["chunk_roles"] = dict(sorted(row["chunk_roles"].items()))
        row["kinds"] = {k: v for k, v in sorted(row["kinds"].items()) if k}

    dirs_page_rows, dirs_page = _page(
        rows,
        offset=dirs_offset_value,
        limit=dirs_limit_value,
    )

    files = list(filtered_files)
    files.sort(key=lambda f: f.get("path") or "")
    if include_files:
        files_page_rows, files_page = _page(
            files,
            offset=files_offset_value,
            limit=files_limit_value,
        )
        file_rows = [
            _compact_file_row(
                file_row,
                include_symbols=include_symbols,
                symbols_limit=symbols_limit_value,
            )
            for file_row in files_page_rows
        ]
    else:
        file_rows = []
        files_page = {
            "offset": files_offset_value,
            "limit": files_limit_value,
            "count": 0,
            "total": len(files),
            "has_more": False,
        }
    files_page["included"] = bool(include_files)

    return {
        "project_id": data.get("project_id"),
        "project_path": data.get("root_path"),
        "index_generation": data.get("generation"),
        "source_type": "static_indexed_source",
        "catalog_schema_version": data.get("schema_version"),
        "depth": depth,
        "sort": sort,
        "dirs_limit": dirs_limit_value,
        "files_limit": files_limit_value,
        "count": dirs_page["count"],
        "total_dirs": dirs_page["total"],
        "has_more": bool(dirs_page["has_more"] or files_page["has_more"]),
        "totals": data.get("totals", {}),
        "filtered_totals": _filtered_totals(files),
        "dirs_page": dirs_page,
        "files_page": files_page,
        "dirs": dirs_page_rows,
        "files": file_rows,
    }


def facet_counts(data: dict, facets: Iterable[str], *, rel_paths: set[str] | None = None) -> dict:
    requested = [f for f in facets if f in SUPPORTED_FACETS]
    counts = {f: Counter() for f in requested}
    for f in data.get("files", []):
        path = f.get("path") or ""
        if rel_paths is not None and path not in rel_paths:
            continue
        for facet in requested:
            if facet == "dir":
                counts[facet][f.get("dir") or "."] += 1
            elif facet == "language":
                value = f.get("language") or "(none)"
                counts[facet][value] += 1
            elif facet == "chunk_role":
                for role, n in (f.get("chunk_roles") or {}).items():
                    counts[facet][role] += int(n or 0)
            elif facet == "kind":
                for kind in f.get("kinds") or []:
                    value = kind.get("kind") or ""
                    if value:
                        counts[facet][value] += 1
    return {facet: dict(sorted(counter.items())) for facet, counter in counts.items()}
