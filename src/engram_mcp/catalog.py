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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

SCHEMA_VERSION = 1
SUPPORTED_FACETS = {"dir", "language", "chunk_role", "kind"}

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
    return {
        "schema_version": SCHEMA_VERSION,
        "project_id": project_id,
        "root_path": root_path,
        "generation": generation,
        "active_table": active_table,
        "indexed_at": indexed_at,
        "totals": totals,
        "files": files,
    }


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


def project_map(data: dict, *, depth: int = 2, sort: str = "path", limit: int = 200) -> dict:
    depth = max(0, min(int(depth), 20))
    limit = max(1, min(int(limit), 1000))
    sort = sort if sort in {"path", "files", "chunks", "symbols"} else "path"

    dirs: dict[str, dict] = {}
    for f in data.get("files", []):
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

    files = list(data.get("files", []))
    files.sort(key=lambda f: f.get("path") or "")
    return {
        "project_id": data.get("project_id"),
        "project_path": data.get("root_path"),
        "index_generation": data.get("generation"),
        "source_type": "static_indexed_source",
        "catalog_schema_version": data.get("schema_version"),
        "depth": depth,
        "sort": sort,
        "limit": limit,
        "count": len(rows[:limit]),
        "total_dirs": len(rows),
        "has_more": len(rows) > limit,
        "totals": data.get("totals", {}),
        "dirs": rows[:limit],
        "files": files[:limit],
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
