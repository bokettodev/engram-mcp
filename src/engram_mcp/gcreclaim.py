"""Reclaim superseded ("stale") LanceDB generation tables + catalog sidecars.

A full rebuild writes a fresh generation table (``chunks_g<N>``) and a fresh
``catalog_g<N>.json``, then atomically swaps the ``project.json`` pointer via
``os.replace``. The *previous* generation is deliberately left on disk so an
in-flight reader that already opened ``project.json`` before the swap keeps a
readable table -- deleting it out from under that reader would break the read.
Historically the only place that ever dropped a stale generation was the
*start* of the next full rebuild (see ``index_repository._full_rebuild``); a project
rebuilt once and then only queried/incrementally-updated afterward keeps that
stale generation forever.

This module adds two more, equally safe, reclaim points:

- an explicit operator command (``engram gc``), and
- clean server startup (``server.main``), before any client can have opened
  ``project.json`` and cached an old pointer.

Both share the same non-destructive default: everything here is dry-run
(report-only) unless the caller explicitly asks to prune, and pruning is
always refused under ``ENGRAM_READONLY=1``. Nothing in this module is ever
called from a read tool or a search/query path.
"""

from __future__ import annotations

import os
from pathlib import Path

import filelock

from engram_mcp import catalog, manifest, paths
from engram_mcp.store.lancedb_store import LanceStore


def _dir_bytes(path: Path) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                total += (Path(dirpath) / name).stat().st_size
            except OSError:
                pass
    return total


def _lance_table_dirs(db_dir: Path) -> list[str]:
    """List ``chunks*.lance`` table directory names on disk (no LanceDB connect).

    Pure filesystem listing so a dry-run report never touches the LanceDB API
    (which would create ``db_dir`` if it didn't already exist).
    """
    if not db_dir.is_dir():
        return []
    names = []
    for entry in db_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("chunks") and entry.name.endswith(".lance"):
            names.append(entry.name[: -len(".lance")])
    return sorted(names)


def project_storage_report(pdir: Path) -> dict:
    """Read-only storage breakdown for one project dir: active vs. stale bytes.

    Safe to call from a read tool (``doctor_project``): never connects to
    LanceDB, never creates ``pdir`` or any subdirectory, only stats files that
    already exist.
    """
    report = {
        "project_id": pdir.name,
        "active_table": "",
        "active_generation": None,
        "active_table_bytes": 0,
        "active_catalog_bytes": 0,
        "stale_tables": [],
        "stale_table_bytes": 0,
        "stale_catalog_bytes": 0,
        "error": None,
    }
    m = manifest.load_project(pdir)
    if m is None:
        report["error"] = "project manifest missing or invalid"
        return report
    active_table = m.active_table or ""
    report["active_table"] = active_table
    report["active_generation"] = m.generation
    db_dir = pdir / "lancedb"
    if active_table:
        active_dir = db_dir / f"{active_table}.lance"
        if active_dir.is_dir():
            report["active_table_bytes"] = _dir_bytes(active_dir)
    active_catalog = catalog.catalog_path(pdir, m.generation)
    if active_catalog.is_file():
        try:
            report["active_catalog_bytes"] = active_catalog.stat().st_size
        except OSError:
            pass

    for name in _lance_table_dirs(db_dir):
        if name == active_table:
            continue
        table_bytes = _dir_bytes(db_dir / f"{name}.lance")
        generation = catalog.generation_for_table(name)
        catalog_bytes = 0
        if generation is not None:
            cat_path = catalog.catalog_path(pdir, generation)
            if cat_path.is_file():
                try:
                    catalog_bytes = cat_path.stat().st_size
                except OSError:
                    catalog_bytes = 0
        report["stale_tables"].append(
            {
                "table": name,
                "generation": generation,
                "table_bytes": table_bytes,
                "catalog_bytes": catalog_bytes,
            }
        )
        report["stale_table_bytes"] += table_bytes
        report["stale_catalog_bytes"] += catalog_bytes
    return report


def reclaim_project(pdir: Path, *, dry_run: bool) -> dict:
    """Report (``dry_run=True``) or reclaim (``dry_run=False``) one project's stale generations.

    Never touches the active table/catalog. When actually pruning, this
    acquires the *same* per-project lock ``index_repository.index_project`` /
    ``remove_project`` hold while writing, non-blocking: if a write is
    already in progress for this project, reclaiming is skipped for this
    pass rather than racing it (it will simply be reclaimed on the next
    call). Refuses to prune under ``ENGRAM_READONLY=1`` (degrades to a
    report).
    """
    report = project_storage_report(pdir)
    report["dry_run"] = dry_run
    report["dropped_tables"] = []
    report["freed_bytes"] = 0
    report["skipped"] = None
    if report["error"] or not report["stale_tables"]:
        return report
    if not dry_run and paths.read_only_enabled():
        report["dry_run"] = True
        report["skipped"] = "read_only"
        dry_run = True
    if dry_run:
        report["dropped_tables"] = [t["table"] for t in report["stale_tables"]]
        report["freed_bytes"] = report["stale_table_bytes"] + report["stale_catalog_bytes"]
        return report

    m = manifest.load_project(pdir)
    if m is None or not m.root_path:
        report["skipped"] = "manifest_unavailable"
        return report
    lock = paths.project_lock(Path(m.root_path))
    try:
        lock.acquire(timeout=0)
    except filelock.Timeout:
        report["skipped"] = "project_busy"
        return report
    try:
        # Re-read after acquiring the lock: a concurrent rebuild that just
        # finished may have changed the active table / already GC'd stale
        # ones (`_full_rebuild` does its own reclaim at the start of a
        # rebuild), so recompute against the current on-disk truth rather
        # than the pre-lock snapshot.
        fresh = project_storage_report(pdir)
        if fresh["error"] or not fresh["stale_tables"]:
            fresh["dry_run"] = False
            fresh["dropped_tables"] = []
            fresh["freed_bytes"] = 0
            fresh["skipped"] = None
            return fresh
        active_table = fresh["active_table"]
        keep = {active_table} if active_table else set()
        db_dir = pdir / "lancedb"
        store = LanceStore(db_dir, max(1, m.dim or 1), table=active_table or "chunks")
        dropped = store.drop_stale_generations(keep)
        stale_generations = {
            gen for gen in (catalog.generation_for_table(name) for name in dropped) if gen is not None
        }
        catalog.drop_catalogs_for_generations(pdir, stale_generations)
        fresh["dry_run"] = False
        fresh["dropped_tables"] = sorted(dropped)
        fresh["freed_bytes"] = sum(
            t["table_bytes"] + t["catalog_bytes"] for t in fresh["stale_tables"] if t["table"] in dropped
        )
        fresh["skipped"] = None
        return fresh
    finally:
        lock.release()


def _project_dirs() -> list[Path]:
    base = paths.data_home(create=False) / "projects"
    if not base.is_dir():
        return []
    return sorted(d for d in base.iterdir() if d.is_dir())


def reclaim_all(*, dry_run: bool = True) -> dict:
    """Reclaim (or report on) stale generations across every indexed project."""
    read_only = paths.read_only_enabled()
    effective_dry_run = dry_run or read_only
    home = paths.data_home(create=False)
    out = {
        "data_home": str(home),
        "dry_run": effective_dry_run,
        "read_only": read_only,
        "projects": [],
        "total_freed_bytes": 0,
        "total_stale_bytes": 0,
        "errors": [],
    }
    for d in _project_dirs():
        report = reclaim_project(d, dry_run=effective_dry_run)
        if report.get("error"):
            out["errors"].append({"project_id": d.name, "error": report["error"]})
        out["projects"].append(report)
        out["total_freed_bytes"] += report.get("freed_bytes", 0)
        out["total_stale_bytes"] += report.get("stale_table_bytes", 0) + report.get("stale_catalog_bytes", 0)
    return out
