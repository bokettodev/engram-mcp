"""Stale LanceDB generation reclamation: explicit `engram gc`, and clean
server startup. Neither may ever touch the active generation, and neither may
ever run under ENGRAM_READONLY=1.
"""

from __future__ import annotations

from typing import Sequence

import lancedb

from engram_mcp import catalog, gcreclaim, manifest, paths, startup
from engram_mcp.store.lancedb_store import LanceStore

DIM = 4


def _row(chunk_id: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "rel_path": "a.py",
        "language": "python",
        "symbol": "",
        "symbol_kind": "",
        "chunk_role": "executable",
        "start_line": 1,
        "end_line": 2,
        "content": "def foo(): pass",
        "search_text": "path: a.py\n\ndef foo(): pass",
        "file_hash": "h",
        "chunk_hash": chunk_id,
        "vector": [0.1, 0.2, 0.3, 0.4],
    }


def _catalog(pdir, *, generation: int, active_table: str) -> None:
    catalog.save_catalog(
        pdir,
        {
            "schema_version": catalog.SCHEMA_VERSION,
            "project_id": "p",
            "root_path": "T:/fake",
            "generation": generation,
            "active_table": active_table,
            "indexed_at": 0.0,
            "totals": {"files": 1, "chunks": 1, "symbols": 0},
            "files": [],
        },
    )


def _make_project(tmp_path, name: str = "proj", *, generations: Sequence[int], active_generation: int):
    """Build a fake indexed project dir with one table per generation in
    ``generations`` (each holding one row) and a manifest whose active_table
    points at ``active_generation``. No embedding provider needed."""
    root = tmp_path / name
    root.mkdir()
    pdir = paths.project_dir(root)
    active_table = f"chunks_g{active_generation}"
    for gen in generations:
        table = f"chunks_g{gen}"
        LanceStore(pdir / "lancedb", DIM, table=table).create([_row(f"{table}-c1")])
        _catalog(pdir, generation=gen, active_table=table)
    m = manifest.ProjectManifest(
        project_id=paths.project_id_for(root),
        root_path=str(root.resolve()),
        logical_project_id=paths.project_id_for(root),
        checkout_kind="non_git",
        active_table=active_table,
        generation=active_generation,
        embedder_id="test:fake",
        dim=DIM,
        chunker_version="x",
        chunk_id_scheme="x",
        files=1,
        chunks=1,
    )
    manifest.save_project(pdir, m)
    return root, pdir


def _tables(pdir) -> set[str]:
    db = lancedb.connect(str(pdir / "lancedb"))
    return set(db.list_tables().tables)


def test_project_storage_report_separates_active_from_stale(tmp_path):
    _root, pdir = _make_project(tmp_path, generations=[1, 2], active_generation=2)

    report = gcreclaim.project_storage_report(pdir)

    assert report["active_table"] == "chunks_g2"
    assert report["active_table_bytes"] > 0
    assert report["active_catalog_bytes"] > 0
    assert [t["table"] for t in report["stale_tables"]] == ["chunks_g1"]
    assert report["stale_tables"][0]["generation"] == 1
    assert report["stale_tables"][0]["table_bytes"] > 0
    assert report["stale_tables"][0]["catalog_bytes"] > 0
    assert report["stale_table_bytes"] > 0
    assert report["stale_catalog_bytes"] > 0


def test_reclaim_project_dry_run_deletes_nothing(tmp_path):
    _root, pdir = _make_project(tmp_path, generations=[1, 2], active_generation=2)

    report = gcreclaim.reclaim_project(pdir, dry_run=True)

    assert report["dry_run"] is True
    assert report["dropped_tables"] == ["chunks_g1"]
    assert report["freed_bytes"] > 0
    assert _tables(pdir) == {"chunks_g1", "chunks_g2"}
    assert (pdir / "catalog_g1.json").is_file()


def test_reclaim_project_prune_drops_stale_keeps_active(tmp_path):
    _root, pdir = _make_project(tmp_path, generations=[1, 2], active_generation=2)

    report = gcreclaim.reclaim_project(pdir, dry_run=False)

    assert report["dry_run"] is False
    assert report["dropped_tables"] == ["chunks_g1"]
    assert report["freed_bytes"] > 0
    assert _tables(pdir) == {"chunks_g2"}
    assert not (pdir / "catalog_g1.json").exists()
    assert (pdir / "catalog_g2.json").is_file()
    # The active table is still fully readable afterward.
    assert LanceStore(pdir / "lancedb", DIM, table="chunks_g2").count() == 1


def test_reclaim_project_never_touches_active_generation_even_with_no_stale(tmp_path):
    _root, pdir = _make_project(tmp_path, generations=[2], active_generation=2)

    report = gcreclaim.reclaim_project(pdir, dry_run=False)

    assert report["dropped_tables"] == []
    assert report["freed_bytes"] == 0
    assert _tables(pdir) == {"chunks_g2"}


def test_reclaim_project_refuses_to_prune_under_readonly(tmp_path, monkeypatch):
    _root, pdir = _make_project(tmp_path, generations=[1, 2], active_generation=2)
    monkeypatch.setenv("ENGRAM_READONLY", "1")

    report = gcreclaim.reclaim_project(pdir, dry_run=False)

    assert report["skipped"] == "read_only"
    assert report["dry_run"] is True
    # A forced-dry-run report still previews what *would* be freed; nothing
    # is actually deleted (checked below) -- that's the invariant that matters.
    assert report["dropped_tables"] == ["chunks_g1"]
    assert _tables(pdir) == {"chunks_g1", "chunks_g2"}
    assert (pdir / "catalog_g1.json").is_file()


def test_reclaim_project_skips_when_project_lock_is_held(tmp_path):
    root, pdir = _make_project(tmp_path, generations=[1, 2], active_generation=2)

    lock = paths.project_lock(root)
    lock.acquire(timeout=0)
    try:
        report = gcreclaim.reclaim_project(pdir, dry_run=False)
    finally:
        lock.release()

    assert report["skipped"] == "project_busy"
    assert _tables(pdir) == {"chunks_g1", "chunks_g2"}


def test_reclaim_all_aggregates_across_projects_and_refuses_under_readonly(tmp_path, monkeypatch):
    _r1, pdir1 = _make_project(tmp_path, name="proj1", generations=[1, 2], active_generation=2)
    _r2, pdir2 = _make_project(tmp_path, name="proj2", generations=[5], active_generation=5)

    dry = gcreclaim.reclaim_all(dry_run=True)
    assert dry["total_stale_bytes"] > 0
    assert dry["total_freed_bytes"] == dry["total_stale_bytes"]
    assert _tables(pdir1) == {"chunks_g1", "chunks_g2"}

    monkeypatch.setenv("ENGRAM_READONLY", "1")
    readonly_prune = gcreclaim.reclaim_all(dry_run=False)
    assert readonly_prune["read_only"] is True
    assert readonly_prune["dry_run"] is True
    assert _tables(pdir1) == {"chunks_g1", "chunks_g2"}
    monkeypatch.delenv("ENGRAM_READONLY")

    pruned = gcreclaim.reclaim_all(dry_run=False)
    assert pruned["total_freed_bytes"] > 0
    assert _tables(pdir1) == {"chunks_g2"}
    assert _tables(pdir2) == {"chunks_g5"}


def test_full_rebuild_previous_generation_survives_and_is_reclaimed_by_gc(tmp_path):
    """The atomic-swap invariant ('previous generation survives a rebuild for
    in-flight readers, GC'd later') still holds with the new reclaim paths --
    an explicit `engram gc`-style call is what finally reclaims it, not the
    rebuild itself claiming it early."""
    from engram_mcp.pipeline import index_project

    class _FakeProvider:
        model_id = "test:fake-gc"
        backend_id = model_id
        dim = 4

        def embed_passages(self, texts):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

        def embed_queries(self, texts):
            return self.embed_passages(texts)

        def release_unused_cache(self):
            pass

    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    provider = _FakeProvider()

    index_project(root, provider, full_rebuild=True)
    pdir = paths.project_dir(root, create=False)
    m1 = manifest.load_project(pdir)

    index_project(root, provider, full_rebuild=True)
    m2 = manifest.load_project(pdir)
    assert m2.active_table != m1.active_table
    assert m1.active_table in _tables(pdir)  # retained for in-flight readers

    # A read tool / dry-run must not touch it.
    report = gcreclaim.project_storage_report(pdir)
    assert m1.active_table in {t["table"] for t in report["stale_tables"]}
    dry = gcreclaim.reclaim_project(pdir, dry_run=True)
    assert m1.active_table in _tables(pdir)
    assert m1.active_table in dry["dropped_tables"]

    # The explicit prune path reclaims it; the current active table survives.
    gcreclaim.reclaim_project(pdir, dry_run=False)
    assert m1.active_table not in _tables(pdir)
    assert m2.active_table in _tables(pdir)


def test_startup_maintenance_is_off_unless_opted_in(tmp_path, monkeypatch):
    """Default-off: one server process runs per MCP client window, all sharing
    ENGRAM_HOME. A generation this process calls stale may be the one another
    live process is mid-search on -- and retaining it across a swap is exactly
    what lets in-flight readers finish."""
    _root, pdir = _make_project(tmp_path, generations=[1, 2], active_generation=2)
    monkeypatch.delenv("ENGRAM_GC_ON_START", raising=False)

    result = startup.run_startup_maintenance()

    assert result is None
    assert _tables(pdir) == {"chunks_g1", "chunks_g2"}


def test_startup_maintenance_reclaims_stale_generations_when_opted_in(tmp_path, monkeypatch):
    _root, pdir = _make_project(tmp_path, generations=[1, 2], active_generation=2)
    monkeypatch.setenv("ENGRAM_GC_ON_START", "1")

    result = startup.run_startup_maintenance()

    assert result is not None
    assert result["stale_generations"]["total_freed_bytes"] > 0
    assert _tables(pdir) == {"chunks_g2"}


def test_startup_maintenance_skips_under_readonly(tmp_path, monkeypatch):
    _root, pdir = _make_project(tmp_path, generations=[1, 2], active_generation=2)
    monkeypatch.setenv("ENGRAM_READONLY", "1")

    result = startup.run_startup_maintenance()

    assert result is None
    assert _tables(pdir) == {"chunks_g1", "chunks_g2"}


def test_startup_maintenance_disabled_by_env(tmp_path, monkeypatch):
    _root, pdir = _make_project(tmp_path, generations=[1, 2], active_generation=2)
    monkeypatch.setenv("ENGRAM_GC_ON_START", "0")

    result = startup.run_startup_maintenance()

    assert result is None
    assert _tables(pdir) == {"chunks_g1", "chunks_g2"}
