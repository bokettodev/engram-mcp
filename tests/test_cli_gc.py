"""`engram gc` end-to-end: orphaned index dirs, stale LanceDB generations, and
the global embedding cache are all reported/reclaimed by one command.
"""

from __future__ import annotations

import json

from engram_mcp import catalog, cli, manifest, paths
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


def _make_stale_project(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    pdir = paths.project_dir(root)
    for gen in (1, 2):
        table = f"chunks_g{gen}"
        LanceStore(pdir / "lancedb", DIM, table=table).create([_row(f"{table}-c1")])
        catalog.save_catalog(
            pdir,
            {
                "schema_version": catalog.SCHEMA_VERSION,
                "project_id": "p",
                "root_path": "T:/fake",
                "generation": gen,
                "active_table": table,
                "indexed_at": 0.0,
                "totals": {"files": 1, "chunks": 1, "symbols": 0},
                "files": [],
            },
        )
    m = manifest.ProjectManifest(
        project_id=paths.project_id_for(root),
        root_path=str(root.resolve()),
        logical_project_id=paths.project_id_for(root),
        checkout_kind="non_git",
        active_table="chunks_g2",
        generation=2,
        embedder_id="test:fake",
        dim=DIM,
        chunker_version="x",
        chunk_id_scheme="x",
        files=1,
        chunks=1,
    )
    manifest.save_project(pdir, m)
    return root, pdir


def test_cli_gc_dry_run_reports_but_does_not_delete(tmp_path, capsys):
    _root, pdir = _make_stale_project(tmp_path)

    rc = cli.main(["gc", "--dry-run"])

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True
    assert out["stale_generations"]["total_freed_bytes"] > 0
    assert out["stale_generations"]["dry_run"] is True
    assert "embedding_cache" in out
    assert (pdir / "catalog_g1.json").exists()
    assert (pdir / "lancedb" / "chunks_g1.lance").exists()


def test_cli_gc_prune_reclaims_stale_generations(tmp_path, capsys):
    _root, pdir = _make_stale_project(tmp_path)

    rc = cli.main(["gc", "--prune"])

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is False
    assert out["stale_generations"]["total_freed_bytes"] > 0
    assert not (pdir / "catalog_g1.json").exists()
    assert (pdir / "catalog_g2.json").exists()
    assert LanceStore(pdir / "lancedb", DIM, table="chunks_g2").count() == 1


def test_cli_gc_prune_refused_under_readonly(tmp_path, monkeypatch, capsys):
    _root, pdir = _make_stale_project(tmp_path)
    monkeypatch.setenv("ENGRAM_READONLY", "1")

    rc = cli.main(["gc", "--prune"])

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["stale_generations"]["read_only"] is True
    assert out["stale_generations"]["dry_run"] is True
    assert (pdir / "catalog_g1.json").exists()
    assert (pdir / "lancedb" / "chunks_g1.lance").exists()
