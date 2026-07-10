"""doctor_project's storage section: active vs. stale generation bytes,
catalog sidecar bytes, and the global embedding-cache size/entry count. As a
read tool it must never create the LanceDB dir or the global-cache dir merely
by reporting on them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from engram_mcp import config, gcreclaim, manifest, paths
from engram_mcp.pipeline import doctor_project, index_project


class _FakeProvider:
    model_id = "test:fake-storage"
    backend_id = model_id
    dim = 4

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_passages(texts)

    def release_unused_cache(self) -> None:
        pass


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    return root


def test_doctor_project_storage_reports_active_and_stale_bytes(tmp_path):
    root = _project(tmp_path)
    provider = _FakeProvider()
    index_project(root, provider, full_rebuild=True)
    index_project(root, provider, full_rebuild=True)  # creates a 2nd generation, keeps the 1st on disk

    health = doctor_project(root, check_git=False)
    storage = health["storage"]

    assert storage["active_generation_bytes"] > 0
    assert storage["active_catalog_bytes"] > 0
    assert storage["stale_generation_bytes"] > 0
    assert storage["stale_catalog_bytes"] > 0
    assert len(storage["stale_generations"]) == 1
    assert storage["reclaim_hint"] is not None
    assert storage["global_cache_bytes"] > 0
    assert storage["global_cache_rows"] > 0
    assert storage["global_cache_path"].endswith("embeddings.sqlite")


def test_doctor_project_storage_reflects_reclaim(tmp_path):
    root = _project(tmp_path)
    provider = _FakeProvider()
    index_project(root, provider, full_rebuild=True)
    index_project(root, provider, full_rebuild=True)
    pdir = paths.project_dir(root, create=False)

    gcreclaim.reclaim_project(pdir, dry_run=False)

    health = doctor_project(root, check_git=False)
    storage = health["storage"]
    assert storage["stale_generation_bytes"] == 0
    assert storage["stale_generations"] == []
    assert storage["reclaim_hint"] is None
    assert storage["active_generation_bytes"] > 0  # active survives


def test_doctor_project_storage_never_creates_lancedb_or_cache_dirs(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    pdir = paths.project_dir(root)
    manifest.save_project(
        pdir,
        manifest.ProjectManifest(
            project_id=paths.project_id_for(root),
            root_path=str(root.resolve()),
            logical_project_id=paths.project_id_for(root),
            checkout_kind="non_git",
            active_table="chunks",
            embedder_id="test:fake",
            dim=4,
            chunker_version=config.CHUNKER_VERSION,
            chunk_id_scheme=config.CHUNK_ID_SCHEME,
            files=1,
            chunks=1,
        ),
    )
    lancedb_dir = pdir / "lancedb"
    cache_dir = paths.data_home(create=False) / "global-cache"
    assert not lancedb_dir.exists()
    assert not cache_dir.exists()

    health = doctor_project(root, check_git=False)

    assert health["storage"]["active_generation_bytes"] == 0
    assert health["storage"]["global_cache_bytes"] == 0
    assert health["storage"]["global_cache_rows"] is None
    assert not lancedb_dir.exists()
    assert not cache_dir.exists()
