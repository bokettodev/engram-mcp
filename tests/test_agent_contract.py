"""Agent-grade MCP contract tests that do not load real embedding models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest

from engram_mcp import config, errors, manifest, paths, retrieval
from engram_mcp.pipeline import (
    ProjectNotIndexedError,
    derive_chunk_role,
    index_project,
    load_query_index,
    search_project,
    _is_compatible,
)
from engram_mcp.store.lancedb_store import LanceStore


class FakeProvider:
    model_id = "test:fake"
    backend_id = model_id
    dim = 4

    def _vec(self, text: str) -> list[float]:
        low = text.lower()
        return [
            1.0 if "add" in low or "sum" in low else 0.0,
            1.0 if "cache" in low else 0.0,
            1.0 if "config" in low else 0.0,
            1.0,
        ]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def release_unused_cache(self) -> None:
        pass


def _write_project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "math_utils.py").write_text(
        "def add_numbers(a, b):\n"
        "    '''Return the sum.'''\n"
        "    return a + b\n",
        encoding="utf-8",
    )
    (proj / "cache.py").write_text(
        "class EmbeddingCache:\n"
        "    def get_many(self):\n"
        "        return {}\n",
        encoding="utf-8",
    )
    return proj


def _indexed_project(tmp_path: Path) -> tuple[Path, FakeProvider]:
    provider = FakeProvider()
    proj = _write_project(tmp_path)
    index_project(proj, provider, full_rebuild=True)
    return proj, provider


def _write_manifest_only(root: Path, dim: int = 4, **overrides) -> Path:
    pdir = paths.project_dir(root)
    m = manifest.ProjectManifest(
        project_id=paths.project_id_for(root),
        root_path=str(root.resolve()),
        active_table="chunks",
        embedder_id="test:fake",
        dim=dim,
        chunker_version=config.CHUNKER_VERSION,
        files=1,
        chunks=1,
    )
    for key, value in overrides.items():
        setattr(m, key, value)
    manifest.save_project(pdir, m)
    return pdir


def _create_one_row_table(root: Path, *, embedder_id: str = "test:fake", dim: int = 4) -> None:
    pdir = _write_manifest_only(root, embedder_id=embedder_id, dim=dim)
    vector = [1.0] + [0.0] * (dim - 1)
    LanceStore(pdir / "lancedb", dim, table="chunks").create(
        [
            {
                "chunk_id": "c1",
                "rel_path": "a.py",
                "language": "python",
                "symbol": "a",
                "symbol_kind": "function_definition",
                "chunk_role": "executable",
                "start_line": 1,
                "end_line": 1,
                "content": "def a(): pass",
                "search_text": "path: a.py\nsymbol: a\n\ndef a(): pass",
                "file_hash": "h",
                "chunk_hash": "ch",
                "vector": vector,
            }
        ]
    )
    manifest.save_files(
        pdir,
        {"a.py": {"file_hash": "h", "mtime_ns": 1, "size": 1, "language": "python", "chunks": 1}},
    )


def test_load_query_index_rejects_missing_corrupt_and_incompatible_manifest(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    with pytest.raises(ProjectNotIndexedError):
        load_query_index(proj)

    pdir = paths.project_dir(proj)
    (pdir / "project.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(errors.EngramError) as corrupt:
        load_query_index(proj)
    assert corrupt.value.code == errors.E_INDEX_INVALID

    manifest.save_project(
        pdir,
        manifest.ProjectManifest(
            project_id=paths.project_id_for(proj),
            root_path=str(proj.resolve()),
            active_table="chunks",
            embedder_id="",
            dim=0,
            chunker_version="old",
        ),
    )
    with pytest.raises(errors.EngramError) as bad:
        load_query_index(proj)
    assert bad.value.code == errors.E_INDEX_INVALID
    assert "embedder_id" in str(bad.value)
    assert "dim" in str(bad.value)
    assert "chunker_version" in str(bad.value)


def test_load_query_index_rejects_missing_or_empty_active_table(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    pdir = _write_manifest_only(proj)
    with pytest.raises(errors.EngramError) as missing:
        load_query_index(proj)
    assert missing.value.code == errors.E_INDEX_INVALID

    LanceStore(pdir / "lancedb", 4, table="chunks").create([])
    with pytest.raises(errors.EngramError) as empty:
        load_query_index(proj)
    assert empty.value.code == errors.E_INDEX_INVALID
    assert "empty" in str(empty.value)


def test_server_error_codes_and_read_only_hint(tmp_path, monkeypatch):
    from engram_mcp import server

    missing = server.do_search(str(tmp_path / "missing"), "add")
    assert missing["code"] == errors.E_PROJECT_NOT_INDEXED

    monkeypatch.setenv("ENGRAM_READONLY", "1")
    readonly = server.do_search(str(tmp_path / "missing2"), "add")
    assert readonly["code"] == errors.E_PROJECT_NOT_INDEXED
    assert "out of band via the engram CLI/operator" in readonly["hint"]
    monkeypatch.delenv("ENGRAM_READONLY", raising=False)

    proj = tmp_path / "bad-index"
    proj.mkdir()
    _create_one_row_table(proj, embedder_id="unsupported:model")
    unknown = server.do_search(str(proj), "add")
    assert unknown["code"] == errors.E_UNKNOWN_PROFILE

    monkeypatch.setattr(
        server,
        "_provider_for_query_model",
        lambda _model_id: (_ for _ in ()).throw(ImportError("missing gpu extra")),
    )
    extra = server.do_search(str(proj), "add")
    assert extra["code"] == errors.E_EXTRA_MISSING

    monkeypatch.setattr(
        server,
        "_provider_for_query_model",
        lambda _model_id: (_ for _ in ()).throw(RuntimeError("download failed")),
    )
    failed = server.do_search(str(proj), "add")
    assert failed["code"] == errors.E_MODEL_LOAD_FAILED

    monkeypatch.setattr(
        server,
        "_provider_for_query_model",
        lambda _model_id: (_ for _ in ()).throw(
            errors.EngramError("loading", errors.E_MODEL_LOADING)
        ),
    )
    loading = server.do_search(str(proj), "add")
    assert loading["code"] == errors.E_MODEL_LOADING
    assert loading["retry_after_sec"] > 0

    bad_req = server.do_search(str(proj), "add", k=0)
    assert bad_req["code"] == errors.E_BAD_REQUEST


def test_invalid_env_index_device_is_index_time_only(tmp_path, monkeypatch):
    from engram_mcp import server
    from engram_mcp.embeddings import factory

    monkeypatch.setenv("ENGRAM_INDEX_DEVICE", "not-a-device")
    with pytest.raises(errors.EngramError) as exc:
        factory.default_index_device()
    assert exc.value.code == errors.E_BAD_REQUEST

    proj, provider = _indexed_project(tmp_path)
    monkeypatch.setattr(server, "_provider_for_query_model", lambda _model_id: provider)
    out = server.do_search(str(proj), "add numbers", k=1)
    assert "error" not in out
    assert out["count"] == 1


def test_compact_search_get_chunk_relevance_and_stale(tmp_path, monkeypatch):
    from engram_mcp import server

    proj, provider = _indexed_project(tmp_path)
    monkeypatch.setattr(server, "_provider_for_query_model", lambda _model_id: provider)

    out = server.do_search(str(proj), "add numbers", k=2)
    assert out["source_type"] == "static_indexed_source"
    assert out["content"] == "preview"
    assert out["mode_requested"] == "auto"
    assert out["mode_used"] in {"vector", "hybrid"}
    hit = out["results"][0]
    assert "chunk_id" in hit
    assert "preview" in hit
    assert "content" not in hit
    assert hit["relevance"] in {"high", "medium", "low", "uncertain"}
    assert isinstance(hit["matched"], bool)
    assert hit["chunk_role"] == "executable"

    full = server.do_get_chunk(str(proj), hit["chunk_id"])
    assert full["content"].startswith("def add_numbers")
    assert full["source_type"] == "static_indexed_source"

    (proj / "math_utils.py").write_text("def add_numbers(a, b):\n    return 999\n", encoding="utf-8")
    stale = server.do_search(str(proj), "add numbers", k=1)
    assert stale["dirty"]["stale"] is True
    assert stale["results"][0]["stale"] is True


def test_content_modes_and_min_relevance_filter(tmp_path, monkeypatch):
    from engram_mcp import server

    proj, provider = _indexed_project(tmp_path)
    monkeypatch.setattr(server, "_provider_for_query_model", lambda _model_id: provider)
    none = server.do_search(str(proj), "add numbers", content="none")
    assert "preview" not in none["results"][0]
    assert "content" not in none["results"][0]

    full = server.do_search(str(proj), "add numbers", content="full", max_chars_per_result=10)
    assert "content" in full["results"][0]
    assert full["results"][0]["truncated"] is True

    high = server.do_search(str(proj), "unknown thing", min_relevance="high")
    assert all(r["relevance"] == "high" for r in high["results"])


def test_relevance_bucket_thresholds():
    assert retrieval.relevance_bucket(0.90) == "high"
    assert retrieval.relevance_bucket(0.60) == "medium"
    assert retrieval.relevance_bucket(0.30) == "low"
    assert retrieval.relevance_bucket(0.10) == "uncertain"


def test_hybrid_fts_degradation_warning(tmp_path, monkeypatch):
    proj, provider = _indexed_project(tmp_path)

    def degraded(self, query, k=8, where=None):
        return [], "fts unavailable in test"

    monkeypatch.setattr(LanceStore, "search_text_with_status", degraded)
    out = search_project(proj, provider, "add_numbers", mode="hybrid", return_meta=True)
    assert out["mode_requested"] == "hybrid"
    assert out["mode_used"] == "vector"
    assert out["warnings"] == ["fts unavailable in test"]
    assert out["rerank_applied"] is False


def test_list_projects_reports_broken_inventory(tmp_path):
    from engram_mcp import server

    projects = paths.data_home() / "projects"
    broken = projects / "broken-id"
    broken.mkdir(parents=True)
    (broken / "project.json").write_text("{broken", encoding="utf-8")
    out = server.list_projects()
    assert out["data_home"] == str(paths.data_home(create=False))
    assert out["projects"] == []
    assert out["errors"]
    assert out["errors"][0]["project_id"] == "broken-id"
    assert out["errors"][0]["code"] == errors.E_INDEX_INVALID


def test_chunk_role_heuristics():
    assert derive_chunk_role("src/app.py", "python", "function_definition") == "executable"
    assert derive_chunk_role("tests/test_app.py", "python", "function_definition") == "test"
    assert derive_chunk_role("templates/page.html", "html", "file") == "template"
    assert derive_chunk_role("pyproject.toml", "toml", "file") == "config"
    assert derive_chunk_role("README.md", "markdown", "section") == "comment"


def test_find_definition_suggestions(tmp_path):
    from engram_mcp import server

    proj, _provider = _indexed_project(tmp_path)
    out = server.do_find_definition(str(proj), "EmbeddingCach")
    assert out["count"] == 0
    assert out["results"] == []
    assert out["suggestions"]
    assert out["suggestions"][0]["symbol"] == "EmbeddingCache"


def test_model_status_does_not_load_model(tmp_path):
    from engram_mcp import server
    from engram_mcp.embeddings import factory

    proj = tmp_path / "proj"
    proj.mkdir()
    _create_one_row_table(
        proj,
        embedder_id=factory.CANONICAL_EMBEDDER_ID,
        dim=config.DEFAULT_EMBED_DIM,
    )
    out = server.do_model_status(str(proj))
    assert out["status"] in {"not_loaded", "loaded"}
    assert out["model_id"] == factory.CANONICAL_EMBEDDER_ID
    assert out["backend_id"] == factory.CANONICAL_EMBEDDER_ID


def test_model_status_rejects_removed_embedder_without_load(tmp_path):
    from engram_mcp import server

    proj = tmp_path / "proj"
    proj.mkdir()
    _create_one_row_table(proj, embedder_id="st:Qwen/Qwen3-Embedding-4B@1024#query/none")
    out = server.do_model_status(str(proj))
    assert out["code"] == errors.E_UNKNOWN_PROFILE


def test_manifest_compat_uses_canonical_model_id():
    from engram_mcp.embeddings import factory

    class CanonicalProvider:
        model_id = factory.CANONICAL_EMBEDDER_ID
        backend_id = f"st:{factory.GRANITE_MODEL}@full#none/none:cuda"
        dim = config.DEFAULT_EMBED_DIM

    m = manifest.ProjectManifest(
        project_id="p",
        root_path="/tmp/p",
        active_table="chunks",
        embedder_id=factory.CANONICAL_EMBEDDER_ID,
        dim=config.DEFAULT_EMBED_DIM,
        chunker_version=config.CHUNKER_VERSION,
    )
    assert _is_compatible(m, CanonicalProvider())


def test_index_job_errors_are_structured(tmp_path, monkeypatch):
    from engram_mcp import server

    job = server._registry.create(str(tmp_path))

    def fail(_device):
        raise errors.EngramError("missing extra", errors.E_EXTRA_MISSING, hint="install gpu")

    monkeypatch.setattr(server, "_get_provider", fail)
    server._index_worker(job.job_id, str(tmp_path), False, "cuda")
    status = server.get_status(job.job_id)
    assert status["status"] == "error"
    assert status["error"] == "missing extra"
    assert status["code"] == errors.E_EXTRA_MISSING
    assert status["hint"] == "install gpu"
