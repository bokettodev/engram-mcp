"""Agent-grade MCP contract tests that do not load real embedding models."""

from __future__ import annotations

import io
import json
import threading
import time
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


class CanonicalFakeProvider(FakeProvider):
    from engram_mcp.embeddings import factory

    model_id = factory.CANONICAL_EMBEDDER_ID
    backend_id = model_id


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


def _indexed_canonical_project(tmp_path: Path) -> tuple[Path, CanonicalFakeProvider]:
    provider = CanonicalFakeProvider()
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


def test_list_projects_compact_pagination_and_orphan_prune(tmp_path):
    from engram_mcp import server
    from engram_mcp.embeddings import factory

    roots = []
    for name in ("alpha", "beta", "gamma"):
        root = tmp_path / name
        root.mkdir()
        roots.append(root)
        pdir = paths.project_dir(root)
        manifest.save_project(
            pdir,
            manifest.ProjectManifest(
                project_id=paths.project_id_for(root),
                root_path=str(root.resolve()),
                active_table="chunks",
                generation=3,
                embedder_id=factory.CANONICAL_EMBEDDER_ID,
                dim=config.DEFAULT_EMBED_DIM,
                chunker_version=config.CHUNKER_VERSION,
                files=2,
                chunks=7,
                indexed_at=123.0,
            ),
        )

    first = server.list_projects(limit=2, prune_orphans=False)
    assert len(first["projects"]) == 2
    assert first["cursor"] is not None
    assert set(first["projects"][0]) == {
        "project_id",
        "root_path",
        "root_exists",
        "files",
        "chunks",
        "indexed_at",
        "embedder_id",
        "generation",
    }
    assert "table_rows" not in first["projects"][0]

    second = server.list_projects(limit=2, cursor=first["cursor"], prune_orphans=False)
    assert len(second["projects"]) == 1
    assert second["cursor"] is None

    orphan_root = roots[0]
    orphan_pdir = paths.project_dir(orphan_root, create=False)
    for child in orphan_root.iterdir():
        child.unlink()
    orphan_root.rmdir()
    pruned = server.list_projects(limit=10, prune_orphans=True)
    assert any(item["root_path"] == str(orphan_root.resolve()) for item in pruned["gc"]["pruned"])
    assert not orphan_pdir.exists()


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


def test_do_search_waits_for_model_future_then_succeeds(tmp_path, monkeypatch):
    from engram_mcp import server
    from engram_mcp.embeddings import factory

    proj, provider = _indexed_canonical_project(tmp_path)
    server._model_loads.clear()
    monkeypatch.setattr(factory, "is_model_loaded", lambda _backend_id: False)
    monkeypatch.setenv("ENGRAM_SEARCH_WAIT_SEC", "1")

    def load(_model_id):
        time.sleep(0.05)
        return provider

    monkeypatch.setattr(server, "_provider_load_worker", load)
    out = server.do_search(str(proj), "add numbers", k=1)
    assert out["count"] == 1
    assert out["results"][0]["symbol"] == "add_numbers"
    server._model_loads.clear()


def test_do_search_returns_model_loading_after_wait_budget(tmp_path, monkeypatch):
    from engram_mcp import server
    from engram_mcp.embeddings import factory

    proj, provider = _indexed_canonical_project(tmp_path)
    server._model_loads.clear()
    monkeypatch.setattr(factory, "is_model_loaded", lambda _backend_id: False)
    monkeypatch.setenv("ENGRAM_SEARCH_WAIT_SEC", "0.01")

    def load(_model_id):
        time.sleep(0.15)
        return provider

    monkeypatch.setattr(server, "_provider_load_worker", load)
    out = server.do_search(str(proj), "add numbers", k=1)
    assert out["code"] == errors.E_MODEL_LOADING
    assert out["retry_after_sec"] > 0
    for fut in list(server._model_loads.values()):
        fut.result(timeout=1)
    server._model_loads.clear()


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


def test_cpu_index_job_errors_are_structured(tmp_path, monkeypatch):
    # The CPU (in-process) path structures a provider-construction failure.
    from engram_mcp import server

    job = server._registry.create(str(tmp_path))

    def fail(_device):
        raise errors.EngramError("missing extra", errors.E_EXTRA_MISSING, hint="install gpu")

    monkeypatch.setattr(server, "_get_provider", fail)
    server._index_worker(job.job_id, str(tmp_path), False, "cpu")
    status = server.get_status(job.job_id)
    assert status["status"] == "error"
    assert status["error"] == "missing extra"
    assert status["code"] == errors.E_EXTRA_MISSING
    assert status["hint"] == "install gpu"


class _StreamingStdout:
    def __init__(self, lines):
        self._lines = list(lines)

    def __iter__(self):
        return iter(self._lines)


class _FakePopen:
    def __init__(self, lines, returncode=0, stderr=""):
        self.stdout = _StreamingStdout(lines)
        self.stderr = io.StringIO(stderr)
        self.returncode = returncode

    def wait(self):
        return self.returncode


def test_gpu_index_job_error_from_subprocess_is_structured(tmp_path, monkeypatch):
    # CUDA jobs run in a subprocess; its JSON error must surface in job status.
    from engram_mcp import server

    job = server._registry.create(str(tmp_path))
    out = '{"event": "result", "version": 1, "ok": false, "error": "missing extra", "code": "E_EXTRA_MISSING", "hint": "install gpu"}\n'
    monkeypatch.setattr(server.subprocess, "Popen", lambda *a, **k: _FakePopen([out], 2))
    server._index_worker(job.job_id, str(tmp_path), False, "cuda")
    status = server.get_status(job.job_id)
    assert status["status"] == "error"
    assert status["code"] == errors.E_EXTRA_MISSING
    assert status["hint"] == "install gpu"


def test_gpu_index_job_streams_progress_and_result(tmp_path, monkeypatch):
    from engram_mcp import server

    job = server._registry.create(str(tmp_path))
    lines = [
        '{"event": "progress", "version": 1, "seq": 1, "stage": "waiting_for_gpu", "unit": "lock", "done": 0, "total": null}\n',
        '{"event": "progress", "version": 1, "seq": 2, "stage": "embedding", "unit": "embeddings", "done": 4, "total": 12, "chunks": 12, "embedded": 4, "reused": 1}\n',
        '{"event": "result", "version": 1, "ok": true, "mode": "full", "files": 3, "chunks": 12, "embedded_unique": 12, '
        '"reused_unique": 1, "embedder_id": "fastembed:granite", "backend_id": "st:granite:cuda", '
        '"device": "cuda", "seconds": 0.1}\n',
    ]
    monkeypatch.setattr(server.subprocess, "Popen", lambda *a, **k: _FakePopen(lines, 0))
    server._index_worker(job.job_id, str(tmp_path), False, "auto")  # auto also routes to the subprocess
    status = server.get_status(job.job_id)
    assert status["status"] == "done"
    assert status["chunks"] == 12
    assert status["embedder_id"] == "fastembed:granite"
    assert status["progress"] == {"unit": "chunks", "done": 12, "total": 12}
    assert status["update_seq"] >= 4


def test_subprocess_progress_updates_registry_before_result(tmp_path, monkeypatch):
    from engram_mcp import server

    first_progress_seen = threading.Event()
    allow_result = threading.Event()

    class BlockingStdout:
        def __iter__(self):
            yield '{"event": "progress", "version": 1, "seq": 1, "stage": "embedding", "unit": "embeddings", "done": 2, "total": 5, "chunks": 5, "embedded": 2, "reused": 0}\n'
            first_progress_seen.set()
            assert allow_result.wait(timeout=2)
            yield '{"event": "result", "version": 1, "ok": true, "mode": "full", "files": 1, "chunks": 5, "embedded_unique": 5, "reused_unique": 0, "embedder_id": "fastembed:granite", "backend_id": "st:granite:cuda", "device": "cuda", "seconds": 0.1}\n'

    class BlockingPopen:
        stdout = BlockingStdout()
        stderr = io.StringIO("")
        returncode = 0

        def wait(self):
            return self.returncode

    job = server._registry.create(str(tmp_path))
    monkeypatch.setattr(server.subprocess, "Popen", lambda *a, **k: BlockingPopen())

    thread = threading.Thread(
        target=server._subprocess_index,
        args=(job.job_id, str(tmp_path), False, "cuda"),
    )
    thread.start()
    assert first_progress_seen.wait(timeout=2)
    mid = server.get_status(job.job_id)
    assert mid["stage"] == "embedding"
    assert mid["progress"] == {"unit": "embeddings", "done": 2, "total": 5}
    allow_result.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    final = server.get_status(job.job_id)
    assert final["status"] == "done"
