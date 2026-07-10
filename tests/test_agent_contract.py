"""Agent-grade MCP contract tests that do not load real embedding models."""

from __future__ import annotations

import asyncio
import io
import json
import subprocess
import sys
import types
import threading
import time
from pathlib import Path
from typing import Sequence

import pytest

from engram_mcp import config, diagnostics, errors, manifest, paths, retrieval
from engram_mcp.pipeline import (
    ProjectNotIndexedError,
    derive_chunk_role,
    doctor_project,
    index_project,
    load_query_index,
    reindex_file,
    search_project,
    _search_count_metadata,
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
        logical_project_id=paths.project_id_for(root),
        checkout_kind="non_git",
        active_table="chunks",
        embedder_id="test:fake",
        dim=dim,
        chunker_version=config.CHUNKER_VERSION,
        chunk_id_scheme=config.CHUNK_ID_SCHEME,
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
        generation=0,
        active_table="chunks",
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
            logical_project_id=paths.project_id_for(proj),
            checkout_kind="non_git",
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


def test_doctor_project_never_creates_missing_lancedb_dir(tmp_path, monkeypatch):
    """doctor_project is a read path: a valid manifest whose lancedb/ dir is
    missing must be reported as an error issue, not silently created by
    `lancedb.connect` (which creates a missing directory as a side effect).
    """

    proj = tmp_path / "proj"
    proj.mkdir()
    pdir = _write_manifest_only(proj)
    lancedb_dir = pdir / "lancedb"
    assert not lancedb_dir.exists()

    monkeypatch.setenv("ENGRAM_READONLY", "1")
    health = doctor_project(proj, check_git=False)

    assert health["ok"] is False
    assert any(issue["code"] == "table_missing" for issue in health["issues"])
    assert not lancedb_dir.exists()


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

    proj, provider = _indexed_canonical_project(tmp_path)
    monkeypatch.setattr(server, "_provider_for_query_model", lambda _model_id: provider)
    out = server.do_search(str(proj), "add numbers", k=1)
    assert "error" not in out
    assert out["count"] == 1


def test_compact_search_get_chunk_relevance_and_stale(tmp_path, monkeypatch):
    from engram_mcp import server

    proj, provider = _indexed_canonical_project(tmp_path)
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


def test_search_shape_facets_budget_and_catalog_tools(tmp_path, monkeypatch):
    from engram_mcp import catalog, server

    proj, provider = _indexed_canonical_project(tmp_path)
    monkeypatch.setattr(server, "_provider_for_query_model", lambda _model_id: provider)

    out = server.do_search(
        str(proj),
        "add numbers",
        k=2,
        facets=["language", "kind", "chunk_role"],
        max_total_chars=12,
    )
    assert out["count"] == 2
    assert "map" not in out  # CUT 1: top-level map[] removed, derivable from results[]
    assert out["body_chars"] <= 12
    assert out["truncated"] is True
    assert "total_matches" in out
    assert out["total_matches"]["vector_estimate"]["exact"] is False
    assert out["facets"]["fields"]["language"]["python"] >= 1
    assert "matched_in" in out["results"][0]

    pdir = paths.project_dir(proj, create=False)
    m = manifest.load_project(pdir)
    data = catalog.load_catalog(pdir, m.generation)
    assert data is not None
    assert data["totals"]["files"] == 2
    assert "content" not in json.dumps(data)

    mapped = server.do_project_map(str(proj), depth=1)
    assert mapped["totals"]["files"] == 2
    assert mapped["dirs"]
    assert mapped["files"] == []
    assert mapped["files_page"]["included"] is False

    mapped_files = server.do_project_map(str(proj), depth=1, include_files=True, files_limit=1)
    assert len(mapped_files["files"]) == 1
    assert "symbols" not in mapped_files["files"][0]
    assert "symbols_count" in mapped_files["files"][0]

    mapped_symbols = server.do_project_map(
        str(proj),
        depth=1,
        include_files=True,
        include_symbols=True,
        symbols_limit=1,
    )
    assert len(mapped_symbols["files"][0]["symbols"]) <= 1

    health = server.do_doctor_project(str(proj), check_git=False)
    assert health["ok"] is True
    assert health["summary"]["manifest_files"] == 2

    grep = diagnostics.grep_index(str(proj), "EmbeddingCache")
    assert grep["total_matches"] >= 1
    assert grep["results"][0]["path"] == "cache.py"


def test_incremental_catalog_crash_window_leaves_unavailable_catalog(tmp_path, monkeypatch):
    from engram_mcp import catalog, server

    proj, provider = _indexed_canonical_project(tmp_path)
    monkeypatch.setattr(server, "_provider_for_query_model", lambda _model_id: provider)
    (proj / "new_file.py").write_text("def new_marker():\n    return 3\n", encoding="utf-8")

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated catalog write crash")

    monkeypatch.setattr(catalog, "save_catalog", boom)
    with pytest.raises(RuntimeError, match="simulated catalog write crash"):
        index_project(proj, provider)

    pdir = paths.project_dir(proj, create=False)
    m = manifest.load_project(pdir)
    assert catalog.load_catalog(pdir, m.generation) is None

    out = server.do_search(str(proj), "new marker", k=2)
    assert out["count"] >= 1
    assert any("catalog sidecar unavailable" in w for w in out["warnings"])

    mapped = server.do_project_map(str(proj), depth=1)
    assert mapped["code"] == errors.E_INDEX_INVALID


def test_same_generation_catalog_drift_is_caught_by_doctor_not_search(tmp_path, monkeypatch):
    """A table mutated outside the normal write path (same generation/active_table,
    catalog+manifest token untouched) is the one class of drift the search-time
    O(1) token check cannot see -- it never re-scans the table, only compares
    the token written alongside the catalog at build time (see
    pipeline._catalog_validation_error). That full id-set comparison still
    exists -- pipeline._catalog_deep_validation_error, used only by
    doctor_project -- so this drift is still detected, just not on every
    search. This is a deliberate move (item 1 of the search-hot-path audit),
    not a regression: before it, this same drift made project_map fail with
    E_INDEX_INVALID on every call; now project_map/search succeed (the token
    still matches what was written) and only doctor_project flags it.
    """
    from engram_mcp import catalog, server

    proj, provider = _indexed_canonical_project(tmp_path)
    monkeypatch.setattr(server, "_provider_for_query_model", lambda _model_id: provider)
    pdir = paths.project_dir(proj, create=False)
    m = manifest.load_project(pdir)
    stale_catalog = catalog.load_catalog(pdir, m.generation)
    assert stale_catalog is not None
    assert m.catalog_token
    assert stale_catalog["catalog_token"] == m.catalog_token

    LanceStore(pdir / "lancedb", provider.dim, table=m.active_table).add(
        [
            {
                "chunk_id": "drift-row",
                "rel_path": "drift.py",
                "language": "python",
                "symbol": "drift",
                "symbol_kind": "function_definition",
                "chunk_role": "executable",
                "start_line": 1,
                "end_line": 2,
                "content": "def drift():\n    return 4\n",
                "search_text": "path: drift.py\nsymbol: drift\n\ndef drift():\n    return 4\n",
                "file_hash": "drift-file",
                "chunk_hash": "drift-chunk",
                "vector": [0.0, 0.0, 0.0, 1.0],
            }
        ]
    )

    # Search-time O(1) validation only compares the (untouched) token, so it
    # does not notice the table now has one more row than the catalog knows
    # about -- non-vacuous proof that the hot path no longer scans the table.
    mapped = server.do_project_map(str(proj), depth=1)
    assert "code" not in mapped

    # doctor_project still does the full id-set comparison and catches it.
    health = server.do_doctor_project(str(proj), check_git=False)
    assert any(issue["code"] == "catalog_count_mismatch" for issue in health["issues"])


def test_catalog_token_mismatch_is_detected_and_degrades(tmp_path, monkeypatch):
    """The O(1) search-time check (pipeline._catalog_validation_error) compares
    only the commit token written into both the catalog sidecar and the
    project manifest at build time. If that token is corrupted/tampered on
    either side -- simulating drift the write-path invariant is supposed to
    prevent -- search-time validation must still catch it and degrade
    (project_map fails loud; search falls back to a warning, same as any
    other catalog-unavailable case), even though it never re-scans the table.
    """
    from engram_mcp import catalog, server

    proj, provider = _indexed_canonical_project(tmp_path)
    monkeypatch.setattr(server, "_provider_for_query_model", lambda _model_id: provider)
    pdir = paths.project_dir(proj, create=False)
    m = manifest.load_project(pdir)
    assert m.catalog_token

    data = catalog.load_catalog(pdir, m.generation)
    assert data is not None
    assert data["catalog_token"] == m.catalog_token
    data["catalog_token"] = "tampered-token-does-not-match-manifest"
    catalog.save_catalog(pdir, data)

    mapped = server.do_project_map(str(proj), depth=1)
    assert mapped.get("code") == errors.E_INDEX_INVALID
    assert "token" in mapped["hint"]

    out = server.do_search(str(proj), "add numbers", k=1)
    assert out["count"] == 1  # search itself still works: catalog is optional there
    assert any("catalog sidecar unavailable" in w for w in out["warnings"])


def test_catalog_token_written_under_same_lock_as_catalog(tmp_path, monkeypatch):
    """The token is computed once, at build time, and written verbatim to both
    the catalog sidecar and the project manifest before the project lock is
    released -- so a freshly built (full rebuild, incremental, and
    single-file reindex) generation always has the two in agreement.
    """
    from engram_mcp import catalog

    provider = FakeProvider()
    proj = _write_project(tmp_path)
    index_project(proj, provider, full_rebuild=True)
    pdir = paths.project_dir(proj, create=False)

    m = manifest.load_project(pdir)
    data = catalog.load_catalog(pdir, m.generation)
    assert m.catalog_token
    assert data["catalog_token"] == m.catalog_token

    # Incremental update: still in agreement afterward.
    (proj / "math_utils.py").write_text(
        "def add_numbers(a, b):\n    return a + b + 0\n", encoding="utf-8"
    )
    index_project(proj, provider)
    m2 = manifest.load_project(pdir)
    data2 = catalog.load_catalog(pdir, m2.generation)
    assert m2.catalog_token
    assert data2["catalog_token"] == m2.catalog_token
    assert m2.catalog_token != m.catalog_token  # table content changed -> token changed

    # Single-file reindex: still in agreement afterward.
    from engram_mcp.pipeline import reindex_file

    reindex_file(proj, provider, "cache.py")
    m3 = manifest.load_project(pdir)
    data3 = catalog.load_catalog(pdir, m3.generation)
    assert m3.catalog_token
    assert data3["catalog_token"] == m3.catalog_token


def test_malformed_catalog_warns_but_search_returns_hits(tmp_path, monkeypatch):
    from engram_mcp import catalog, server

    proj, provider = _indexed_canonical_project(tmp_path)
    monkeypatch.setattr(server, "_provider_for_query_model", lambda _model_id: provider)
    pdir = paths.project_dir(proj, create=False)
    m = manifest.load_project(pdir)
    catalog.catalog_path(pdir, m.generation).write_text(
        json.dumps(
            {
                "schema_version": catalog.SCHEMA_VERSION,
                "generation": None,
                "active_table": m.active_table,
                "totals": {"files": 0, "chunks": 0, "symbols": 0},
                "files": [],
            }
        ),
        encoding="utf-8",
    )

    out = server.do_search(str(proj), "add numbers", k=1)
    assert out["count"] == 1
    assert "error" not in out
    assert any("catalog sidecar unavailable" in w for w in out["warnings"])


def test_grep_index_pathological_regex_times_out(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGRAM_GREP_REGEX_TIMEOUT_SEC", "0.2")
    proj = tmp_path / "proj"
    proj.mkdir()
    pdir = _write_manifest_only(proj, chunks=1)
    content = ("a" * 5000) + "!"
    LanceStore(pdir / "lancedb", 4, table="chunks").create(
        [
            {
                "chunk_id": "slow",
                "rel_path": "slow.py",
                "language": "python",
                "symbol": "",
                "symbol_kind": "",
                "chunk_role": "comment",
                "start_line": 1,
                "end_line": 1,
                "content": content,
                "search_text": content,
                "file_hash": "h",
                "chunk_hash": "ch",
                "vector": [0.0, 0.0, 0.0, 1.0],
            }
        ]
    )
    manifest.save_files(
        pdir,
        {
            "slow.py": {
                "file_hash": "h",
                "mtime_ns": 1,
                "size": len(content),
                "language": "python",
                "chunks": 1,
            }
        },
        generation=0,
        active_table="chunks",
    )

    out = diagnostics.grep_index(str(proj), r"(a+)+$")
    assert "error" not in out
    assert out["status"] == "partial"
    assert "timed out" in out["warning"]


def test_grep_index_limits_are_clamped_with_warning(tmp_path, monkeypatch):
    from engram_mcp.pipeline import MAX_GREP_LIMIT, MAX_GREP_MAX_MATCHES, MAX_GREP_SCAN_CHUNKS

    proj, _provider = _indexed_project(tmp_path)

    out = diagnostics.grep_index(
        str(proj),
        "def",
        limit=MAX_GREP_LIMIT + 50,
        max_matches=MAX_GREP_MAX_MATCHES + 50,
        max_scan_chunks=MAX_GREP_SCAN_CHUNKS + 50,
    )
    assert out["limit"] == MAX_GREP_LIMIT
    assert out["max_matches"] == MAX_GREP_MAX_MATCHES
    assert out["max_scan_chunks"] == MAX_GREP_SCAN_CHUNKS
    assert any("limit clamped" in w for w in out["warnings"])
    assert any("max_matches clamped" in w for w in out["warnings"])
    assert any("max_scan_chunks clamped" in w for w in out["warnings"])

    # A within-budget request is untouched and carries no clamp warning.
    within_budget = diagnostics.grep_index(str(proj), "def", limit=10)
    assert within_budget["warnings"] == []


def test_cli_grep_json_output(tmp_path, capsys):
    # grep_index is not an MCP tool (CUT 2): the capability survives only via
    # `engram grep` (this CLI path) and engram_mcp.diagnostics.grep_index.
    from engram_mcp import cli

    proj, _provider = _indexed_project(tmp_path)
    code = cli.cmd_grep(
        types.SimpleNamespace(
            path=str(proj),
            pattern="EmbeddingCache",
            ignore_case=False,
            limit=50,
            offset=0,
            max_matches=500,
            max_scan_chunks=10000,
            include_lines=False,
            json=True,
        )
    )
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out.strip())
    assert payload["event"] == "result"
    assert payload["total_matches"] >= 1
    assert payload["results"][0]["path"] == "cache.py"


def test_cli_grep_human_output(tmp_path, capsys):
    from engram_mcp import cli

    proj, _provider = _indexed_project(tmp_path)
    code = cli.cmd_grep(
        types.SimpleNamespace(
            path=str(proj),
            pattern="EmbeddingCache",
            ignore_case=False,
            limit=50,
            offset=0,
            max_matches=500,
            max_scan_chunks=10000,
            include_lines=False,
            json=False,
        )
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "cache.py" in captured.out
    assert "matches:" in captured.out


def test_grep_index_not_registered_as_mcp_tool():
    from engram_mcp import server

    names = {t.name for t in asyncio.run(server.mcp.list_tools())}
    assert "grep_index" not in names
    assert not hasattr(server, "do_grep_index")


def test_rerank_candidate_k_env_default_and_validation(tmp_path, monkeypatch):
    from engram_mcp import server

    proj, provider = _indexed_canonical_project(tmp_path)
    monkeypatch.setattr(server, "_provider_for_query_model", lambda _model_id: provider)
    monkeypatch.setenv("ENGRAM_RERANK_CANDIDATE_K", "7")

    out = server.do_search(str(proj), "add numbers", k=1)
    assert out["candidate_k"] == 7

    explicit = server.do_search(str(proj), "add numbers", k=1, candidate_k=9)
    assert explicit["candidate_k"] == 9


def test_search_k_over_budget_is_clamped_with_warning(tmp_path, monkeypatch):
    """k above MAX_SEARCH_K is clamped (not rejected); k below 1 is still an
    error -- see pipeline._validate_search_k / item 3 of the search-hot-path
    audit. Confirmed non-vacuous against the pre-fix behavior by the test
    edit history: this used to assert E_BAD_REQUEST for an over-budget k.
    """
    from engram_mcp import server
    from engram_mcp.pipeline import MAX_SEARCH_K

    proj, provider = _indexed_canonical_project(tmp_path)
    monkeypatch.setattr(server, "_provider_for_query_model", lambda _model_id: provider)

    clamped = server.do_search(str(proj), "add numbers", k=MAX_SEARCH_K + 25)
    assert "code" not in clamped
    assert any(f"k clamped to server maximum {MAX_SEARCH_K}" in w for w in clamped["warnings"])

    still_bad = server.do_search(str(proj), "add numbers", k=0)
    assert still_bad["code"] == errors.E_BAD_REQUEST


def test_search_response_char_budgets_are_clamped_with_warning(tmp_path, monkeypatch):
    from engram_mcp import server
    from engram_mcp.server import _MAX_RESULT_CHARS, _MAX_TOTAL_CHARS

    proj, provider = _indexed_canonical_project(tmp_path)
    monkeypatch.setattr(server, "_provider_for_query_model", lambda _model_id: provider)

    out = server.do_search(
        str(proj),
        "add numbers",
        k=1,
        content="full",
        max_chars_per_result=_MAX_RESULT_CHARS + 5000,
        max_total_chars=_MAX_TOTAL_CHARS + 5000,
    )
    assert "code" not in out
    assert out["max_chars_per_result"] == _MAX_RESULT_CHARS
    assert out["max_total_chars"] == _MAX_TOTAL_CHARS
    assert any("max_chars_per_result clamped" in w for w in out["warnings"])
    assert any("max_total_chars clamped" in w for w in out["warnings"])

    # Sub-minimum is still a request error, not a clamp.
    bad = server.do_search(str(proj), "add numbers", k=1, max_chars_per_result=0)
    assert bad["code"] == errors.E_BAD_REQUEST

    # get_chunk's max_chars follows the same rule.
    found = server.do_find_definition(str(proj), "EmbeddingCache")
    chunk_id = found["results"][0]["chunk_id"]
    chunk = server.do_get_chunk(str(proj), chunk_id, max_chars=_MAX_RESULT_CHARS + 1)
    assert any("max_chars clamped" in w for w in chunk.get("warnings") or [])


def test_search_response_source_revision_branch_mismatch_top_level(tmp_path, monkeypatch):
    from engram_mcp import gitmeta, server

    proj, provider = _indexed_canonical_project(tmp_path)
    monkeypatch.setattr(server, "_provider_for_query_model", lambda _model_id: provider)

    indexed = {
        "git_worktree_root": "C:/repo",
        "indexed_ref": "main",
        "indexed_commit": "1111111111111111111111111111111111111111",
        "indexed_dirty": False,
    }
    current = {
        "git_worktree_root": "C:/repo",
        "indexed_ref": "feature/x",
        "indexed_commit": "2222222222222222222222222222222222222222",
        "indexed_dirty": True,
    }

    monkeypatch.setattr(
        gitmeta,
        "current_staleness",
        lambda _root, _indexed: {
            "available": True,
            "git_stale": True,
            "reasons": ["indexed_ref", "indexed_commit", "indexed_dirty"],
            "indexed": indexed,
            "current": current,
        },
    )

    out = server.do_search(str(proj), "add numbers", k=1, content="none")
    revision = out["source_revision"]
    assert revision == {
        "available": True,
        "indexed": {
            "worktree_root": "C:/repo",
            "ref": "main",
            "commit": "1111111111111111111111111111111111111111",
            "dirty": False,
        },
        "current": {
            "worktree_root": "C:/repo",
            "ref": "feature/x",
            "commit": "2222222222222222222222222222222222222222",
            "dirty": True,
        },
        "stale": True,
        "branch_mismatch": True,
        "commit_mismatch": True,
        "dirty_mismatch": True,
        "reasons": ["indexed_ref", "indexed_commit", "indexed_dirty"],
    }
    assert any("indexed ref 'main'" in w and "feature/x" in w for w in out["warnings"])
    assert "git" not in out
    assert "git_stale" not in out["results"][0]
    assert "stale" in out["results"][0]
    assert "source_revision" not in out["results"][0]


def test_search_response_source_revision_unavailable_has_no_mismatch(tmp_path, monkeypatch):
    from engram_mcp import gitmeta

    proj, provider = _indexed_project(tmp_path)
    monkeypatch.setattr(
        gitmeta,
        "current_staleness",
        lambda _root, _indexed: {
            "available": False,
            "git_stale": False,
            "reasons": ["indexed_ref"],
            "indexed": {"indexed_ref": "main", "indexed_commit": "a", "indexed_dirty": False},
            "current": {"indexed_ref": "feature/x", "indexed_commit": "b", "indexed_dirty": True},
        },
    )

    out = search_project(proj, provider, "add numbers", k=1, return_meta=True)
    revision = out["source_revision"]
    assert revision["available"] is False
    assert revision["stale"] is False
    assert revision["branch_mismatch"] is False
    assert revision["commit_mismatch"] is False
    assert revision["dirty_mismatch"] is False
    assert revision["reasons"] == []
    assert not any("indexed ref" in w for w in out["warnings"])


def test_cli_search_prints_revision_warning_to_stderr(tmp_path, monkeypatch, capsys):
    from engram_mcp import cli, pipeline
    from engram_mcp.embeddings import factory

    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.setattr(
        pipeline,
        "load_query_index",
        lambda _root: types.SimpleNamespace(
            manifest=types.SimpleNamespace(embedder_id="test:fake")
        ),
    )
    monkeypatch.setattr(factory, "provider_for_model_id", lambda _model_id: object())
    monkeypatch.setattr(pipeline, "rerank_enabled", lambda: True)

    def fake_search_project(*_args, **_kwargs):
        return {
            "hits": [
                {
                    "rel_path": "math_utils.py",
                    "start_line": 1,
                    "end_line": 3,
                    "symbol_kind": "function_definition",
                    "symbol": "add_numbers",
                    "score": 1.0,
                    "content": "def add_numbers(a, b):\n    return a + b\n",
                }
            ],
            "source_revision": {
                "available": True,
                "indexed": {
                    "worktree_root": "C:/repo",
                    "ref": "feature/x",
                    "commit": "1111111111111111111111111111111111111111",
                    "dirty": False,
                },
                "current": {
                    "worktree_root": "C:/repo",
                    "ref": "feature/x",
                    "commit": "2222222222222222222222222222222222222222",
                    "dirty": False,
                },
                "stale": True,
                "branch_mismatch": False,
                "commit_mismatch": True,
                "dirty_mismatch": False,
                "reasons": ["indexed_commit"],
            },
        }

    monkeypatch.setattr(pipeline, "search_project", fake_search_project)
    code = cli.cmd_search(
        types.SimpleNamespace(
            path=str(proj),
            query="add numbers",
            k=1,
            lang=None,
            mode="vector",
            rerank=False,
        )
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "warning: results are from indexed commit '111111111111'" in captured.err
    assert "not current commit '222222222222'" in captured.err
    assert "warning:" not in captured.out
    assert "def add_numbers" in captured.out


def test_rerank_failure_degrades_to_base_ranking(tmp_path, monkeypatch):
    # Rerank is best-effort: a non-ImportError failure (e.g. model download
    # race, onnxruntime error) must NOT nuke the search — return base ranking.
    from engram_mcp import rerankers, server

    proj, provider = _indexed_canonical_project(tmp_path)
    monkeypatch.setattr(server, "_provider_for_query_model", lambda _model_id: provider)
    monkeypatch.setenv("ENGRAM_RERANK_ENABLED", "1")

    def boom(*_a, **_k):
        raise RuntimeError("onnxruntime exploded mid-rerank")

    monkeypatch.setattr(rerankers, "get_reranker", boom)

    out = server.do_search(str(proj), "add numbers", k=2, mode="vector", rerank=True)
    assert out["count"] == 2  # results survived, not an empty error payload
    assert out["rerank_applied"] is False
    assert any("rerank unavailable" in w for w in out["warnings"])


def test_rerank_master_switch_off_by_default(tmp_path, monkeypatch):
    # ENGRAM_RERANK_ENABLED is the master switch: off by default, and when off a
    # per-call rerank=true must NOT construct/load any reranker model.
    from engram_mcp import rerankers, server

    proj, provider = _indexed_canonical_project(tmp_path)
    monkeypatch.setattr(server, "_provider_for_query_model", lambda _model_id: provider)
    monkeypatch.delenv("ENGRAM_RERANK_ENABLED", raising=False)

    def must_not_load(*_a, **_k):
        raise AssertionError("reranker must not load when ENGRAM_RERANK_ENABLED is off")

    monkeypatch.setattr(rerankers, "get_reranker", must_not_load)

    out = server.do_search(str(proj), "add numbers", k=2, mode="vector", rerank=True)
    assert out["rerank_applied"] is False
    assert "ENGRAM_RERANK_ENABLED" in (out["rerank_skipped_reason"] or "")
    assert out["count"] == 2

    # flip the switch on -> vector-mode rerank now runs
    monkeypatch.setenv("ENGRAM_RERANK_ENABLED", "1")
    calls = {"n": 0}

    class _FakeReranker:
        model_id = "fake"

        def rerank(self, query, hits, top_k=None):
            calls["n"] += 1
            return hits[:top_k] if top_k else hits

    monkeypatch.setattr(rerankers, "get_reranker", lambda **_k: _FakeReranker())
    on = server.do_search(str(proj), "add numbers", k=2, mode="vector", rerank=True)
    assert on["rerank_applied"] is True
    assert calls["n"] == 1


def test_rerank_gated_to_vector_mode(tmp_path, monkeypatch):
    # Rerank must be skipped for hybrid-routed (identifier/literal) queries and
    # only run for vector mode. Reranking hybrid demotes exact symbol defs.
    from engram_mcp import rerankers, server

    proj, provider = _indexed_canonical_project(tmp_path)
    monkeypatch.setattr(server, "_provider_for_query_model", lambda _model_id: provider)
    monkeypatch.setenv("ENGRAM_RERANK_ENABLED", "1")

    calls = {"n": 0}

    class _FakeReranker:
        model_id = "fake-reranker"

        def rerank(self, query, hits, top_k=None):
            calls["n"] += 1
            return list(reversed(hits))[:top_k] if top_k else list(reversed(hits))

    monkeypatch.setattr(rerankers, "get_reranker", lambda **_k: _FakeReranker())

    hybrid = server.do_search(str(proj), "add numbers", k=2, mode="hybrid", rerank=True)
    assert hybrid["rerank_applied"] is False
    assert hybrid["rerank_skipped_reason"] is not None
    assert calls["n"] == 0  # reranker never constructed/called for hybrid

    vector = server.do_search(str(proj), "add numbers", k=2, mode="vector", rerank=True)
    assert vector["rerank_applied"] is True
    assert vector["rerank_skipped_reason"] is None
    assert calls["n"] == 1

    monkeypatch.setenv("ENGRAM_RERANK_CANDIDATE_K", "999")
    assert server._rerank_candidate_k_default() == 50
    monkeypatch.setenv("ENGRAM_RERANK_CANDIDATE_K", "0")
    assert server._rerank_candidate_k_default() == 1
    monkeypatch.setenv("ENGRAM_RERANK_CANDIDATE_K", "bad")
    assert server._rerank_candidate_k_default() == 20

    # candidate_k above the server budget is clamped (not rejected): the
    # search still runs, capped at MAX_RERANK_CANDIDATES, with a warning
    # instead of an error (see pipeline.search_project / item 3 of the
    # search-hot-path audit).
    clamped = server.do_search(str(proj), "add numbers", k=1, candidate_k=51)
    assert "code" not in clamped
    assert clamped["candidate_k"] == 50
    assert any("candidate_k clamped" in w for w in clamped["warnings"])

    still_bad = server.do_search(str(proj), "add numbers", k=1, candidate_k=0)
    assert still_bad["code"] == errors.E_BAD_REQUEST


def test_fts_count_metadata_labels_capped_lower_bound(monkeypatch):
    from engram_mcp.store import lancedb_store

    monkeypatch.setenv("ENGRAM_FTS_COUNT_MAX_SCAN", "3")
    assert lancedb_store.fts_count_max_scan() == 3
    monkeypatch.setenv("ENGRAM_FTS_COUNT_MAX_SCAN", "0")
    assert lancedb_store.fts_count_max_scan() == 1
    monkeypatch.setenv("ENGRAM_FTS_COUNT_MAX_SCAN", "bad")
    assert lancedb_store.fts_count_max_scan() == lancedb_store.DEFAULT_FTS_COUNT_MAX_SCAN

    class CappedStore:
        def fts_metadata(self, query, columns, where=None):
            return (
                [
                    {"chunk_id": "a", "rel_path": "a.py", "language": "python", "chunk_role": "executable"},
                    {"chunk_id": "b", "rel_path": "b.py", "language": "python", "chunk_role": "comment"},
                ],
                None,
                {"capped": True, "cap": 2, "limit": 2, "table_rows": 10},
            )

    qi = types.SimpleNamespace(store=CappedStore())
    total, facets, warnings = _search_count_metadata(
        qi,
        "needle",
        None,
        "hybrid",
        [],
        ["language", "chunk_role"],
        None,
    )
    assert total["fts_exact"] == {
        "available": True,
        "count": 2,
        "exact": False,
        "capped": True,
        "method": "lancedb_0_33_fts_metadata_scan",
    }
    assert facets["scope"] == "fts_capped_lower_bound"
    assert facets["exact"] is False
    assert facets["fields"]["language"]["python"] == 2
    assert "lower bound" in warnings[0]

    class ExactStore:
        def fts_metadata(self, query, columns, where=None):
            return (
                [{"chunk_id": "a", "rel_path": "a.py", "language": "python", "chunk_role": "executable"}],
                None,
                {"capped": False, "cap": 50, "limit": 10, "table_rows": 10},
            )

    total, facets, warnings = _search_count_metadata(
        types.SimpleNamespace(store=ExactStore()),
        "needle",
        None,
        "hybrid",
        [],
        ["language"],
        None,
    )
    assert total["fts_exact"]["exact"] is True
    assert total["fts_exact"]["capped"] is False
    assert facets["scope"] == "fts_exact"
    assert warnings == []


def test_fts_count_scan_skipped_when_facets_not_requested(monkeypatch):
    """Item 2 of the search-hot-path audit: in hybrid mode, the second FTS
    metadata scan (up to ENGRAM_FTS_COUNT_MAX_SCAN rows) must not run unless
    the caller actually requested facets. Proven directly (not just by
    absence of a slowdown): fts_metadata() raises if called at all.
    """

    class MustNotScanStore:
        def fts_metadata(self, query, columns, where=None):
            raise AssertionError("fts_metadata must not be called when facets were not requested")

    total, facets, warnings = _search_count_metadata(
        types.SimpleNamespace(store=MustNotScanStore()),
        "needle",
        None,
        "hybrid",
        [],
        [],  # no facets requested
        None,
    )
    assert facets is None
    assert warnings == []
    assert total["fts_exact"]["available"] is False
    assert total["fts_exact"]["count"] is None
    assert "reason" in total["fts_exact"]

    # Non-vacuity / no-regression: requesting facets in hybrid mode still
    # triggers the scan and still returns the same exact numbers as before.
    class ScanningStore:
        def __init__(self) -> None:
            self.calls = 0

        def fts_metadata(self, query, columns, where=None):
            self.calls += 1
            return (
                [{"chunk_id": "a", "rel_path": "a.py", "language": "python", "chunk_role": "executable"}],
                None,
                {"capped": False, "cap": 50, "limit": 1, "table_rows": 1},
            )

    store = ScanningStore()
    total2, facets2, _warnings2 = _search_count_metadata(
        types.SimpleNamespace(store=store), "needle", None, "hybrid", [], ["language"], None,
    )
    assert store.calls == 1
    assert total2["fts_exact"]["available"] is True
    assert total2["fts_exact"]["count"] == 1
    assert facets2["scope"] == "fts_exact"


def test_search_code_skips_fts_scan_end_to_end_when_facets_omitted(tmp_path, monkeypatch):
    """End-to-end version of the unit test above, through the real MCP
    search_code path: a hybrid-routed query (an exact identifier) with no
    facets requested must not touch the store's fts_metadata at all.
    """
    from engram_mcp import server
    from engram_mcp.store.lancedb_store import LanceStore as _LS

    proj, provider = _indexed_canonical_project(tmp_path)
    monkeypatch.setattr(server, "_provider_for_query_model", lambda _model_id: provider)

    def boom(self, query, columns, where=None):
        raise AssertionError("fts_metadata must not run when facets were not requested")

    monkeypatch.setattr(_LS, "fts_metadata", boom)

    out = server.do_search(str(proj), "EmbeddingCache", k=2)  # identifier -> hybrid mode
    assert out["mode_used"] == "hybrid"
    assert out["total_matches"]["fts_exact"]["available"] is False

    # Requesting facets flips the scan back on: it now reaches the monkeypatched
    # fts_metadata, which raises -- do_search converts that into an error
    # payload instead of silently succeeding, proving the scan path really is
    # reached once facets are requested (i.e. the first assertion above was
    # not vacuously true because this query type never scans).
    with_facets = server.do_search(str(proj), "EmbeddingCache", k=2, facets=["language"])
    assert "error" in with_facets


def test_get_chunk_neighbors_are_opt_in(tmp_path):
    from engram_mcp import server

    proj, _provider = _indexed_project(tmp_path)
    found = server.do_find_definition(str(proj), "EmbeddingCache")
    chunk_id = found["results"][0]["chunk_id"]
    plain = server.do_get_chunk(str(proj), chunk_id)
    assert "neighbors" not in plain
    expanded = server.do_get_chunk(str(proj), chunk_id, include_neighbors=True, include_parent=True)
    assert "neighbors" in expanded
    assert isinstance(expanded["neighbors"], list)


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


def test_plan_index_reports_missing_unique_without_loading_model(tmp_path):
    proj = _write_project(tmp_path)
    from engram_mcp.pipeline import index_project, plan_index

    provider = FakeProvider()
    before = plan_index(proj, model_id=provider.model_id, dim=provider.dim)
    assert before.mode == "full"
    assert before.missing_unique_chunks >= 1
    index_project(proj, provider)
    after = plan_index(proj, model_id=provider.model_id, dim=provider.dim)
    assert after.mode == "incremental"
    assert after.missing_unique_chunks == 0


def test_start_index_job_returns_before_planning_and_routes_small_delta_to_cpu(tmp_path, monkeypatch):
    """start_index_job must not walk/plan the project itself (that used to
    block the index_project tool call on a filesystem walk); planning and
    "auto" delta-cpu routing now happen inside the background job body."""
    from engram_mcp import server

    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    plan = types.SimpleNamespace(
        mode="incremental",
        files=1,
        chunks=1,
        added=0,
        changed=1,
        deleted=0,
        unchanged=0,
        missing_unique_chunks=1,
    )
    submitted = []
    plan_calls = []

    class Pool:
        def submit(self, fn, *args):
            submitted.append((fn, args))

    def fake_plan_index(*a, **k):
        plan_calls.append((a, k))
        return plan

    monkeypatch.setattr(server, "_plan_index", fake_plan_index)
    monkeypatch.setattr(server, "_index_pool", Pool())
    monkeypatch.setenv("ENGRAM_DELTA_CPU_MAX", "5")

    out = server.start_index_job(str(proj), index_device="auto")

    # The tool call itself never plans/walks: it only queued the job.
    assert plan_calls == []
    assert out["index_device"] == "auto"
    assert out["index_device_requested"] == "auto"
    assert out["coalesced"] is False
    assert "routing" not in out
    assert "plan" not in out
    assert submitted and submitted[0][0] is server._index_worker
    job_id = submitted[0][1][0]
    status_before = server.get_status(job_id)
    assert status_before["stage"] in ("", "planning", "queued")

    # Now run the queued worker body (as the real thread pool would) with the
    # subprocess dispatch stubbed out so this stays a unit test of routing.
    subprocess_calls = []
    monkeypatch.setattr(
        server, "_subprocess_index",
        lambda job_id, project_path, full_rebuild, setting, git_max_commits, git_analytics:
        subprocess_calls.append(setting),
    )
    fn, args = submitted[0]
    fn(*args)

    assert plan_calls  # planning happened inside the background job, not before
    status = server.get_status(job_id)
    assert status["index_device"] == "cpu"
    assert status["index_device_requested"] == "auto"
    assert status["routing"] == "delta_cpu"
    assert status["plan"]["missing_unique_chunks"] == 1
    assert subprocess_calls == ["cpu"]


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
                logical_project_id=paths.project_id_for(root),
                checkout_kind="non_git",
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
        "logical_project_id",
        "root_path",
        "root_exists",
        "checkout_kind",
        "indexed_ref",
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
        logical_project_id="p",
        checkout_kind="non_git",
        active_table="chunks",
        embedder_id=factory.CANONICAL_EMBEDDER_ID,
        dim=config.DEFAULT_EMBED_DIM,
        chunker_version=config.CHUNKER_VERSION,
        chunk_id_scheme=config.CHUNK_ID_SCHEME,
        schema_version=manifest.SCHEMA_VERSION,
    )
    assert _is_compatible(m, CanonicalProvider())


def test_cpu_index_job_subprocess_launch_failure_is_structured(tmp_path, monkeypatch):
    """CPU indexing now always dispatches to the subprocess (see the CPU
    indexing-blocks-search fix): a launch failure there must surface the same
    structured error as the GPU/auto path, and _get_provider/_run_index must
    never be consulted for a plain "cpu" request (ENGRAM_INPROCESS_CPU_MAX
    defaults to 0/disabled)."""
    from engram_mcp import server

    job = server._registry.create(str(tmp_path))

    def fail_popen(*_a, **_k):
        raise OSError("no such executable")

    monkeypatch.setattr(server.subprocess, "Popen", fail_popen)
    server._index_worker(job.job_id, str(tmp_path), False, "cpu")
    status = server.get_status(job.job_id)
    assert status["status"] == "error"
    assert "failed to launch index subprocess" in status["error"]
    assert status["code"] == errors.E_MODEL_LOAD_FAILED


def test_index_worker_planning_error_is_structured(tmp_path, monkeypatch):
    """When the optional in-process CPU fast path is enabled, _index_worker
    plans before routing/dispatching; a planning failure must end the job in
    a structured error without ever spawning a subprocess."""
    from engram_mcp import server

    monkeypatch.setenv("ENGRAM_INPROCESS_CPU_MAX", "10")
    job = server._registry.create(str(tmp_path))
    popen_calls = []
    monkeypatch.setattr(
        server.subprocess, "Popen", lambda *a, **k: popen_calls.append((a, k)) or None
    )

    def fail_plan(*_a, **_k):
        raise errors.EngramError("bad plan", errors.E_BAD_REQUEST, hint="fix the project")

    monkeypatch.setattr(server, "_plan_index", fail_plan)
    server._index_worker(job.job_id, str(tmp_path), False, "cpu")
    status = server.get_status(job.job_id)
    assert status["status"] == "error"
    assert status["error"] == "bad plan"
    assert status["code"] == errors.E_BAD_REQUEST
    assert status["hint"] == "fix the project"
    assert popen_calls == []


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


def test_index_job_never_constructs_cached_query_provider(tmp_path, monkeypatch):
    """No index path may embed inside the server process: with the optional
    in-process fast path disabled (the default, ENGRAM_INPROCESS_CPU_MAX=0),
    a "cpu" index job must never construct the cached FastEmbed singleton
    shared with search (embeddings/factory._fastembed_granite_cpu) -- only
    the subprocess it spawns is allowed to embed."""
    from engram_mcp import server
    from engram_mcp.embeddings import factory

    def fail_if_called():
        raise AssertionError(
            "an index job constructed the cached query provider -- it would "
            "share its inference lock with concurrent search"
        )

    monkeypatch.setattr(factory, "_fastembed_granite_cpu", fail_if_called)

    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")

    success_line = (
        '{"event": "result", "version": 1, "ok": true, "mode": "full", "files": 1, '
        '"chunks": 1, "embedded_unique": 1, "reused_unique": 0, '
        '"embedder_id": "fastembed:granite", "backend_id": "fastembed:granite", '
        '"device": "cpu", "seconds": 0.01}\n'
    )
    popen_calls = []

    def fake_popen(cmd, **_k):
        popen_calls.append(cmd)
        return _FakePopen([success_line], 0)

    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)

    started = server.start_index_job(str(proj), index_device="cpu")
    job_id = started["job_id"]
    status = {}
    for _ in range(200):
        status = server.get_status(job_id)
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.01)
    assert status["status"] == "done", status
    assert popen_calls, "the index job must dispatch to the subprocess"


def test_inprocess_cpu_fast_path_never_reuses_cached_query_provider(tmp_path, monkeypatch):
    """The optional ENGRAM_INPROCESS_CPU_MAX fast path (off by default) must
    build a fresh, uncached provider -- never the process-wide singleton
    shared with search -- so turning it on can never reintroduce the
    query-lock contention defect."""
    from engram_mcp import server
    from engram_mcp.embeddings import factory

    def fail_if_called():
        raise AssertionError("in-process fast path must not touch the cached query provider")

    monkeypatch.setattr(factory, "_fastembed_granite_cpu", fail_if_called)

    built = []

    class _FakeUncachedProvider:
        # Deliberately not CANONICAL_EMBEDDER_ID: the post-index query-model
        # warmup (_schedule_query_model_warmup) legitimately touches the real
        # cached provider for the *canonical* id, which is unrelated to what
        # this test checks (that the index job itself never does). A
        # non-canonical id makes warmup a fast no-op (unsupported embedder)
        # instead of racing a background thread against this test's mock.
        model_id = "test:fake-uncached"
        backend_id = model_id
        dim = 4
        device = "cpu"

        def embed_passages(self, texts):
            return [[0.0, 0.0, 0.0, 1.0] for _ in texts]

        def embed_queries(self, texts):
            return [[0.0, 0.0, 0.0, 1.0] for _ in texts]

        def release_unused_cache(self):
            pass

    def fake_make_uncached():
        provider = _FakeUncachedProvider()
        built.append(provider)
        return provider

    monkeypatch.setattr(factory, "make_uncached_cpu_provider", fake_make_uncached)

    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")

    job = server._registry.create(str(proj.resolve()))
    server._inprocess_cpu_index(job.job_id, str(proj.resolve()), True)

    status = server.get_status(job.job_id)
    assert status["status"] == "done", status
    assert len(built) == 1  # exactly one fresh, uncached provider was constructed


def test_index_project_tool_returns_before_planning_completes(tmp_path, monkeypatch):
    """The MCP index_project tool call must hand off to the background job
    and return immediately, not block on plan_index's filesystem walk (the
    second half of the CPU-indexing-blocks-search fix)."""
    from engram_mcp import server

    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")

    planning_may_return = threading.Event()

    def blocking_plan_index(*_a, **_k):
        assert planning_may_return.wait(timeout=5)
        return types.SimpleNamespace(
            mode="full", files=1, chunks=1, added=1, changed=0, deleted=0,
            unchanged=0, missing_unique_chunks=1,
        )

    monkeypatch.setattr(server, "_plan_index", blocking_plan_index)
    monkeypatch.setattr(server, "_subprocess_index", lambda *_a, **_k: None)

    async def _call() -> dict:
        return await server.index_project(str(proj), index_device="auto")

    t0 = time.time()
    started = asyncio.run(_call())
    elapsed = time.time() - t0

    assert "job_id" in started
    assert elapsed < 1.0  # plan_index blocks for up to 5s; the tool call must not wait on it

    planning_may_return.set()
    job_id = started["job_id"]
    for _ in range(200):
        status = server.get_status(job_id)
        if status["status"] in ("done", "error", "cancelled"):
            break
        time.sleep(0.01)


def test_second_index_project_call_coalesces_onto_running_job(tmp_path, monkeypatch):
    """Two concurrent index_project calls for the same resolved project path
    must not both run: the second returns the first job's id, marked
    coalesced, instead of starting a duplicate job."""
    from engram_mcp import server

    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")

    submissions = []

    class Pool:
        def submit(self, fn, *args):
            # Deliberately never run the job body, so it stays "queued" and
            # the second call below observes it as still active.
            submissions.append((fn, args))

    monkeypatch.setattr(server, "_index_pool", Pool())

    first = server.start_index_job(str(proj), index_device="cpu")
    second = server.start_index_job(str(proj), index_device="cpu")

    assert first["coalesced"] is False
    assert second["coalesced"] is True
    assert second["job_id"] == first["job_id"]
    assert len(submissions) == 1  # only one job was ever actually submitted


def test_cancel_index_moves_job_to_cancelled_and_leaves_previous_index_searchable(
    tmp_path, monkeypatch
):
    """cancel_index must: request a kill of the tracked subprocess, end the
    job in a terminal "cancelled" status (never "error"), and never disturb
    whatever index was already published before the cancelled job started --
    the atomic manifest/generation swap must simply never happen."""
    from engram_mcp import server

    proj, provider = _indexed_project(tmp_path)
    monkeypatch.setattr(server, "_provider_for_query_model", lambda _m: provider)
    before = server.do_search(str(proj), "add numbers", k=1)
    assert before["count"] == 1

    reached_embedding = threading.Event()
    allow_unblock = threading.Event()

    class BlockingStdout:
        def __iter__(self):
            yield (
                '{"event": "progress", "version": 1, "seq": 1, "stage": "embedding", '
                '"unit": "embeddings", "done": 1, "total": 99, "chunks": 99, '
                '"embedded": 1, "reused": 0}\n'
            )
            reached_embedding.set()
            allow_unblock.wait(timeout=5)
            return  # simulates the pipe closing when the child is killed

    class BlockingPopen:
        stdout = BlockingStdout()
        stderr = io.StringIO("")
        returncode = 1  # a killed child's exit code

        def wait(self):
            return self.returncode

    killed = []

    def fake_kill_tree(proc):
        killed.append(proc)
        allow_unblock.set()

    monkeypatch.setattr(server.gitmeta, "_kill_tree", fake_kill_tree)
    monkeypatch.setattr(server.subprocess, "Popen", lambda *a, **k: BlockingPopen())

    started = server.start_index_job(str(proj), index_device="cpu")
    job_id = started["job_id"]
    assert reached_embedding.wait(timeout=2)

    cancel_result = server.do_cancel_index(job_id)

    assert killed  # the tracked subprocess was targeted for a real kill
    assert cancel_result["status"] == "cancelled"
    assert cancel_result["already_terminal"] is False

    final = server.get_status(job_id)
    assert final["status"] == "cancelled"
    assert final["error"] is None

    # The previously published index is untouched and still searchable.
    after = server.do_search(str(proj), "add numbers", k=1)
    assert after["count"] == 1
    assert after["results"][0]["chunk_id"] == before["results"][0]["chunk_id"]


def test_cancel_index_already_terminal_job_is_a_noop(tmp_path):
    from engram_mcp import server

    job = server._registry.create(str(tmp_path))
    server._registry.update(job.job_id, status="done", stage="done", finished_at=time.time())

    result = server.do_cancel_index(job.job_id)
    assert result["status"] == "done"
    assert result["already_terminal"] is True


def test_cancel_index_unknown_job_id_is_structured_error():
    from engram_mcp import server

    result = server.do_cancel_index("no-such-job")
    assert result["code"] == errors.E_BAD_REQUEST


def test_cancel_index_kills_real_process_tree(tmp_path, monkeypatch):
    """End-to-end with a real child process and the real (not mocked)
    gitmeta._kill_tree: cancel_index must actually terminate the OS process,
    and _subprocess_index's own loop -- observing the pipe close -- must end
    the job in "cancelled", not "error"."""
    from engram_mcp import server

    real_popen = subprocess.Popen

    def spawn_sleeper(cmd, **kwargs):
        # Swap only the `engram_mcp.cli index` invocation for a plain
        # sleeping child (so this test needs neither a model nor a real
        # project index, while still exercising a genuine OS process/pipe/
        # kill). subprocess.Popen is a single process-wide name -- patching
        # it also redirects gitmeta._kill_tree's own `subprocess.run(taskkill
        # ...)` call (subprocess.run constructs a Popen internally), so
        # anything that isn't the index invocation must fall through to the
        # real Popen or the kill itself would be swallowed by this fake.
        if any("engram_mcp.cli" in str(part) for part in cmd):
            return real_popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
        return real_popen(cmd, **kwargs)

    monkeypatch.setattr(server.subprocess, "Popen", spawn_sleeper)

    proj = tmp_path / "proj"
    proj.mkdir()
    job = server._registry.create(str(proj.resolve()))
    thread = threading.Thread(
        target=server._subprocess_index,
        args=(job.job_id, str(proj.resolve()), False, "cpu"),
    )
    thread.start()
    try:
        proc = None
        deadline = time.time() + 5
        while time.time() < deadline:
            with server._job_processes_lock:
                proc = server._job_processes.get(job.job_id)
            if proc is not None:
                break
            time.sleep(0.02)
        assert proc is not None, "the subprocess should have registered by now"
        assert proc.poll() is None  # really running

        result = server.do_cancel_index(job.job_id)

        assert result["status"] == "cancelled"
        assert proc.poll() is not None  # the real OS process is actually gone
    finally:
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_chunk_id_stable_across_full_and_single_file_reindex_batches(tmp_path):
    """chunk_id must be a pure function of the chunk, not the batch ordinal.

    Regression for the bug where `_rows()` keyed chunk_id on the global
    enumeration index of the current indexing batch: b.py's chunk got a
    different id in a full build that also included a.py (nonzero global
    index) than when b.py alone was re-embedded via `reindex_file` (index 0
    in that isolated batch), even though b.py's own content never changed.
    """
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "a.py").write_text("def helper_a():\n    return 1\n", encoding="utf-8")
    (proj / "b.py").write_text("def helper_b():\n    return 2\n", encoding="utf-8")
    provider = FakeProvider()
    index_project(proj, provider, full_rebuild=True)

    qi = load_query_index(proj)
    before = sorted(r["chunk_id"] for r in qi.store.metadata_rows(where="rel_path = 'b.py'"))
    assert before  # b.py produced at least one chunk

    result = reindex_file(proj, provider, "b.py")
    assert result["action"] == "reindexed"

    qi_after = load_query_index(proj)
    after = sorted(r["chunk_id"] for r in qi_after.store.metadata_rows(where="rel_path = 'b.py'"))
    assert after == before


def test_search_hit_chunk_id_hydrates_via_get_chunk_after_unrelated_reindex(tmp_path, monkeypatch):
    """A chunk_id handed out by search_code must stay valid after an
    unrelated file is added and the project is reindexed -- otherwise a
    cached citation rots the moment anything else in the project changes.
    """
    from engram_mcp import server

    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "target.py").write_text(
        "def target_marker():\n    return 'target marker'\n", encoding="utf-8"
    )
    provider = FakeProvider()
    monkeypatch.setattr(server, "_provider_for_query_model", lambda _model_id: provider)
    index_project(proj, provider, full_rebuild=True)

    hit = server.do_search(str(proj), "target marker", k=1, mode="vector")["results"][0]
    chunk_id = hit["chunk_id"]
    fetched = server.do_get_chunk(str(proj), chunk_id)
    assert "error" not in fetched

    # An unrelated file that sorts before target.py joins the project and the
    # index is rebuilt; target.py's own chunk never changed.
    (proj / "aaa_new.py").write_text(
        "def new_marker():\n    return 'new marker'\n", encoding="utf-8"
    )
    index_project(proj, provider, full_rebuild=True)

    rehydrated = server.do_get_chunk(str(proj), chunk_id)
    assert "error" not in rehydrated
    assert rehydrated["content"] == fetched["content"]


def test_corrupt_files_manifest_forces_full_rebuild_not_silent_incremental(tmp_path):
    """A corrupt files.json must not be treated as "nothing was ever
    indexed": that silently drops deletion detection and leaves rows for
    files no longer on disk searchable. The index run must instead force a
    full rebuild, which naturally re-derives the correct file set.
    """
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "a.py").write_text("def keep():\n    return 1\n", encoding="utf-8")
    (proj / "deleted.py").write_text("def gone():\n    return 2\n", encoding="utf-8")
    provider = FakeProvider()
    index_project(proj, provider, full_rebuild=True)

    pdir = paths.project_dir(proj, create=False)
    (pdir / "files.json").write_text("{not json", encoding="utf-8")
    (proj / "deleted.py").unlink()

    stats = index_project(proj, provider)  # incremental requested; must not trust files.json
    assert stats.mode == "full"

    qi = load_query_index(proj)
    assert qi.store.metadata_rows(where="rel_path = 'deleted.py'") == []
    assert len(qi.store.metadata_rows(where="rel_path = 'a.py'")) == 1
