"""Tests for the embedding cache + hash keys (no model needed)."""

from __future__ import annotations

import pytest

from engram_mcp.embeddings import cache as cache_module
from engram_mcp.embeddings.cache import (
    EmbeddingCache,
    cache_max_bytes,
    cache_max_mb,
    global_cache_path,
    global_cache_report,
    read_only_stats,
)
from engram_mcp.indexing.hash import embedding_input_hash, sha256_text


def test_cache_round_trip(tmp_path):
    cache = EmbeddingCache(tmp_path / "emb.sqlite")
    cache.put_many({"h1": [0.1, 0.2, 0.3], "h2": [1.0, -1.0, 0.5]})
    got = cache.get_many(["h1", "h2", "missing"])
    assert "missing" not in got
    # stored as float32, so compare with tolerance
    assert got["h1"] == pytest.approx([0.1, 0.2, 0.3], rel=1e-6)
    assert got["h2"] == pytest.approx([1.0, -1.0, 0.5], rel=1e-6)
    cache.close()


def test_cache_persists_across_connections(tmp_path):
    db = tmp_path / "emb.sqlite"
    c1 = EmbeddingCache(db)
    c1.put_many({"x": [0.5, 0.5]})
    c1.close()
    c2 = EmbeddingCache(db)
    assert c2.get_many(["x"])["x"] == [0.5, 0.5]
    c2.close()


def test_cache_get_many_can_filter_by_stored_dim(tmp_path):
    cache = EmbeddingCache(tmp_path / "emb.sqlite")
    cache.put_many({"x": [0.5, 0.5]})
    assert cache.get_many(["x"], dim=2)["x"] == [0.5, 0.5]
    assert cache.get_many(["x"], dim=3) == {}
    cache.close()


def test_embedding_input_hash_changes_with_model_and_chunker():
    base = embedding_input_hash("m1", "1", "def f(): pass")
    assert base != embedding_input_hash("m2", "1", "def f(): pass")  # model swap
    assert base != embedding_input_hash("m1", "2", "def f(): pass")  # chunker bump
    assert base != embedding_input_hash("m1", "1", "def g(): pass")  # text change
    assert base == embedding_input_hash("m1", "1", "def f(): pass")  # deterministic


def test_sha256_text_deterministic():
    assert sha256_text("abc") == sha256_text("abc")
    assert sha256_text("abc") != sha256_text("abd")


def test_cache_max_mb_unset_or_non_positive_means_unlimited(monkeypatch):
    monkeypatch.delenv("ENGRAM_CACHE_MAX_MB", raising=False)
    assert cache_max_mb() is None
    assert cache_max_bytes() is None

    for raw in ("0", "-5", "not-a-number", "  "):
        monkeypatch.setenv("ENGRAM_CACHE_MAX_MB", raw)
        assert cache_max_mb() is None

    monkeypatch.setenv("ENGRAM_CACHE_MAX_MB", "256")
    assert cache_max_mb() == 256
    assert cache_max_bytes() == 256 * 1024 * 1024


def test_cache_stats_reports_rows_and_bytes(tmp_path):
    cache = EmbeddingCache(tmp_path / "emb.sqlite")
    cache.put_many({"h1": [0.1, 0.2, 0.3], "h2": [1.0, -1.0, 0.5]})
    stats = cache.stats()
    assert stats["rows"] == 2
    assert stats["bytes"] > 0
    cache.close()


def test_cache_prune_to_budget_unlimited_is_a_noop(tmp_path):
    cache = EmbeddingCache(tmp_path / "emb.sqlite")
    cache.put_many({"h1": [0.1, 0.2, 0.3]})
    before = cache.stats()

    result = cache.prune_to_budget(None)

    assert result["pruned_rows"] == 0
    assert result["vacuumed"] is False
    assert cache.stats() == before
    cache.close()


def test_cache_prune_to_budget_evicts_least_recently_used(tmp_path, monkeypatch):
    db = tmp_path / "emb.sqlite"
    cache = EmbeddingCache(db)
    vec = [0.1] * 64  # large enough that row count materially affects file size

    # 20 rows, strictly increasing last_used_at set purely by insertion order
    # (no separate "touch" step -- an UPDATE between the "before" measurement
    # and the prune call would grow the WAL and skew the byte accounting).
    fake_now = [1000.0]
    monkeypatch.setattr(cache_module.time, "time", lambda: fake_now[0])
    n = 20
    for i in range(n):
        fake_now[0] = 1000.0 + i
        cache.put_many({f"h{i}": vec})

    before = cache.stats()
    assert before["rows"] == n

    # Target roughly half the rows. SQLite page/WAL overhead means the exact
    # boundary row is not guaranteed, so assert the LRU *ordering* invariant
    # with a fuzz margin around the midpoint rather than an exact row index:
    # the oldest quarter must be evicted, the newest quarter must survive.
    avg_bytes_per_row = before["bytes"] // before["rows"]
    budget = avg_bytes_per_row * (n // 2)
    result = cache.prune_to_budget(budget)

    assert result["vacuumed"] is True
    assert 0 < result["pruned_rows"] < n
    after = cache.stats()
    assert 0 < after["rows"] < before["rows"]
    assert after["bytes"] < before["bytes"]  # VACUUM actually reclaimed bytes

    remaining = cache.get_many([f"h{i}" for i in range(n)])
    oldest_quarter = {f"h{i}" for i in range(n // 4)}
    newest_quarter = {f"h{i}" for i in range(n - n // 4, n)}
    assert oldest_quarter.isdisjoint(remaining)  # least-recently-used: evicted
    assert newest_quarter <= set(remaining)  # most-recently-used: survives
    cache.close()


def test_cache_migrates_legacy_schema_without_last_used_at(tmp_path):
    import sqlite3

    db = tmp_path / "emb.sqlite"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE embeddings (h TEXT PRIMARY KEY, dim INTEGER, vec BLOB)")
    conn.execute(
        "INSERT INTO embeddings (h, dim, vec) VALUES (?, ?, ?)", ("legacy", 1, b"\x00\x00\x80?")
    )
    conn.commit()
    conn.close()

    cache = EmbeddingCache(db)  # must not raise on the pre-existing schema
    assert cache.get_many(["legacy"])["legacy"] == pytest.approx([1.0])
    stats = cache.stats()
    assert stats["rows"] == 1
    cache.close()


def test_read_only_stats_never_writes_to_disk(tmp_path):
    db = tmp_path / "emb.sqlite"
    cache = EmbeddingCache(db)
    cache.put_many({"h1": [0.1, 0.2]})
    cache.close()

    before_bytes = db.read_bytes()
    stats = read_only_stats(db)
    assert stats["exists"] is True
    assert stats["rows"] == 1
    assert stats["bytes"] > 0
    assert db.read_bytes() == before_bytes  # byte-for-byte unchanged


def test_read_only_stats_missing_file_reports_zero(tmp_path):
    stats = read_only_stats(tmp_path / "missing.sqlite")
    assert stats == {"path": str(tmp_path / "missing.sqlite"), "exists": False, "rows": None, "bytes": 0}


def test_global_cache_report_missing_cache_reports_zero_without_creating_it():
    # `isolated_engram_home` (autouse) already points ENGRAM_HOME at a fresh
    # tmp dir per test, so the cache genuinely does not exist yet here.
    report = global_cache_report(dry_run=True)
    assert report["exists"] is False
    assert report["rows"] == 0
    assert report["bytes"] == 0
    assert not global_cache_path().parent.exists()


def test_global_cache_report_dry_run_never_mutates_file():
    db_path = global_cache_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cache = EmbeddingCache(db_path)
    cache.put_many({"h1": [0.1, 0.2, 0.3]})
    cache.close()
    before_bytes = db_path.read_bytes()

    report = global_cache_report(dry_run=True, max_bytes=1)  # budget irrelevant to a dry run

    assert report["exists"] is True
    assert report["dry_run"] is True
    assert report["rows"] == 1
    assert db_path.read_bytes() == before_bytes


def test_global_cache_report_prunes_and_reports_before_after():
    db_path = global_cache_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cache = EmbeddingCache(db_path)
    vec = [0.1] * 64
    # Enough rows that the data genuinely dominates the fixed SQLite
    # schema/page overhead -- a handful of tiny rows can fit inside that
    # fixed floor, making a shrink-after-VACUUM assertion meaningless.
    for i in range(500):
        cache.put_many({f"h{i}": vec})
    before_rows = cache.stats()["rows"]
    cache.close()

    report = global_cache_report(dry_run=False, max_bytes=1)

    assert report["dry_run"] is False
    assert report["vacuumed"] is True
    assert report["before_rows"] == before_rows
    assert report["before_bytes"] > 0
    assert report["after_bytes"] < report["before_bytes"]
    assert report["after_rows"] < report["before_rows"]


def test_cache_pruning_is_never_referenced_from_search_or_index_hot_paths():
    """`prune_to_budget`/`global_cache_report` must only be reachable from an
    explicit operator action (`engram gc`) or the startup task -- never from
    `pipeline.py` (indexing + search + doctor_project) or `server.py` (the
    MCP tool surface, including the search-time query path)."""
    import inspect

    from engram_mcp import pipeline, server

    for mod in (pipeline, server):
        src = inspect.getsource(mod)
        assert "prune_to_budget" not in src
        assert "global_cache_report" not in src


def test_global_cache_report_refuses_to_prune_under_readonly(monkeypatch):
    db_path = global_cache_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cache = EmbeddingCache(db_path)
    cache.put_many({"h1": [0.1, 0.2, 0.3]})
    before = cache.stats()
    cache.close()

    monkeypatch.setenv("ENGRAM_READONLY", "1")
    report = global_cache_report(dry_run=False, max_bytes=1)

    assert report["read_only"] is True
    assert report["dry_run"] is True
    assert read_only_stats(db_path)["rows"] == before["rows"]
