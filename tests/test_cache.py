"""Tests for the embedding cache + hash keys (no model needed)."""

from __future__ import annotations

import pytest

from engram_mcp.embeddings.cache import EmbeddingCache
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
