"""Embedder factory: profiles, id parsing, optional-extra gating (no model)."""

from __future__ import annotations

import importlib.util

import pytest

from engram_mcp.embeddings import factory


def test_profiles_include_fastembed_and_qwen_all_local():
    assert {"local_fast", "local_quality"} <= set(factory.PROFILES)
    assert {"local_granite", "local_granite_quality"} <= set(factory.PROFILES)
    assert {"local_qwen_small", "local_qwen"} <= set(factory.PROFILES)


def test_granite_profiles_are_no_torch_fastembed():
    # Granite runs on the FastEmbed (ONNX) path, so it needs no `gpu` extra.
    from engram_mcp.embeddings import fastembed_provider as fe

    for name in ("local_granite", "local_granite_quality"):
        model_name, device = factory._FASTEMBED[name]
        assert model_name in fe._CUSTOM_ONNX
        assert device == "cpu"
    # all local — no cloud/api profile leaked in
    assert not any(
        tok in p for p in factory.PROFILES for tok in ("openai", "voyage", "cohere", "api")
    )


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        factory.make_provider("gpt4")


def test_unsupported_model_id_raises():
    with pytest.raises(ValueError):
        factory.provider_for_model_id("openai:text-embedding-3-small")


@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is not None,
    reason="gpu extra installed; loading the model would be heavy",
)
def test_qwen_profile_without_extra_raises_helpful_error():
    with pytest.raises(ImportError):
        factory.make_provider("local_qwen")


@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is not None,
    reason="gpu extra installed; loading the reranker would be heavy",
)
def test_reranker_without_extra_raises_helpful_error():
    from engram_mcp.rerankers import get_reranker

    with pytest.raises(ImportError):
        get_reranker()


def test_make_provider_loads_and_reports_dim(provider):
    p = factory.make_provider("local_fast")
    assert p.dim == 384
    assert p.model_id == "fastembed:BAAI/bge-small-en-v1.5"
    # device-independent cache key (CPU/GPU must not invalidate)
    assert "cpu" not in p.model_id and "cuda" not in p.model_id


def test_fastembed_release_unused_cache_is_noop(provider):
    # FastEmbed has no torch CUDA cache; release must be a safe no-op.
    factory.make_provider("local_fast").release_unused_cache()


def test_st_batch_size_from_env(monkeypatch):
    # Pure parsing — no model load (importing the module doesn't import torch/ST).
    from engram_mcp.embeddings import sentence_transformers_provider as st

    monkeypatch.delenv("ENGRAM_ST_BATCH_SIZE", raising=False)
    assert st._env_batch_size() == st._DEFAULT_ST_BATCH
    monkeypatch.setenv("ENGRAM_ST_BATCH_SIZE", "8")
    assert st._env_batch_size() == 8
    for bad in ("0", "-4", "abc", ""):
        monkeypatch.setenv("ENGRAM_ST_BATCH_SIZE", bad)
        assert st._env_batch_size() == st._DEFAULT_ST_BATCH
