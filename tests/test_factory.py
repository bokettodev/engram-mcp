"""Embedder factory: profiles, id parsing, optional-extra gating (no model)."""

from __future__ import annotations

import importlib.util

import pytest

from engram_mcp.embeddings import factory


def test_profiles_include_fastembed_and_qwen_all_local():
    assert {"local_fast", "local_quality"} <= set(factory.PROFILES)
    assert {"local_qwen_small", "local_qwen"} <= set(factory.PROFILES)
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
