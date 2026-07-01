"""Embedder factory: profiles, id parsing, optional-extra gating (no model)."""

from __future__ import annotations

import importlib.util
import sys

import pytest

from engram_mcp.embeddings import factory


def test_profiles_are_the_unified_multilingual_set():
    assert set(factory.PROFILES) == {
        "local_cpu_small", "local_cpu_large", "local_gpu_small", "local_gpu_large"
    }


def test_cpu_profiles_are_no_torch_fastembed():
    # local_cpu_* run on the FastEmbed (ONNX) path, so they need no `gpu` extra.
    from engram_mcp.embeddings import fastembed_provider as fe

    for name in ("local_cpu_small", "local_cpu_large"):
        model_name, device = factory._FASTEMBED[name]
        assert model_name in fe._CUSTOM_ONNX
        assert device == "cpu"
    # all local — no cloud/api profile leaked in
    assert not any(
        tok in p for p in factory.PROFILES for tok in ("openai", "voyage", "cohere", "api")
    )


@pytest.mark.skipif(
    sys.platform != "win32" or sys.flags.utf8_mode,
    reason="the open() shim only applies on Windows without UTF-8 mode",
)
def test_utf8_text_open_defaults_encoding_to_utf8(tmp_path):
    from engram_mcp.embeddings.fastembed_provider import _utf8_text_open

    f = tmp_path / "cfg.json"
    f.write_text("тест", encoding="utf-8")  # non-ASCII → cp1252 would misread/raise
    with _utf8_text_open():
        with open(f) as fh:  # no encoding passed → shim injects utf-8
            assert fh.encoding.lower().replace("-", "") == "utf8"
            assert fh.read() == "тест"
    # binary + explicit-encoding opens are untouched, and open() is restored after
    assert open is __import__("builtins").open


def test_granite_custom_onnx_metadata():
    from engram_mcp.embeddings import fastembed_provider as fe

    assert fe._CUSTOM_ONNX["ibm-granite/granite-embedding-97m-multilingual-r2"] == {
        "pooling": "CLS", "dim": 384,
    }
    assert fe._CUSTOM_ONNX["ibm-granite/granite-embedding-311m-multilingual-r2"] == {
        "pooling": "CLS", "dim": 768,
    }


def test_ensure_custom_registered_is_idempotent():
    # Registration is metadata-only (no download) and must survive a repeat call.
    from fastembed import TextEmbedding

    from engram_mcp.embeddings import fastembed_provider as fe

    name = "ibm-granite/granite-embedding-97m-multilingual-r2"
    fe._ensure_custom_registered(name)
    fe._ensure_custom_registered(name)  # second call must not raise
    assert any(m["model"] == name for m in TextEmbedding.list_supported_models())


def test_granite_replay_pins_cpu_matching_index_device(monkeypatch):
    # The local_cpu_* profiles must build ONE shared provider for index + search:
    # replay has to use the same (model, device) lru key, and device must be cpu
    # (never the caller's "auto"), so it can't quietly land on CUDA.
    calls = []
    model = "ibm-granite/granite-embedding-97m-multilingual-r2"

    class _Fake:
        model_id = f"fastembed:{model}"
        dim = 384

    monkeypatch.setattr(factory, "_fastembed", lambda m, d: calls.append((m, d)) or _Fake())
    factory.make_provider("local_cpu_small")
    factory.provider_for_model_id(_Fake.model_id, device="auto")
    assert calls == [(model, "cpu"), (model, "cpu")]


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
        factory.make_provider("local_gpu_large")


@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is not None,
    reason="gpu extra installed; loading the reranker would be heavy",
)
def test_reranker_without_extra_raises_helpful_error():
    from engram_mcp.rerankers import get_reranker

    with pytest.raises(ImportError):
        get_reranker()


def test_make_provider_loads_and_reports_dim(provider):
    p = factory.make_provider("local_cpu_small")
    assert p.dim == 384
    assert p.model_id == "fastembed:ibm-granite/granite-embedding-97m-multilingual-r2"
    # device-independent cache key (CPU/GPU must not invalidate)
    assert "cpu" not in p.model_id and "cuda" not in p.model_id


def test_fastembed_release_unused_cache_is_noop(provider):
    # FastEmbed has no torch CUDA cache; release must be a safe no-op.
    factory.make_provider("local_cpu_small").release_unused_cache()


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
