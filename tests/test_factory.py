"""Embedder factory: one canonical Granite id and explicit index devices."""

from __future__ import annotations

import importlib.util
import sys
import types

import pytest

from engram_mcp import errors
from engram_mcp.embeddings import factory


class _FakeFastEmbed:
    model_name = factory.GRANITE_MODEL
    model_id = factory.CANONICAL_EMBEDDER_ID
    backend_id = factory.CANONICAL_EMBEDDER_ID
    device = "cpu"
    dim = 384

    def release_unused_cache(self) -> None:
        pass


def _clear_loaded() -> None:
    with factory._LOADED_LOCK:
        factory._LOADED_BACKEND_IDS.clear()


def test_single_supported_embedder_constants():
    assert factory.GRANITE_MODEL == "ibm-granite/granite-embedding-97m-multilingual-r2"
    assert factory.CANONICAL_EMBEDDER_ID == f"fastembed:{factory.GRANITE_MODEL}"
    assert factory.SUPPORTED_INDEX_DEVICES == ("auto", "cpu", "cuda")


@pytest.mark.skipif(
    sys.platform != "win32" or sys.flags.utf8_mode,
    reason="the open() shim only applies on Windows without UTF-8 mode",
)
def test_utf8_text_open_defaults_encoding_to_utf8(tmp_path):
    from engram_mcp.embeddings.fastembed_provider import _utf8_text_open

    f = tmp_path / "cfg.json"
    f.write_text("тест", encoding="utf-8")  # non-ASCII -> cp1252 would misread/raise
    with _utf8_text_open():
        with open(f) as fh:  # no encoding passed -> shim injects utf-8
            assert fh.encoding.lower().replace("-", "") == "utf8"
            assert fh.read() == "тест"
    assert open is __import__("builtins").open


def test_granite_custom_onnx_metadata():
    from engram_mcp.embeddings import fastembed_provider as fe

    assert fe._CUSTOM_ONNX[factory.GRANITE_MODEL] == {"pooling": "CLS", "dim": 384}
    assert "ibm-granite/granite-embedding-311m-multilingual-r2" not in fe._CUSTOM_ONNX


def test_ensure_custom_registered_is_idempotent():
    # Registration is metadata-only (no download) and must survive a repeat call.
    from fastembed import TextEmbedding

    from engram_mcp.embeddings import fastembed_provider as fe

    fe._ensure_custom_registered(factory.GRANITE_MODEL)
    fe._ensure_custom_registered(factory.GRANITE_MODEL)  # second call must not raise
    assert any(m["model"] == factory.GRANITE_MODEL for m in TextEmbedding.list_supported_models())


def test_provider_for_model_id_allows_only_canonical_cpu_fastembed(monkeypatch):
    _clear_loaded()
    calls = []

    def fake_cpu():
        calls.append("cpu")
        return _FakeFastEmbed()

    monkeypatch.setattr(factory, "_fastembed_granite_cpu", fake_cpu)
    p = factory.provider_for_model_id(factory.CANONICAL_EMBEDDER_ID)

    assert calls == ["cpu"]
    assert p.model_name == factory.GRANITE_MODEL
    assert p.model_id == factory.CANONICAL_EMBEDDER_ID
    assert p.backend_id == factory.CANONICAL_EMBEDDER_ID
    assert p.device == "cpu"
    assert factory.loaded_model_ids() == [factory.CANONICAL_EMBEDDER_ID]


def test_removed_embedder_ids_raise_without_loading(monkeypatch):
    monkeypatch.setattr(
        factory,
        "_fastembed_granite_cpu",
        lambda: (_ for _ in ()).throw(AssertionError("must not load fastembed")),
    )
    monkeypatch.setattr(
        factory,
        "_sentence_transformers_granite_cuda",
        lambda: (_ for _ in ()).throw(AssertionError("must not load st")),
    )

    removed = [
        "st:Qwen/Qwen3-Embedding-4B@1024#query/none",
        "fastembed:ibm-granite/granite-embedding-311m-multilingual-r2",
    ]
    for model_id in removed:
        with pytest.raises(errors.EngramError) as exc:
            factory.provider_for_model_id(model_id)
        assert exc.value.code == errors.E_UNKNOWN_PROFILE
        assert "removed/unsupported embedder" in (exc.value.hint or "")


def test_unsupported_model_id_raises_unknown_profile():
    with pytest.raises(errors.EngramError) as exc:
        factory.provider_for_model_id("openai:text-embedding-3-small")
    assert exc.value.code == errors.E_UNKNOWN_PROFILE


@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is not None,
    reason="gpu extra installed; loading the model would be heavy",
)
def test_cuda_index_provider_without_extra_raises_helpful_error():
    with pytest.raises(errors.EngramError) as exc:
        factory.make_index_provider("cuda")
    assert exc.value.code == errors.E_EXTRA_MISSING


def test_make_index_provider_cpu_uses_fastembed(monkeypatch):
    monkeypatch.setattr(factory, "_fastembed_granite_cpu", lambda: _FakeFastEmbed())
    p = factory.make_index_provider("cpu")
    assert p.model_id == factory.CANONICAL_EMBEDDER_ID
    assert p.backend_id == factory.CANONICAL_EMBEDDER_ID
    assert p.device == "cpu"


def test_resolve_index_device_default_is_auto_gpu_priority(monkeypatch):
    monkeypatch.delenv("ENGRAM_INDEX_DEVICE", raising=False)
    assert factory.resolve_index_device() == "auto"          # GPU-priority default
    monkeypatch.setenv("ENGRAM_INDEX_DEVICE", "cuda")
    assert factory.resolve_index_device() == "cuda"
    monkeypatch.setenv("ENGRAM_INDEX_DEVICE", "cpu")
    assert factory.resolve_index_device(gpu=True) == "cuda"   # flag beats env
    assert factory.resolve_index_device(cpu=True) == "cpu"
    with pytest.raises(errors.EngramError) as exc:
        factory.resolve_index_device(gpu=True, cpu=True)
    assert exc.value.code == errors.E_BAD_REQUEST
    monkeypatch.setenv("ENGRAM_INDEX_DEVICE", "bogus")
    with pytest.raises(errors.EngramError) as exc:
        factory.resolve_index_device()
    assert exc.value.code == errors.E_BAD_REQUEST


def test_effective_index_device_prefers_gpu_falls_back_to_cpu(monkeypatch):
    # "auto" resolves to cuda when a GPU is present, else cpu. Explicit passes through.
    monkeypatch.setattr(factory, "_cuda_available", lambda: True)
    assert factory.effective_index_device("auto") == "cuda"
    monkeypatch.setattr(factory, "_cuda_available", lambda: False)
    assert factory.effective_index_device("auto") == "cpu"
    assert factory.effective_index_device("cuda") == "cuda"
    assert factory.effective_index_device("cpu") == "cpu"


def test_canonical_st_provider_identity_and_loaded_tracking(monkeypatch):
    from engram_mcp.embeddings.sentence_transformers_provider import (
        SentenceTransformersProvider,
    )

    class _FakeVecs:
        def __init__(self, n: int):
            self.n = n

        def tolist(self):
            return [[1.0] * 384 for _ in range(self.n)]

    class _FakeSentenceTransformer:
        def __init__(self, model_name, device, trust_remote_code, truncate_dim):
            self.model_name = model_name
            self.device = device

        def encode(self, texts, **_kwargs):
            return _FakeVecs(len(texts))

    st_mod = types.ModuleType("sentence_transformers")
    st_mod.SentenceTransformer = _FakeSentenceTransformer
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "sentence_transformers", st_mod)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    _clear_loaded()
    p = SentenceTransformersProvider(
        factory.GRANITE_MODEL,
        device="cuda",
        canonical_id=factory.CANONICAL_EMBEDDER_ID,
        strict_device=True,
    )
    assert p.model_id == factory.CANONICAL_EMBEDDER_ID
    assert p.backend_id == f"st:{factory.GRANITE_MODEL}@full#none/none:cuda"
    assert p.backend_id != p.model_id

    factory._remember_loaded(p)
    assert factory.is_model_loaded(p.backend_id)
    assert not factory.is_model_loaded(p.model_id)


@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is not None,
    reason="gpu extra installed; loading the reranker would be heavy",
)
def test_reranker_without_extra_raises_helpful_error():
    from engram_mcp.rerankers import get_reranker

    with pytest.raises(ImportError):
        get_reranker()


def test_make_index_provider_loads_and_reports_dim(provider):
    p = factory.make_index_provider("cpu")
    assert p.dim == 384
    assert p.model_id == factory.CANONICAL_EMBEDDER_ID
    assert p.backend_id == factory.CANONICAL_EMBEDDER_ID
    assert "cpu" not in p.model_id and "cuda" not in p.model_id


def test_fastembed_release_unused_cache_is_noop(provider):
    factory.make_index_provider("cpu").release_unused_cache()


def test_st_batch_size_from_env(monkeypatch):
    # Pure parsing; importing the module doesn't import torch/ST.
    from engram_mcp.embeddings import sentence_transformers_provider as st

    monkeypatch.delenv("ENGRAM_ST_BATCH_SIZE", raising=False)
    assert st._env_batch_size() == st._DEFAULT_ST_BATCH
    monkeypatch.setenv("ENGRAM_ST_BATCH_SIZE", "8")
    assert st._env_batch_size() == 8
    for bad in ("0", "-4", "abc", ""):
        monkeypatch.setenv("ENGRAM_ST_BATCH_SIZE", bad)
        assert st._env_batch_size() == st._DEFAULT_ST_BATCH
