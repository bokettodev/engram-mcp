"""Regression tests for model-identity pinning and fingerprinting.

Engram's canonical embedder id used to be a bare Hugging Face repo name
(``fastembed:ibm-granite/granite-embedding-97m-multilingual-r2``). That is
mutable upstream: the org can push new weights/tokenizer/ONNX artifacts under
the SAME repo name at any time, and a bare-name id can't detect it -- an old
index and old embedding-cache rows would silently keep matching a new,
numerically different model. See config.py (pinned revisions) and
embeddings/hf_pin.py (fingerprint construction, snapshot pinning, artifact
digest) for the fix.
"""

from __future__ import annotations

import importlib.util
import math
import os

import pytest

from engram_mcp import config, errors, manifest, paths
from engram_mcp.embeddings import factory, hf_pin
from engram_mcp.embeddings.cache import EmbeddingCache
from engram_mcp.indexing.hash import embedding_input_hash
from engram_mcp.pipeline import _is_compatible, doctor_project, index_project, search_project


class CanonicalFakeProvider:
    """A fake embedding provider that reports the REAL canonical id, so
    manifests it produces are checked against the real compatibility gates
    (factory.provider_for_model_id, _is_compatible, search_project) instead of
    an arbitrary test-only id."""

    model_id = factory.CANONICAL_EMBEDDER_ID
    backend_id = model_id
    dim = 4
    artifact_digest = ""

    def _vec(self, text: str) -> list[float]:
        low = text.lower()
        return [
            1.0 if "add" in low or "sum" in low else 0.0,
            1.0 if "cache" in low else 0.0,
            1.0 if "config" in low else 0.0,
            1.0,
        ]

    def embed_passages(self, texts):
        return [self._vec(t) for t in texts]

    def embed_queries(self, texts):
        return [self._vec(t) for t in texts]

    def release_unused_cache(self) -> None:
        pass


def _write_project(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "math_utils.py").write_text(
        "def add_numbers(a, b):\n    '''Return the sum.'''\n    return a + b\n",
        encoding="utf-8",
    )
    return proj


# ---------------------------------------------------------------------------
# Fingerprint construction
# ---------------------------------------------------------------------------


def test_canonical_model_id_embeds_the_pinned_revision():
    # The whole point of pinning: the revision must actually change the id,
    # not just be accepted as an unused parameter.
    assert config.EMBED_MODEL_REVISION in factory.CANONICAL_EMBEDDER_ID
    other = hf_pin.canonical_model_id(
        backend="fastembed",
        repo_id=factory.GRANITE_MODEL,
        revision="0" * 40,
        dim=config.DEFAULT_EMBED_DIM,
        pooling="cls",
    )
    assert other != factory.CANONICAL_EMBEDDER_ID


def test_canonical_model_id_changes_with_each_component():
    base = dict(backend="fastembed", repo_id="org/repo", revision="a" * 40, dim=384, pooling="cls")
    reference = hf_pin.canonical_model_id(**base)
    variants = [
        {**base, "revision": "b" * 40},
        {**base, "dim": 768},
        {**base, "pooling": "mean"},
        {**base, "backend": "st"},
        {**base, "normalize": False},
    ]
    for variant in variants:
        assert hf_pin.canonical_model_id(**variant) != reference


# ---------------------------------------------------------------------------
# Global embedding-cache key: a row written under one fingerprint must never
# be served for another (revision, or artifact digest).
# ---------------------------------------------------------------------------


def test_embedding_cache_misses_across_different_revisions(tmp_path):
    text = "def add(a, b):\n    return a + b"
    id_a = hf_pin.canonical_model_id(
        backend="fastembed", repo_id="org/repo", revision="a" * 40, dim=384, pooling="cls"
    )
    id_b = hf_pin.canonical_model_id(
        backend="fastembed", repo_id="org/repo", revision="b" * 40, dim=384, pooling="cls"
    )
    h_a = embedding_input_hash(id_a, config.CHUNKER_VERSION, text)
    h_b = embedding_input_hash(id_b, config.CHUNKER_VERSION, text)
    assert h_a != h_b

    with EmbeddingCache(tmp_path / "cache.sqlite") as cache:
        cache.put_many({h_a: [1.0, 2.0, 3.0]})
        # Real cache lookups, not a string comparison: a row written under
        # fingerprint A's hash is simply absent under fingerprint B's hash.
        assert cache.get_many([h_b]) == {}
        assert cache.get_many([h_a]) == {h_a: [1.0, 2.0, 3.0]}


def test_embedding_cache_misses_across_different_artifact_digests(tmp_path):
    text = "def add(a, b):\n    return a + b"
    model_id = factory.CANONICAL_EMBEDDER_ID
    h_a = embedding_input_hash(model_id, config.CHUNKER_VERSION, text, artifact_digest="blob-a")
    h_b = embedding_input_hash(model_id, config.CHUNKER_VERSION, text, artifact_digest="blob-b")
    assert h_a != h_b

    with EmbeddingCache(tmp_path / "cache.sqlite") as cache:
        cache.put_many({h_a: [9.0]})
        assert cache.get_many([h_b]) == {}
        assert cache.get_many([h_a]) == {h_a: [9.0]}


# ---------------------------------------------------------------------------
# An index recorded under a different fingerprint fails loud -- it must not
# be served, and the failure must carry a specific, checkable error code.
# ---------------------------------------------------------------------------


def test_stale_revision_embedder_id_is_rejected_with_unknown_profile(monkeypatch):
    """The primary, model-load-free gate every real query goes through
    (factory.provider_for_model_id, used by server.do_search /
    server.do_reindex_file / the CLI) before a model is even loaded."""
    stale_id = hf_pin.canonical_model_id(
        backend="fastembed",
        repo_id=factory.GRANITE_MODEL,
        revision="0" * 40,  # a plausible-looking but wrong/old revision
        dim=config.DEFAULT_EMBED_DIM,
        pooling="cls",
    )
    assert stale_id != factory.CANONICAL_EMBEDDER_ID
    monkeypatch.setattr(
        factory,
        "_fastembed_granite_cpu",
        lambda: (_ for _ in ()).throw(AssertionError("must not load the model to reject a stale id")),
    )
    with pytest.raises(errors.EngramError) as exc:
        factory.provider_for_model_id(stale_id)
    assert exc.value.code == errors.E_UNKNOWN_PROFILE


def test_search_project_rejects_index_built_with_different_embedder_id(tmp_path):
    proj = _write_project(tmp_path)
    provider = CanonicalFakeProvider()
    index_project(proj, provider, full_rebuild=True)

    class DifferentRevisionProvider(CanonicalFakeProvider):
        model_id = hf_pin.canonical_model_id(
            backend="fastembed",
            repo_id=factory.GRANITE_MODEL,
            revision="1" * 40,
            dim=config.DEFAULT_EMBED_DIM,
            pooling="cls",
        )
        backend_id = model_id

    with pytest.raises(errors.EngramError) as exc:
        search_project(proj, DifferentRevisionProvider(), "add numbers")
    assert exc.value.code == errors.E_INDEX_INVALID


def test_search_project_rejects_artifact_digest_mismatch_under_same_revision(tmp_path):
    """Same canonical id (same revision/dim/pooling) but a different recorded
    artifact digest must still fail loud -- this is the layer that catches an
    artifact swap under an *unchanged* revision, which the id string alone
    can't see (see config.py's threat model)."""

    class DigestAProvider(CanonicalFakeProvider):
        artifact_digest = "blob-a"

    class DigestBProvider(CanonicalFakeProvider):
        artifact_digest = "blob-b"

    proj = _write_project(tmp_path)
    index_project(proj, DigestAProvider(), full_rebuild=True)

    pdir = paths.project_dir(proj, create=False)
    m = manifest.load_project(pdir)
    assert m.embedder_artifact_digest == "blob-a"  # was actually recorded

    with pytest.raises(errors.EngramError) as exc:
        search_project(proj, DigestBProvider(), "add numbers")
    assert exc.value.code == errors.E_INDEX_INVALID
    assert "digest" in str(exc.value).lower()

    # Querying with the SAME digest the index was built under still works.
    hits = search_project(proj, DigestAProvider(), "add numbers")
    assert hits


def test_incremental_index_forces_rebuild_on_artifact_digest_mismatch(tmp_path):
    """`_is_compatible` (the switch between an incremental update and a full
    rebuild) must also see a digest mismatch, not just the id string --
    otherwise an artifact swap under an unchanged revision would get
    incrementally *merged into* an index built from the old artifact."""

    class DigestAProvider(CanonicalFakeProvider):
        artifact_digest = "blob-a"

    class DigestBProvider(CanonicalFakeProvider):
        artifact_digest = "blob-b"

    proj = _write_project(tmp_path)
    index_project(proj, DigestAProvider(), full_rebuild=True)
    pdir = paths.project_dir(proj, create=False)
    m = manifest.load_project(pdir)

    assert _is_compatible(m, DigestAProvider()) is True
    assert _is_compatible(m, DigestBProvider()) is False


def test_doctor_project_uses_the_live_canonical_id_not_a_stale_literal(tmp_path):
    """doctor_project's model_drift check must compare against the CURRENT
    canonical id (a computed fingerprint that changes with config.py's pinned
    revision), not a hardcoded string literal that would silently stop
    matching real up-to-date indexes the moment the revision is bumped."""
    proj = _write_project(tmp_path)
    index_project(proj, CanonicalFakeProvider(), full_rebuild=True)

    health = doctor_project(proj, check_git=False)
    assert not any(issue["code"] == "model_drift" for issue in health["issues"])


def test_manifest_compat_treats_missing_digest_as_unknown_not_mismatch():
    """A provider/manifest with no digest on one side (e.g. a GPU-built index
    -- see sentence_transformers_provider.py's module docstring) must not be
    treated as incompatible on that basis alone; only a real, present-on-both
    disagreement is a mismatch."""

    class NoDigestProvider(CanonicalFakeProvider):
        artifact_digest = ""

    m = manifest.ProjectManifest(
        project_id="p",
        root_path="/tmp/p",
        logical_project_id="p",
        checkout_kind="non_git",
        active_table="chunks",
        embedder_id=factory.CANONICAL_EMBEDDER_ID,
        embedder_artifact_digest="",  # never recorded, e.g. GPU-built
        dim=CanonicalFakeProvider.dim,
        chunker_version=config.CHUNKER_VERSION,
        chunk_id_scheme=config.CHUNK_ID_SCHEME,
        schema_version=manifest.SCHEMA_VERSION,
    )
    assert _is_compatible(m, NoDigestProvider()) is True


# ---------------------------------------------------------------------------
# Canonical id vs backend id: preserved after adding the revision/dim/pooling
# fingerprint (see embeddings/factory.py's docstring on the split).
# ---------------------------------------------------------------------------


def test_cuda_built_index_does_not_mark_cpu_search_model_loaded():
    from engram_mcp.embeddings.sentence_transformers_provider import SentenceTransformersProvider

    class _FakeVecs:
        def __init__(self, n):
            self.n = n

        def tolist(self):
            return [[1.0] * config.DEFAULT_EMBED_DIM for _ in range(self.n)]

    class _FakeSentenceTransformer:
        def __init__(self, model_name, device, truncate_dim, revision=None):
            pass

        def encode(self, texts, **_kwargs):
            return _FakeVecs(len(texts))

    import sys
    import types

    st_mod = types.ModuleType("sentence_transformers")
    st_mod.SentenceTransformer = _FakeSentenceTransformer
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None)

    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    try:
        mp.setitem(sys.modules, "sentence_transformers", st_mod)
        mp.setitem(sys.modules, "torch", torch_mod)
        with factory._LOADED_LOCK:
            factory._LOADED_BACKEND_IDS.clear()

        provider = SentenceTransformersProvider(
            factory.GRANITE_MODEL,
            device="cuda",
            canonical_id=factory.CANONICAL_EMBEDDER_ID,
            strict_device=True,
            revision=config.EMBED_MODEL_REVISION,
        )
        # The GPU-built index still records the canonical (CPU/FastEmbed) id
        # so it can be *queried* by the CPU backend -- but the runtime
        # bookkeeping key ("what's actually loaded right now") must stay
        # distinct, so a CUDA load is never mistaken for the CPU model being
        # warm.
        assert provider.model_id == factory.CANONICAL_EMBEDDER_ID
        assert provider.backend_id != provider.model_id
        assert config.EMBED_MODEL_REVISION in provider.backend_id

        factory._remember_loaded(provider)
        assert factory.is_model_loaded(provider.backend_id) is True
        assert factory.is_model_loaded(factory.CANONICAL_EMBEDDER_ID) is False
    finally:
        mp.undo()
        with factory._LOADED_LOCK:
            factory._LOADED_BACKEND_IDS.clear()


# ---------------------------------------------------------------------------
# ONNX (FastEmbed) vs sentence-transformers vector-space parity. Model- and
# gpu-extra-dependent; skips cleanly rather than failing when either is
# unavailable, per the torch-free serving-process invariant.
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


@pytest.mark.skipif(os.environ.get("ENGRAM_SKIP_MODEL") == "1", reason="model-dependent test disabled")
@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None
    or importlib.util.find_spec("torch") is None,
    reason="gpu extra (sentence-transformers/torch) not installed",
)
def test_onnx_and_sentence_transformers_agree_on_vectors():
    """The README claims the two backends produce the same 384-dim CLS-pooled
    vectors (cosine ~0.99997). This actually exercises both loaders and
    checks it, instead of leaving it an unverified claim."""
    from engram_mcp.embeddings.fastembed_provider import FastEmbedProvider
    from engram_mcp.embeddings.sentence_transformers_provider import SentenceTransformersProvider

    texts = [
        "def add(a, b):\n    return a + b",
        "Как найти определение функции в проекте?",
        "class EmbeddingCache:\n    def get_many(self):\n        return {}",
    ]

    onnx = FastEmbedProvider(config.DEFAULT_EMBED_MODEL, device="cpu")
    st = SentenceTransformersProvider(
        config.DEFAULT_EMBED_MODEL, device="cpu", revision=config.EMBED_MODEL_REVISION
    )
    try:
        assert onnx.dim == st.dim == config.DEFAULT_EMBED_DIM

        onnx_vecs = onnx.embed_passages(texts)
        st_vecs = st.embed_passages(texts)
        for a, b in zip(onnx_vecs, st_vecs):
            assert _cosine(a, b) > 0.999

        onnx_qvecs = onnx.embed_queries(texts)
        st_qvecs = st.embed_queries(texts)
        for a, b in zip(onnx_qvecs, st_qvecs):
            assert _cosine(a, b) > 0.999
    finally:
        st.unload()


# ---------------------------------------------------------------------------
# Real-model CI lane smoke tests. The fast/default CI lane sets
# ENGRAM_SKIP_MODEL=1 and never runs these (they `pytest.skip`, per
# conftest.py); the real-model lane in ci.yml deliberately does NOT set that
# var, so this is the only place the actual torch-free ONNX embedder and ONNX
# cross-encoder reranker are exercised end to end in CI.
# ---------------------------------------------------------------------------


def test_onnx_embedder_smoke(provider):
    """Embed a couple of strings with the real FastEmbed/ONNX embedder and
    check the output is well-formed: right shape/dim, every value finite, and
    (per the model's `normalization=True` registration -- see
    embeddings/fastembed_provider.py) unit-norm."""
    texts = ["def add(a, b):\n    return a + b", "как найти определение функции"]

    passages = provider.embed_passages(texts)
    queries = provider.embed_queries(texts)

    for vecs in (passages, queries):
        assert len(vecs) == len(texts)
        for vec in vecs:
            assert len(vec) == provider.dim == config.DEFAULT_EMBED_DIM
            assert all(math.isfinite(x) for x in vec)
            norm = math.sqrt(sum(x * x for x in vec))
            assert abs(norm - 1.0) < 1e-3, f"expected unit-norm vector, got norm={norm}"


@pytest.mark.skipif(os.environ.get("ENGRAM_SKIP_MODEL") == "1", reason="model-dependent test disabled")
def test_onnx_reranker_smoke():
    """Load the real default (torch-free ONNX) cross-encoder reranker and
    rerank a couple of candidates -- proves the pinned
    jinaai/jina-reranker-v2-base-multilingual snapshot actually loads and
    scores, not just that the (faked-out) selection logic is correct."""
    from engram_mcp.rerankers import get_reranker

    get_reranker.cache_clear()
    try:
        try:
            reranker = get_reranker(backend="fastembed")
        except Exception as exc:  # pragma: no cover - environment dependent
            pytest.skip(f"reranker unavailable: {exc}")
        assert reranker.backend == "fastembed"
        candidates = [
            {"content": "def subtract(a, b):\n    return a - b"},
            {"content": "def add(a, b):\n    return a + b"},
        ]
        ranked = reranker.rerank("function that adds two numbers", candidates, top_k=2)
        assert len(ranked) == 2
        assert all(math.isfinite(r["score"]) for r in ranked)
        assert ranked[0]["content"].startswith("def add")
        assert ranked[0]["reranked"] is True
    finally:
        get_reranker.cache_clear()
