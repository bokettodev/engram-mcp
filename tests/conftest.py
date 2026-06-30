"""Shared fixtures. The `provider` fixture loads the real FastEmbed model
once per session; tests using it are skipped when the model is unavailable or
ENGRAM_SKIP_MODEL=1.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def isolated_engram_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path / "home"))
    from engram_mcp import paths

    paths._reset_data_home_for_tests()
    yield
    paths._reset_data_home_for_tests()


@pytest.fixture(scope="session")
def provider():
    if os.environ.get("ENGRAM_SKIP_MODEL") == "1":
        pytest.skip("model-dependent test disabled")
    try:
        from engram_mcp.embeddings.fastembed_provider import FastEmbedProvider

        return FastEmbedProvider()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"embedder unavailable: {exc}")
