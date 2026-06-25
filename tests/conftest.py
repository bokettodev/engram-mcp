"""Shared fixtures. The `provider` fixture loads the real FastEmbed model
once per session; tests using it are skipped when the model is unavailable or
ENGRAM_SKIP_MODEL=1.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session")
def provider():
    if os.environ.get("ENGRAM_SKIP_MODEL") == "1":
        pytest.skip("model-dependent test disabled")
    try:
        from engram_mcp.embeddings.fastembed_provider import FastEmbedProvider

        return FastEmbedProvider()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"embedder unavailable: {exc}")
