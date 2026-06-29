"""TLS download configuration: env precedence + cert-error classification.

All network-free — the truststore/insecure side effects are monkeypatched out so
these never touch the global ssl module or requests.
"""

from __future__ import annotations

import pytest

from engram_mcp import net

_ALL_ENV = (
    "ENGRAM_INSECURE_DOWNLOADS", "ENGRAM_SYSTEM_TRUST",
    "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
)


@pytest.fixture
def clean_env(monkeypatch):
    # Record + clear every var so configure_tls()'s setdefault writes are torn
    # down by monkeypatch (it removes keys that were originally absent).
    for v in _ALL_ENV:
        monkeypatch.delenv(v, raising=False)
    return monkeypatch


def test_insecure_takes_precedence(clean_env, monkeypatch):
    calls = []
    monkeypatch.setattr(net, "_disable_verification", lambda: calls.append(1))
    clean_env.setenv("ENGRAM_INSECURE_DOWNLOADS", "1")
    clean_env.setenv("REQUESTS_CA_BUNDLE", "/some/ca.pem")  # ignored under insecure
    assert net.configure_tls() == "insecure"
    assert calls == [1]


def test_ca_bundle_is_mirrored_across_clients(clean_env):
    clean_env.setenv("REQUESTS_CA_BUNDLE", "/corp/ca.pem")
    assert net.configure_tls() == "ca-bundle"
    import os

    assert os.environ["SSL_CERT_FILE"] == "/corp/ca.pem"
    assert os.environ["CURL_CA_BUNDLE"] == "/corp/ca.pem"


def test_existing_bundle_vars_are_not_overwritten(clean_env):
    clean_env.setenv("SSL_CERT_FILE", "/a.pem")
    clean_env.setenv("REQUESTS_CA_BUNDLE", "/b.pem")
    assert net.configure_tls() == "ca-bundle"
    import os

    # setdefault must not clobber a var the user set explicitly
    assert os.environ["SSL_CERT_FILE"] == "/a.pem"
    assert os.environ["REQUESTS_CA_BUNDLE"] == "/b.pem"
    assert os.environ["CURL_CA_BUNDLE"] in ("/a.pem", "/b.pem")


def test_system_trust_disabled_falls_back_to_default(clean_env):
    clean_env.setenv("ENGRAM_SYSTEM_TRUST", "0")
    assert net.configure_tls() == "default"


def test_system_trust_is_default(clean_env, monkeypatch):
    injected = []
    fake = type("M", (), {"inject_into_ssl": staticmethod(lambda: injected.append(1))})
    monkeypatch.setitem(__import__("sys").modules, "truststore", fake)
    assert net.configure_tls() == "system-trust"
    assert injected == [1]


def test_is_cert_error_matches_chain():
    inner = OSError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:1010)")
    outer = ConnectionError("connect failed")
    outer.__cause__ = inner
    assert net.is_cert_error(outer)
    assert net.is_cert_error(inner)


def test_is_cert_error_ignores_unrelated():
    assert not net.is_cert_error(ValueError("nope"))
    assert not net.is_cert_error(ConnectionError("timed out"))


def test_guard_download_wraps_cert_error_with_hint():
    with pytest.raises(RuntimeError, match="certificate verification failed"):
        with net.guard_download("Qwen/Qwen3-Embedding-4B"):
            raise OSError("certificate verify failed: self-signed certificate in chain")


def test_guard_download_passes_through_other_errors():
    with pytest.raises(ValueError):
        with net.guard_download():
            raise ValueError("unrelated")
