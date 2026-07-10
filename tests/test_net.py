"""TLS download configuration: env precedence + cert-error classification.

All network-free — the truststore/insecure side effects are monkeypatched out so
these never touch the global ssl module or requests.
"""

from __future__ import annotations

import os

import pytest

from engram_mcp import net

_ALL_ENV = (
    "ENGRAM_INSECURE_DOWNLOADS", "ENGRAM_SYSTEM_TRUST",
    "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
)

# Snapshot taken at import time, before any test in this module has run, so
# the final regression test (bottom of file) has a trustworthy "before" to
# compare against.
_ORIGINAL_ENV_SNAPSHOT = {v: os.environ.get(v) for v in _ALL_ENV}


@pytest.fixture
def clean_env(monkeypatch):
    # `configure_tls()` doesn't only setdefault through monkeypatch: its
    # CA-bundle mirroring path (`os.environ[v] = bundle`) writes DIRECTLY to
    # os.environ for vars monkeypatch never touched (e.g. a var that was
    # absent before the test and that this test itself never called
    # monkeypatch.setenv/delenv on). monkeypatch only undoes changes it made
    # itself, so those direct writes have no registered undo action and leak
    # into the rest of the test session -- previously observed as a leaked
    # SSL_CERT_FILE breaking an unrelated real-subprocess test later in the
    # suite. Snapshot + restore the real environment ourselves so this is
    # correct regardless of how configure_tls() chooses to mutate os.environ.
    saved = {v: os.environ.get(v) for v in _ALL_ENV}
    for v in _ALL_ENV:
        monkeypatch.delenv(v, raising=False)
    try:
        yield monkeypatch
    finally:
        for v, value in saved.items():
            if value is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = value


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


def test_existing_nonempty_bundle_vars_are_not_overwritten(clean_env):
    clean_env.setenv("SSL_CERT_FILE", "/a.pem")
    clean_env.setenv("REQUESTS_CA_BUNDLE", "/b.pem")
    assert net.configure_tls() == "ca-bundle"
    import os

    # a deliberately-different non-empty var is the user's call, left as-is
    assert os.environ["SSL_CERT_FILE"] == "/a.pem"
    assert os.environ["REQUESTS_CA_BUNDLE"] == "/b.pem"
    assert os.environ["CURL_CA_BUNDLE"] in ("/a.pem", "/b.pem")


def test_empty_bundle_var_is_overwritten_not_left_blank(clean_env):
    # The footgun: an empty SSL_CERT_FILE makes httpx fall back to certifi.
    clean_env.setenv("SSL_CERT_FILE", "")
    clean_env.setenv("REQUESTS_CA_BUNDLE", "/corp/ca.pem")
    assert net.configure_tls() == "ca-bundle"
    import os

    assert os.environ["SSL_CERT_FILE"] == "/corp/ca.pem"
    assert os.environ["CURL_CA_BUNDLE"] == "/corp/ca.pem"


def test_system_trust_disabled_falls_back_to_default(clean_env):
    clean_env.setenv("ENGRAM_SYSTEM_TRUST", "0")
    assert net.configure_tls() == "default"


def test_system_trust_is_default(clean_env, monkeypatch):
    injected = []
    fake = type("M", (), {"inject_into_ssl": staticmethod(lambda: injected.append(1))})
    monkeypatch.setitem(__import__("sys").modules, "truststore", fake)
    assert net.configure_tls() == "system-trust"
    assert injected == [1]


def test_is_cert_error_matches_cause_chain():
    inner = OSError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:1010)")
    outer = ConnectionError("connect failed")
    outer.__cause__ = inner
    assert net.is_cert_error(outer)
    assert net.is_cert_error(inner)


def test_is_cert_error_matches_via_context_only():
    # A re-raise can attach the real error to __context__ (implicit) rather than
    # __cause__ (explicit `from`); both must be walked.
    inner = OSError("certificate verify failed: self-signed certificate in chain")
    outer = RuntimeError("wrapper")
    outer.__context__ = inner  # no __cause__
    assert net.is_cert_error(outer)


def test_is_cert_error_ignores_unrelated():
    assert not net.is_cert_error(ValueError("nope"))
    assert not net.is_cert_error(ConnectionError("timed out"))


def test_disable_verification_patches_requests_and_httpx():
    import httpx
    import requests
    import ssl

    saved = (
        ssl._create_default_https_context,
        requests.Session.request,
        httpx.Client.__init__,
        httpx.AsyncClient.__init__,
    )
    try:
        net._disable_verification()
        # httpx is the huggingface_hub 1.x path — must be covered, not just requests
        assert getattr(httpx.Client.__init__, "_engram_insecure", False)
        assert getattr(httpx.AsyncClient.__init__, "_engram_insecure", False)
        assert ssl._create_default_https_context is ssl._create_unverified_context
        # idempotent: a second call must not double-wrap
        net._disable_verification()
        assert getattr(httpx.Client.__init__, "_engram_insecure", False)
    finally:
        (
            ssl._create_default_https_context,
            requests.Session.request,
            httpx.Client.__init__,
            httpx.AsyncClient.__init__,
        ) = saved


def test_guard_download_wraps_cert_error_with_hint():
    with pytest.raises(RuntimeError, match="certificate verification failed"):
        with net.guard_download("ibm-granite/granite-embedding-97m-multilingual-r2"):
            raise OSError("certificate verify failed: self-signed certificate in chain")


def test_guard_download_passes_through_other_errors():
    with pytest.raises(ValueError):
        with net.guard_download():
            raise ValueError("unrelated")


# Must stay the LAST test defined in this module: it compares the ambient
# process environment against the snapshot taken at import time (before any
# test here ran), so it only proves something if every other TLS test in the
# file has already executed. No ordering plugin (pytest-randomly/xdist) is
# configured for this project, so in-file declaration order is reliable.
def test_ambient_env_unchanged_after_tls_tests() -> None:
    current = {v: os.environ.get(v) for v in _ALL_ENV}
    assert current == _ORIGINAL_ENV_SNAPSHOT, (
        "a TLS test leaked one of these vars into the real process "
        "environment instead of restoring it on teardown"
    )
