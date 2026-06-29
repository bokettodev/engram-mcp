"""TLS configuration for model downloads behind corporate MITM proxies.

Model weights are fetched over HTTPS (huggingface_hub / fastembed -> requests +
httpx), which verify against ``certifi``. On a corporate network with a
TLS-inspecting proxy the proxy's root CA is trusted by the OS but absent from
certifi, so verification fails with ``CERTIFICATE_VERIFY_FAILED``. ``curl`` and
the browser work because they trust the OS store; Python doesn't, by default.

``configure_tls()`` resolves this, in priority order:

  1. ``ENGRAM_INSECURE_DOWNLOADS=1`` -> disable verification entirely (loud
     warning). Last resort for a trusted network where nothing else works.
  2. a CA bundle env var set (``SSL_CERT_FILE`` / ``REQUESTS_CA_BUNDLE`` /
     ``CURL_CA_BUNDLE``) -> mirror it across all three so every client honors
     the same bundle (the matrix of which client reads which var is uneven).
  3. otherwise (default) -> ``truststore.inject_into_ssl()``: Python verifies
     against the OS trust store, where the corporate CA already lives — exactly
     like curl / the OS. Disable with ``ENGRAM_SYSTEM_TRUST=0``.

Call once at process start (CLI / server ``main``), BEFORE the embedder is
imported, so the patch is in place before any ``SSLContext`` is created. The
global ``ssl`` patch is inherited by the background index worker thread.
"""

from __future__ import annotations

import contextlib
import logging
import os

logger = logging.getLogger(__name__)

_CA_ENV_VARS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")

CERT_HINT = (
    "TLS certificate verification failed while downloading the model.\n"
    "If you are behind a corporate TLS-inspecting proxy:\n"
    "  - Engram uses your OS trust store by default; make sure the proxy's root "
    "CA is installed there (curl/the browser working is a good sign it is).\n"
    "  - or point SSL_CERT_FILE at your corporate CA bundle (.pem).\n"
    "  - last resort: set ENGRAM_INSECURE_DOWNLOADS=1 to skip verification "
    "(trusted networks only)."
)


def _truthy(val: str | None) -> bool:
    return (val or "").strip().lower() in ("1", "true", "yes", "on")


def _falsy(val: str | None) -> bool:
    return (val or "").strip().lower() in ("0", "false", "no", "off")


def configure_tls() -> str:
    """Set up TLS trust for model downloads. Returns the mode applied (for logs/tests)."""
    if _truthy(os.environ.get("ENGRAM_INSECURE_DOWNLOADS")):
        _disable_verification()
        return "insecure"

    bundle = next((os.environ[v] for v in _CA_ENV_VARS if os.environ.get(v)), None)
    if bundle:
        for v in _CA_ENV_VARS:
            os.environ.setdefault(v, bundle)
        logger.debug("using CA bundle from env for downloads: %s", bundle)
        return "ca-bundle"

    if _falsy(os.environ.get("ENGRAM_SYSTEM_TRUST")):
        return "default"
    try:
        import truststore

        truststore.inject_into_ssl()
        return "system-trust"
    except Exception as exc:  # pragma: no cover - truststore is a hard dependency
        logger.warning(
            "could not enable the system trust store (%r); falling back to certifi "
            "defaults. Behind a corporate proxy, set SSL_CERT_FILE to your CA bundle.",
            exc,
        )
        return "default"


def _disable_verification() -> None:
    import ssl
    import warnings

    warnings.warn(
        "ENGRAM_INSECURE_DOWNLOADS is set: TLS certificate verification is DISABLED "
        "for model downloads. Use only on a trusted network.",
        stacklevel=2,
    )
    ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[attr-defined]
    # requests (used by huggingface_hub / fastembed) ignores the stdlib default
    # context, so force verify=False on every session and silence the noise.
    try:
        from functools import partialmethod

        import requests
        import urllib3

        requests.Session.request = partialmethod(requests.Session.request, verify=False)  # type: ignore[assignment]
        urllib3.disable_warnings()
    except Exception:  # pragma: no cover - requests always present via hf_hub
        pass


def is_cert_error(exc: BaseException) -> bool:
    """True if ``exc`` (or its cause chain) is a TLS certificate-verification failure."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if "certificate verify failed" in repr(cur).lower():
            return True
        cur = cur.__cause__ or cur.__context__
    return False


@contextlib.contextmanager
def guard_download(model_name: str = ""):
    """Wrap a model load; re-raise TLS cert failures with an actionable hint."""
    try:
        yield
    except Exception as exc:
        if is_cert_error(exc):
            where = f" for {model_name}" if model_name else ""
            raise RuntimeError(f"{CERT_HINT}\n  (downloading{where}; original: {exc})") from exc
        raise
