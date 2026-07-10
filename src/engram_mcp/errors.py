"""Structured error codes returned by the MCP read tools."""

from __future__ import annotations

from dataclasses import dataclass

E_PROJECT_NOT_INDEXED = "E_PROJECT_NOT_INDEXED"
E_REF_NOT_INDEXED = "E_REF_NOT_INDEXED"
E_INDEX_INVALID = "E_INDEX_INVALID"
E_UNKNOWN_PROFILE = "E_UNKNOWN_PROFILE"
E_EXTRA_MISSING = "E_EXTRA_MISSING"
E_MODEL_LOAD_FAILED = "E_MODEL_LOAD_FAILED"
E_MODEL_LOADING = "E_MODEL_LOADING"
E_BAD_REQUEST = "E_BAD_REQUEST"


@dataclass(slots=True)
class EngramError(ValueError):
    """An exception that already has the user-facing MCP error shape."""

    message: str
    code: str
    hint: str | None = None

    def __str__(self) -> str:
        return self.message


def error_result(
    message: str,
    code: str,
    *,
    hint: str | None = None,
    results: list | None = None,
    **extra,
) -> dict:
    """Return the common MCP error payload shape."""

    out = {"error": message, "code": code}
    if hint:
        out["hint"] = hint
    if results is not None:
        out["results"] = results
    out.update(extra)
    return out
