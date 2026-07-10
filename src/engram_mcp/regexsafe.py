"""Validation and bounded execution for caller supplied regular expressions."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field

from engram_mcp import gitmeta

MAX_USER_REGEX_CHARS = 500
DEFAULT_USER_REGEX_TIMEOUT_SEC = 1.0


def _regex_timeout_seconds() -> float:
    raw = os.environ.get("ENGRAM_USER_REGEX_TIMEOUT_SEC", "").strip()
    if not raw:
        return DEFAULT_USER_REGEX_TIMEOUT_SEC
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_USER_REGEX_TIMEOUT_SEC
    return max(0.05, min(value, 30.0))


def _run_worker(
    op: str,
    *,
    pattern: str,
    flags: int,
    texts: list[str],
    timeout_sec: float,
) -> tuple[bool, object, str]:
    """Run a regex job in a bare interpreter with private stdio."""

    if os.environ.get("ENGRAM_USER_REGEX_SUBPROCESS", "").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False, None, "regex subprocess disabled by ENGRAM_USER_REGEX_SUBPROCESS=0"
    job = json.dumps(
        {"op": op, "pattern": pattern, "flags": flags, "texts": texts},
        ensure_ascii=False,
    ).encode("utf-8")
    creation = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if sys.platform == "win32"
        else {"start_new_session": True}
    )
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "engram_mcp._regex_child"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            **creation,
        )
    except Exception as exc:
        return False, None, str(exc) or repr(exc)
    try:
        stdout, _ = proc.communicate(input=job, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        gitmeta.kill_process_tree(proc)
        try:
            proc.communicate(timeout=1)
        except Exception:
            gitmeta.kill_process_tree(proc)
        return False, None, f"regex execution timed out after {timeout_sec:.2f}s"
    except Exception as exc:
        gitmeta.kill_process_tree(proc)
        return False, None, str(exc) or repr(exc)
    try:
        result = json.loads(stdout)
        if proc.returncode != 0:
            return False, None, f"regex child exited with status {proc.returncode}"
        if result.get("status") == "ok":
            return True, result.get("payload"), ""
        return False, None, str(result.get("error") or "regex child failed")
    except Exception as exc:
        return False, None, str(exc) or repr(exc)


def _validated_pattern(
    pattern: str | None,
    default: str,
    *,
    flags: int,
    label: str,
) -> tuple[str, list[str]]:
    candidate = str(pattern or default)
    if len(candidate) > MAX_USER_REGEX_CHARS:
        return default, [f"{label} is too long; using default regex"]
    try:
        re.compile(candidate, flags)
    except re.error as exc:
        return default, [f"invalid {label}: {exc}; using default regex"]
    return candidate, []


def _search_many_in_process(pattern: str, flags: int, texts: list[str]) -> list[bool]:
    rx = re.compile(pattern, flags)
    return [bool(rx.search(text)) for text in texts]


def _extract_first_in_process(pattern: str, flags: int, texts: list[str]) -> list[str]:
    rx = re.compile(pattern, flags)
    out: list[str] = []
    for text in texts:
        match = rx.search(text)
        if not match:
            out.append("")
        elif "ticket" in match.groupdict():
            out.append(match.group("ticket"))
        elif match.groups():
            out.append(match.group(1))
        else:
            out.append(match.group(0))
    return out


def _unsafe_warning(label: str, reason: str) -> str:
    detail = f": {reason}" if reason else ""
    return f"{label} is unsafe or too slow{detail}; using default regex"


def pattern_or_default(
    pattern: str | None,
    default: str,
    *,
    flags: int = 0,
    label: str = "regex",
) -> tuple[str, list[str]]:
    """Return a syntactically valid regex pattern, plus degradation warnings."""

    return _validated_pattern(pattern, default, flags=flags, label=label)


def compile_or_default(
    pattern: str | None,
    default: str,
    *,
    flags: int = 0,
    label: str = "regex",
) -> tuple[re.Pattern[str], str, list[str]]:
    safe_pattern, warnings = pattern_or_default(
        pattern,
        default,
        flags=flags,
        label=label,
    )
    return re.compile(safe_pattern, flags), safe_pattern, warnings


def search_many_or_default(
    pattern: str | None,
    default: str,
    texts: list[str],
    *,
    flags: int = 0,
    label: str = "regex",
    timeout_sec: float | None = None,
) -> tuple[str, list[bool], list[str]]:
    """Search actual texts with a caller regex in a bounded worker.

    The default regex is trusted and runs in-process. Non-default regexes are
    compiled in-process for syntax/length checks, then executed over the real
    input in a subprocess so catastrophic backtracking cannot wedge the caller.
    """

    text_list = [str(text or "") for text in texts]
    candidate, warnings = _validated_pattern(pattern, default, flags=flags, label=label)
    if candidate == default:
        return candidate, _search_many_in_process(candidate, flags, text_list), warnings
    ok, payload, reason = _run_worker(
        "search",
        pattern=candidate,
        flags=flags,
        texts=text_list,
        timeout_sec=_regex_timeout_seconds() if timeout_sec is None else timeout_sec,
    )
    if ok and isinstance(payload, list):
        return candidate, [bool(item) for item in payload], []
    fallback = _search_many_in_process(default, flags, text_list)
    return default, fallback, [_unsafe_warning(label, reason)]


def extract_first_or_default(
    pattern: str | None,
    default: str,
    texts: list[str],
    *,
    flags: int = 0,
    label: str = "regex",
    timeout_sec: float | None = None,
) -> tuple[str, list[str], list[str]]:
    """Extract first regex matches from actual texts with timeout isolation."""

    text_list = [str(text or "") for text in texts]
    candidate, warnings = _validated_pattern(pattern, default, flags=flags, label=label)
    if candidate == default:
        return candidate, _extract_first_in_process(candidate, flags, text_list), warnings
    ok, payload, reason = _run_worker(
        "extract",
        pattern=candidate,
        flags=flags,
        texts=text_list,
        timeout_sec=_regex_timeout_seconds() if timeout_sec is None else timeout_sec,
    )
    if ok and isinstance(payload, list):
        return candidate, [str(item or "") for item in payload], []
    fallback = _extract_first_in_process(default, flags, text_list)
    return default, fallback, [_unsafe_warning(label, reason)]


@dataclass(slots=True)
class RegexRequestCache:
    """Per-request memoization for bounded regex passes over identical corpora."""

    _search: dict[tuple[str, str, int, tuple[str, ...]], tuple[str, tuple[bool, ...], tuple[str, ...]]] = field(
        default_factory=dict
    )
    _extract: dict[tuple[str, str, int, tuple[str, ...]], tuple[str, tuple[str, ...], tuple[str, ...]]] = field(
        default_factory=dict
    )

    @staticmethod
    def _key(pattern: str | None, default: str, flags: int, texts: list[str]) -> tuple[str, str, int, tuple[str, ...]]:
        return str(pattern or default), str(default), int(flags), tuple(str(text or "") for text in texts)

    def search_many_or_default(
        self,
        pattern: str | None,
        default: str,
        texts: list[str],
        *,
        flags: int = 0,
        label: str = "regex",
        timeout_sec: float | None = None,
    ) -> tuple[str, list[bool], list[str]]:
        key = self._key(pattern, default, flags, texts)
        cached = self._search.get(key)
        if cached is None:
            resolved, hits, warnings = search_many_or_default(
                pattern,
                default,
                texts,
                flags=flags,
                label=label,
                timeout_sec=timeout_sec,
            )
            cached = (resolved, tuple(bool(hit) for hit in hits), tuple(str(w) for w in warnings if str(w)))
            self._search[key] = cached
        resolved, hits, warnings = cached
        return resolved, list(hits), list(warnings)

    def extract_first_or_default(
        self,
        pattern: str | None,
        default: str,
        texts: list[str],
        *,
        flags: int = 0,
        label: str = "regex",
        timeout_sec: float | None = None,
    ) -> tuple[str, list[str], list[str]]:
        key = self._key(pattern, default, flags, texts)
        cached = self._extract.get(key)
        if cached is None:
            resolved, extracted, warnings = extract_first_or_default(
                pattern,
                default,
                texts,
                flags=flags,
                label=label,
                timeout_sec=timeout_sec,
            )
            cached = (resolved, tuple(str(item or "") for item in extracted), tuple(str(w) for w in warnings if str(w)))
            self._extract[key] = cached
        resolved, extracted, warnings = cached
        return resolved, list(extracted), list(warnings)
