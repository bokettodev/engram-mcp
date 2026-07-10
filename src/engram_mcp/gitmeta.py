"""Read-only git metadata helpers.

The metadata is diagnostic only. Engram records commit/ref/dirty state so tools
can report staleness, but it never uses churn or recency as a relevance prior.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

from engram_mcp import paths, regexsafe

# Env that makes git non-interactive and non-blocking: never prompt for
# credentials, never wait on/ take .git/index.lock (git status), never talk to
# a long-running fsmonitor daemon. These are the real reasons `git status` /
# `rev-parse` hang on Windows — and when git spawns such a grandchild it also
# inherits the stdout pipe, which defeats subprocess timeout (the post-kill
# read blocks forever). Prevent the hang at the source, then hard-kill the tree.
_GIT_ENV = {
    **os.environ,
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_PAGER": "cat",
    "GIT_ASKPASS": "",
    "GIT_CONFIG_NOSYSTEM": "1",
}
GIT_STALENESS_TIMEOUT_SEC = 3.0
# Compatibility alias for tests and external callers using the former name.
_GIT_STALENESS_TIMEOUT_SEC = GIT_STALENESS_TIMEOUT_SEC
_GIT_INDEX_TIMEOUT_SEC = 120.0
_LOG_RECORD_SEP = "\x1e"
_LOG_FIELD_SEP = "\x1f"
DEFAULT_FIX_REGEX = r"(?i)\b(fix(e[sd])?|bug|hotfix|patch|close[sd]?\s+#\d+)\b"
_HUNK_RE = re.compile(r"@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? \+\d+(?:,\d+)? @@")
_REGEX_DIAGNOSTIC_CHARS = 120


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


def requested_fix_regex_value(fix_regex: str | None) -> str:
    """Return the operator-requested git fix regex identity."""

    return str(fix_regex) if fix_regex is not None else DEFAULT_FIX_REGEX


def _clip_regex_for_warning(pattern: str) -> str:
    if len(pattern) <= _REGEX_DIAGNOSTIC_CHARS:
        return pattern
    return pattern[:_REGEX_DIAGNOSTIC_CHARS] + "..."


def _fix_regex_warnings(
    requested: str | None,
    effective: str,
    warnings: Iterable[str],
) -> list[str]:
    out = [str(w) for w in warnings if str(w)]
    requested_text = requested_fix_regex_value(requested)
    if requested_text != effective:
        diagnostic = (
            f"requested git_fix_regex {_clip_regex_for_warning(requested_text)!r} "
            f"resolved to {effective!r}; using effective regex"
        )
        if diagnostic not in out:
            out.insert(0, diagnostic)
    return out[:50]


def git_index_timeout_seconds() -> float:
    """Return the generous git timeout used only for index/background analytics."""

    raw = os.environ.get("ENGRAM_GIT_INDEX_TIMEOUT", "").strip()
    if not raw:
        return _GIT_INDEX_TIMEOUT_SEC
    try:
        value = float(raw)
    except ValueError:
        return _GIT_INDEX_TIMEOUT_SEC
    return max(1.0, min(value, 3600.0))


def _szz_worker_config() -> tuple[int, str | None]:
    default = min(8, max(2, (os.cpu_count() or 2) - 1))
    raw = os.environ.get("ENGRAM_SZZ_WORKERS", "").strip()
    if not raw:
        return default, None
    try:
        value = int(raw)
    except ValueError:
        return default, f"invalid ENGRAM_SZZ_WORKERS={raw!r}; using {default}"
    if value < 1:
        return default, f"invalid ENGRAM_SZZ_WORKERS={raw!r}; using {default}"
    return value, None


def szz_worker_count() -> int:
    """Return the adaptive SZZ blame worker count for this process."""

    return _szz_worker_config()[0]


def _staleness_disabled() -> bool:
    return os.environ.get("ENGRAM_GIT_STALENESS", "").strip().lower() in {"0", "false", "no", "off"}


def git_analytics_default() -> bool:
    """Default for whether index/project_map compute git-history analytics.

    Precedence (enforced by callers, not here): explicit argument >
    ``ENGRAM_GIT_ANALYTICS`` > this built-in default (enabled). Parsed the
    same way as ``ENGRAM_GIT_STALENESS``/``ENGRAM_RERANK_ENABLED``: only
    ``0``/``false``/``no``/``off`` (case-insensitive) disables; unset or any
    other value keeps analytics on.
    """
    return os.environ.get("ENGRAM_GIT_ANALYTICS", "").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
            )
        else:  # pragma: no cover - posix
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.wait(timeout=3)
    except Exception:
        pass


def kill_process_tree(proc: subprocess.Popen) -> None:
    """Terminate a subprocess and its descendants without propagating errors."""

    _kill_tree(proc)


def _git(root: Path, *args: str, timeout_sec: float = _GIT_STALENESS_TIMEOUT_SEC) -> str | None:
    """Run a read-only git command, guaranteed to return (never hang).

    Diagnostic only: any failure/timeout returns None, and the whole process
    tree is killed so a stuck git child can't wedge a search or doctor call.
    """
    if _staleness_disabled():
        return None
    cmd = [
        "git",
        "-C",
        str(root),
        "-c",
        "core.fsmonitor=",
        "-c",
        "gc.auto=0",
        "-c",
        "core.quotePath=false",
        *args,
    ]
    creation = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if sys.platform == "win32"
        else {"start_new_session": True}
    )
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_GIT_ENV,
            **creation,
        )
    except OSError:
        return None
    try:
        out, _ = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        return None
    except (OSError, ValueError, UnicodeError):
        _kill_tree(proc)
        return None
    if proc.returncode != 0:
        return None
    return out.strip()


def _git_dir(
    git_dir: Path,
    *args: str,
    timeout_sec: float = _GIT_STALENESS_TIMEOUT_SEC,
) -> str | None:
    """Run a read-only git command directly against a git common dir."""

    if _staleness_disabled():
        return None
    cmd = [
        "git",
        f"--git-dir={git_dir}",
        "-c",
        "core.fsmonitor=",
        "-c",
        "gc.auto=0",
        "-c",
        "core.quotePath=false",
        *args,
    ]
    creation = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if sys.platform == "win32"
        else {"start_new_session": True}
    )
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_GIT_ENV,
            **creation,
        )
    except OSError:
        return None
    try:
        out, _ = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        return None
    except (OSError, ValueError, UnicodeError):
        _kill_tree(proc)
        return None
    if proc.returncode != 0:
        return None
    return out.strip()


def _normalize_git_path(value: str | None) -> str:
    text = (value or "").replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text.strip("/")


def _canonical_git_path(value: str | None, *, base: Path) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = base / candidate
    try:
        return candidate.resolve().as_posix()
    except OSError:
        return ""


def _path_key(value: str) -> str:
    return os.path.normcase(value.replace("\\", "/"))


def _parse_status_v2_branch(text: str | None) -> tuple[str, str, bool]:
    ref = ""
    commit = ""
    dirty = False
    for line in (text or "").splitlines():
        if line.startswith("# branch.oid "):
            oid = line.removeprefix("# branch.oid ").strip()
            if oid != "(initial)":
                commit = oid
        elif line.startswith("# branch.head "):
            head = line.removeprefix("# branch.head ").strip()
            if head and head != "(detached)":
                ref = head
        elif line and not line.startswith("#"):
            dirty = True
    if not ref and commit:
        ref = commit[:12]
    return ref, commit, dirty


def common_dir_for_worktree(root: str | Path) -> str:
    """Return the resolved git common dir for a worktree, or ``""``."""

    try:
        path = Path(root).expanduser().resolve()
    except OSError:
        return ""
    common = _git(path, "rev-parse", "--git-common-dir")
    if not common:
        return ""
    return _canonical_git_path(common, base=path)


def _fingerprint_from_refs(common_dir: Path) -> dict:
    refs = _git_dir(common_dir, "for-each-ref", "--format=%(refname)%00%(objectname)", "refs")
    if refs is None:
        return {"status": "unavailable", "warning": "git refs unavailable"}
    lines = sorted(line.strip() for line in refs.splitlines() if line.strip())
    if not lines:
        head = _git_dir(common_dir, "rev-parse", "--verify", "HEAD")
        if head:
            lines = [f"HEAD\x00{head}"]
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    tip_raw = _git_dir(common_dir, "log", "--all", "-1", "--format=%H%x1f%ct") or ""
    tip_commit = ""
    max_commit_ts = 0
    if tip_raw:
        parts = tip_raw.split("\x1f", 1)
        tip_commit = parts[0].strip()
        if len(parts) > 1:
            try:
                max_commit_ts = int(parts[1].strip() or "0")
            except ValueError:
                max_commit_ts = 0
    return {
        "status": "ready",
        "algorithm": "git-refs-sha256-v1",
        "hash": digest,
        "refs": len(lines),
        "max_commit_ts": max_commit_ts,
        "tip_commit": tip_commit,
    }


def repo_ref_fingerprint(root: str | Path) -> dict:
    """Return a repo-wide fingerprint of all refs reachable from the common dir."""

    if _staleness_disabled():
        return {
            "status": "unavailable",
            "warning": "git disabled by ENGRAM_GIT_STALENESS=0",
            "common_dir": "",
        }
    common = common_dir_for_worktree(root)
    if not common:
        return {"status": "unavailable", "warning": "not a git worktree", "common_dir": ""}
    fp = _fingerprint_from_refs(Path(common))
    fp["common_dir"] = common
    return fp


def _non_git_snapshot(root: Path | None) -> dict:
    logical_project_id = ""
    if root is not None:
        try:
            logical_project_id = paths.project_id_for(root)
        except OSError:
            logical_project_id = ""
    return {
        "logical_project_id": logical_project_id,
        "checkout_kind": "non_git",
        "git_worktree_root": "",
        "indexed_ref": "",
        "indexed_commit": "",
        "indexed_dirty": False,
    }


def _parse_int_stat(value: str) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _parse_raw_entry(line: str) -> dict[str, Any] | None:
    parts = line.split("\t")
    if len(parts) < 2:
        return None
    meta = parts[0].split()
    status_token = meta[-1] if meta else ""
    status = status_token[:1] or "M"
    if status in {"R", "C"} and len(parts) >= 3:
        old_path = _normalize_git_path(parts[1])
        path = _normalize_git_path(parts[2])
    else:
        old_path = ""
        path = _normalize_git_path(parts[-1])
    if not path:
        return None
    out: dict[str, Any] = {"path": path, "status": status}
    if old_path and old_path != path:
        out["old_path"] = old_path
    return out


def _parse_log_output(text: str) -> list[dict]:
    commits: list[dict] = []
    for record in text.split(_LOG_RECORD_SEP):
        record = record.strip("\r\n")
        if not record:
            continue
        lines = record.splitlines()
        if not lines:
            continue
        fields = lines[0].split(_LOG_FIELD_SEP, 4)
        if len(fields) != 5:
            continue
        commit, parents_raw, ts_raw, author, message = fields
        commit = commit.strip()
        if not commit:
            continue
        parents = [p for p in parents_raw.split() if p]
        try:
            ts = int(ts_raw)
        except ValueError:
            ts = 0
        raw_entries: list[dict[str, Any]] = []
        paths: list[dict[str, Any]] = []
        stat_index = 0
        for line in lines[1:]:
            if not line:
                continue
            if line.startswith(":"):
                raw_entry = _parse_raw_entry(line)
                if raw_entry is not None:
                    raw_entries.append(raw_entry)
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            added_raw, deleted_raw = parts[0], parts[1]
            if not (
                added_raw == "-"
                or deleted_raw == "-"
                or added_raw.isdigit()
                or deleted_raw.isdigit()
            ):
                continue
            base: dict[str, Any]
            if stat_index < len(raw_entries):
                base = dict(raw_entries[stat_index])
            else:
                base = {
                    "path": _normalize_git_path("\t".join(parts[2:])),
                    "status": "M",
                }
            stat_index += 1
            path = _normalize_git_path(base.get("path"))
            if not path:
                continue
            entry = {
                "path": path,
                "status": str(base.get("status") or "M"),
                "added": _parse_int_stat(added_raw),
                "deleted": _parse_int_stat(deleted_raw),
            }
            old_path = _normalize_git_path(base.get("old_path"))
            if old_path and old_path != path:
                entry["old_path"] = old_path
            paths.append(entry)
        commits.append(
            {
                "commit": commit,
                "parents": parents,
                "ts": ts,
                "author": author[:200],
                "message": message[:240],
                "paths": paths,
            }
        )
    return commits


def _coerce_max_commits(max_commits: int | None) -> int | None:
    if max_commits is None:
        return None
    try:
        value = int(max_commits)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return min(value, 1_000_000)


def _limit_commits(commits: list[dict], max_commits: int | None) -> list[dict]:
    limit = _coerce_max_commits(max_commits)
    if limit is None:
        return list(commits)
    return list(commits)[:limit]


def _strip_diff_path(value: str) -> str:
    text = value.strip()
    if not text or text == "/dev/null":
        return ""
    if "\t" in text:
        text = text.split("\t", 1)[0]
    if " " in text and (text.startswith("a/") or text.startswith("b/")):
        text = text.split(" ", 1)[0]
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    if text.startswith("a/") or text.startswith("b/"):
        text = text[2:]
    return _normalize_git_path(text)


def _old_line_ranges_from_diff(text: str) -> dict[str, list[tuple[int, int]]]:
    ranges: dict[str, list[tuple[int, int]]] = {}
    old_path = ""
    for line in text.splitlines():
        if line.startswith("--- "):
            old_path = _strip_diff_path(line[4:])
            continue
        match = _HUNK_RE.search(line)
        if not match or not old_path:
            continue
        old_start = _parse_int_stat(match.group("old_start"))
        old_count = _parse_int_stat(match.group("old_count") or "1")
        if old_start <= 0 or old_count <= 0:
            continue
        ranges.setdefault(old_path, []).append((old_start, old_count))
    return ranges


def _blame_commits_for_range(
    root: Path,
    *,
    revision: str,
    path: str,
    start_line: int,
    line_count: int,
    excluded_commit: str,
) -> Counter[str]:
    end_line = start_line + line_count - 1
    out = _git(
        root,
        "blame",
        "-w",
        "--root",
        "--line-porcelain",
        "-L",
        f"{start_line},{end_line}",
        revision,
        "--",
        path,
        timeout_sec=git_index_timeout_seconds(),
    )
    commits: Counter[str] = Counter()
    if out is None:
        return commits
    for line in out.splitlines():
        token = line.split(" ", 1)[0]
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", token):
            continue
        if token == excluded_commit or set(token) == {"0"}:
            continue
        commits[token] += 1
    return commits


def _szz_empty_payload(
    *,
    status: str,
    warning: str,
    fix_regex: str | None,
    workers: int,
    warnings: list[str] | None = None,
) -> dict:
    warning_list = list(warnings or ([warning] if warning else []))
    return {
        "status": status,
        "warning": warning,
        "fix_regex": fix_regex or DEFAULT_FIX_REGEX,
        "fix_commits": 0,
        "blamed_lines": 0,
        "attributions": [],
        "commit_attributions": {},
        "workers": workers,
        "cached_commits": 0,
        "blamed_commits": 0,
        "timings_ms": {"total": 0.0, "blame": 0.0},
        "warnings": warning_list[:50],
    }


def _fix_candidates_and_messages(
    commits: Iterable[dict],
) -> tuple[list[dict], list[str]]:
    candidates: list[dict] = []
    messages: list[str] = []
    seen: set[str] = set()
    for commit in commits:
        if not isinstance(commit, dict):
            continue
        commit_id = str(commit.get("commit") or "")
        if not commit_id or commit_id in seen:
            continue
        parents = [str(parent) for parent in (commit.get("parents") or []) if str(parent)]
        if len(parents) != 1:
            continue
        seen.add(commit_id)
        candidates.append(commit)
        messages.append(str(commit.get("message") or ""))
    return candidates, messages


def _fix_commits_for_szz(
    commits: Iterable[dict],
    fix_regex: str | None,
) -> tuple[str, list[dict], list[str]]:
    """Resolve requested git_fix_regex to the effective regex over real messages.

    This is the single point where the caller-supplied regex is actually
    executed against commit messages (in a bounded worker when non-default).
    Callers that already have this result (e.g. from ``history_for_shared_store``)
    must reuse it via ``fix_commits_from_ids`` instead of resolving again —
    each additional resolution is a chance for a nondeterministic worker
    timeout/crash to disagree with the first resolution and desync the
    persisted fix-regex identity.
    """

    candidates, messages = _fix_candidates_and_messages(commits)
    out: list[dict] = []
    fix_regex_value, hits, regex_warnings = regexsafe.search_many_or_default(
        fix_regex,
        DEFAULT_FIX_REGEX,
        messages,
        label="git_fix_regex",
    )
    regex_warnings = _fix_regex_warnings(fix_regex, fix_regex_value, regex_warnings)
    for commit, hit in zip(candidates, hits, strict=False):
        if hit:
            out.append(commit)
    return fix_regex_value, out, regex_warnings


def fix_commits_from_ids(commits: Iterable[dict], fix_commit_ids: Iterable[str] | None) -> list[dict]:
    """Recover the fix-commit dicts a prior regex resolution selected.

    Used to reuse a fix-commit identity computed once (e.g. at history-build
    time) without re-executing the caller's regex against commit messages.
    """

    if fix_commit_ids is None:
        return []
    wanted = {str(cid) for cid in fix_commit_ids if str(cid)}
    if not wanted:
        return []
    return [commit for commit in commits if isinstance(commit, dict) and str(commit.get("commit") or "") in wanted]


def _normalize_szz_attribution(item: dict, *, fix_commit: str) -> dict | None:
    introducing_commit = str(item.get("introducing_commit") or "")
    rel_path = _normalize_git_path(item.get("path"))
    if not fix_commit or not introducing_commit or not rel_path:
        return None
    try:
        lines = max(1, int(item.get("lines", 1) or 1))
    except (TypeError, ValueError):
        lines = 1
    return {
        "fix_commit": fix_commit,
        "introducing_commit": introducing_commit,
        "path": rel_path,
        "lines": lines,
    }


def _normalize_szz_commit_payload(fix_commit: str, payload: dict) -> dict | None:
    if not fix_commit or not isinstance(payload, dict):
        return None
    counts: Counter[tuple[str, str, str]] = Counter()
    for item in payload.get("attributions") or []:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_szz_attribution(item, fix_commit=fix_commit)
        if normalized is None:
            continue
        key = (
            normalized["fix_commit"],
            normalized["introducing_commit"],
            normalized["path"],
        )
        counts[key] += int(normalized["lines"])
    attributions = [
        {
            "fix_commit": fix,
            "introducing_commit": introducing,
            "path": rel_path,
            "lines": lines,
        }
        for (fix, introducing, rel_path), lines in sorted(counts.items())
    ]
    warnings = [str(w) for w in (payload.get("warnings") or []) if str(w)]
    status = str(payload.get("status") or ("partial" if warnings else "ready"))
    if status not in {"ready", "partial"}:
        status = "partial" if warnings else "ready"
    return {
        "fix_commit": fix_commit,
        "status": status,
        "attributions": attributions,
        "blamed_lines": sum(int(item.get("lines", 0) or 0) for item in attributions),
        "warnings": warnings[:50],
    }


def _szz_cache_from_previous(previous: dict | None) -> dict[str, dict]:
    if not isinstance(previous, dict):
        return {}
    cache: dict[str, dict] = {}
    commit_payloads = previous.get("commit_attributions")
    if isinstance(commit_payloads, dict):
        for raw_commit, payload in commit_payloads.items():
            fix_commit = str(raw_commit or "")
            normalized = _normalize_szz_commit_payload(fix_commit, payload)
            if normalized is not None:
                cache[fix_commit] = normalized
    for item in previous.get("attributions") or []:
        if not isinstance(item, dict):
            continue
        fix_commit = str(item.get("fix_commit") or "")
        normalized = _normalize_szz_attribution(item, fix_commit=fix_commit)
        if normalized is None or fix_commit in cache:
            continue
        cache[fix_commit] = {
            "fix_commit": fix_commit,
            "status": "partial" if previous.get("status") == "partial" else "ready",
            "attributions": [],
            "blamed_lines": 0,
            "warnings": [],
        }
    if previous.get("attributions"):
        grouped: dict[str, list[dict]] = {}
        for item in previous.get("attributions") or []:
            if not isinstance(item, dict):
                continue
            fix_commit = str(item.get("fix_commit") or "")
            normalized = _normalize_szz_attribution(item, fix_commit=fix_commit)
            if normalized is not None:
                grouped.setdefault(fix_commit, []).append(normalized)
        for fix_commit, attributions in grouped.items():
            if fix_commit in cache and cache[fix_commit].get("attributions"):
                continue
            payload = _normalize_szz_commit_payload(
                fix_commit,
                {
                    "status": "partial" if previous.get("status") == "partial" else "ready",
                    "attributions": attributions,
                    "warnings": [],
                },
            )
            if payload is not None:
                cache[fix_commit] = payload
    return cache


def _failed_szz_commit_payload(fix_commit: str, warning: str) -> dict:
    return {
        "fix_commit": fix_commit,
        "status": "partial",
        "attributions": [],
        "blamed_lines": 0,
        "warnings": [warning],
    }


def _szz_attributions_for_fix_commit(root: Path, commit: dict) -> dict:
    commit_id = str(commit.get("commit") or "")
    parents = [str(parent) for parent in (commit.get("parents") or []) if str(parent)]
    if not commit_id or len(parents) != 1:
        return _failed_szz_commit_payload(commit_id, "invalid fix commit metadata")
    parent = parents[0]
    warnings: list[str] = []
    attribution_counts: Counter[tuple[str, str, str]] = Counter()
    diff = _git(
        root,
        "diff",
        "--unified=0",
        "-w",
        "--no-ext-diff",
        "--no-color",
        "-M",
        parent,
        commit_id,
        "--",
        timeout_sec=git_index_timeout_seconds(),
    )
    if diff is None:
        return _failed_szz_commit_payload(commit_id, f"diff unavailable for {commit_id[:12]}")
    ranges_by_path = _old_line_ranges_from_diff(diff)
    for rel_path, ranges in sorted(ranges_by_path.items()):
        for start_line, line_count in ranges:
            blamed = _blame_commits_for_range(
                root,
                revision=parent,
                path=rel_path,
                start_line=start_line,
                line_count=line_count,
                excluded_commit=commit_id,
            )
            if not blamed:
                warnings.append(f"blame unavailable for {commit_id[:12]}:{rel_path}:{start_line}")
                continue
            for introducing_commit, lines in blamed.items():
                attribution_counts[(commit_id, introducing_commit, rel_path)] += int(lines)
    attributions = [
        {
            "fix_commit": fix_commit,
            "introducing_commit": introducing_commit,
            "path": rel_path,
            "lines": lines,
        }
        for (fix_commit, introducing_commit, rel_path), lines in sorted(attribution_counts.items())
    ]
    return {
        "fix_commit": commit_id,
        "status": "partial" if warnings else "ready",
        "attributions": attributions,
        "blamed_lines": sum(int(item.get("lines", 0) or 0) for item in attributions),
        "warnings": warnings[:50],
    }


def _combine_szz_commit_cache(
    *,
    fix_commits: list[dict],
    cache: dict[str, dict],
    fix_regex: str,
    workers: int,
    cached_commits: int,
    blamed_commits: int,
    worker_warning: str | None,
    total_start: float,
    blame_start: float,
    complete: bool,
) -> dict:
    fix_ids = [str(commit.get("commit") or "") for commit in fix_commits if str(commit.get("commit") or "")]
    selected: dict[str, dict] = {}
    for fix_id in fix_ids:
        payload = _normalize_szz_commit_payload(fix_id, cache.get(fix_id) or {})
        if payload is not None:
            selected[fix_id] = payload

    counts: Counter[tuple[str, str, str]] = Counter()
    warnings: list[str] = []
    for fix_id in sorted(selected):
        payload = selected[fix_id]
        warnings.extend(str(w) for w in (payload.get("warnings") or []) if str(w))
        for item in payload.get("attributions") or []:
            normalized = _normalize_szz_attribution(item, fix_commit=fix_id)
            if normalized is None:
                continue
            key = (
                normalized["fix_commit"],
                normalized["introducing_commit"],
                normalized["path"],
            )
            counts[key] += int(normalized["lines"])
    if worker_warning:
        warnings.insert(0, worker_warning)
    attributions = [
        {
            "fix_commit": fix_commit,
            "introducing_commit": introducing_commit,
            "path": rel_path,
            "lines": lines,
        }
        for (fix_commit, introducing_commit, rel_path), lines in sorted(counts.items())
    ]
    missing = [fix_id for fix_id in fix_ids if fix_id not in selected]
    partial_commits = [
        fix_id for fix_id, payload in selected.items() if payload.get("status") == "partial"
    ]
    status = "ready"
    if missing or partial_commits or warnings or not complete:
        status = "partial"
    return {
        "status": status,
        "warning": "; ".join(warnings[:5]),
        "fix_regex": fix_regex,
        "fix_commits": len(fix_ids),
        "blamed_lines": sum(int(item.get("lines", 0) or 0) for item in attributions),
        "attributions": attributions,
        "commit_attributions": {fix_id: selected[fix_id] for fix_id in sorted(selected)},
        "workers": workers,
        "cached_commits": cached_commits,
        "blamed_commits": blamed_commits,
        "timings_ms": {
            "total": _ms_since(total_start),
            "blame": _ms_since(blame_start) if blamed_commits else 0.0,
        },
        "warnings": warnings[:50],
    }


def szz_attributions_with_status(
    root: str | Path,
    commits: Iterable[dict],
    *,
    fix_regex: str | None = None,
    fix_commits: Iterable[dict] | None = None,
    previous: dict | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Return SZZ-style fix-to-introducer blame attributions.

    Failures are reported as ``partial``/``unavailable`` instead of raised so
    index jobs do not fail solely because optional VCS analytics degraded.

    ``fix_commits``, when given, is the fix-commit identity a prior regex
    resolution already computed (e.g. via ``history_for_shared_store`` /
    ``fix_commits_from_ids``) for this exact ``fix_regex`` + corpus. Passing it
    skips re-executing the caller regex here: a second execution of a
    catastrophic-backtracking pattern is a second chance for the bounded
    worker to time out/crash and disagree with the first resolution, which
    would desync the persisted fix-regex identity between the history and the
    SZZ sidecar. Omit it only for standalone callers (tests, ad-hoc CLI use)
    that have not already resolved the regex.
    """

    total_start = time.perf_counter()
    workers, worker_warning = _szz_worker_config()
    fix_regex_value, regex_warnings = regexsafe.pattern_or_default(
        fix_regex,
        DEFAULT_FIX_REGEX,
        label="git_fix_regex",
    )
    regex_warnings = _fix_regex_warnings(fix_regex, fix_regex_value, regex_warnings)
    if _staleness_disabled():
        return {
            **_szz_empty_payload(
                status="unavailable",
                warning="git disabled by ENGRAM_GIT_STALENESS=0",
                fix_regex=fix_regex_value,
                workers=workers,
            ),
            "timings_ms": {"total": _ms_since(total_start), "blame": 0.0},
        }
    try:
        path = Path(root).resolve()
    except OSError as exc:
        return {
            **_szz_empty_payload(
                status="unavailable",
                warning=str(exc),
                fix_regex=fix_regex_value,
                workers=workers,
            ),
            "timings_ms": {"total": _ms_since(total_start), "blame": 0.0},
        }
    if fix_commits is not None:
        fix_commits = [c for c in fix_commits if isinstance(c, dict)]
    else:
        fix_regex_value, fix_commits, regex_warnings = _fix_commits_for_szz(commits, fix_regex)
    cache = _szz_cache_from_previous(previous)
    fix_ids = {str(commit.get("commit") or "") for commit in fix_commits}
    cached = {fix_id: payload for fix_id, payload in cache.items() if fix_id in fix_ids}
    to_blame = [commit for commit in fix_commits if str(commit.get("commit") or "") not in cached]
    cached_commits = len(cached)
    blamed_commits = 0
    blame_start = time.perf_counter()
    combined_warning = "; ".join(
        [*(regex_warnings or []), *([worker_warning] if worker_warning else [])]
    ) or None

    def emit_progress() -> None:
        if progress is None:
            return
        progress(
            _combine_szz_commit_cache(
                fix_commits=fix_commits,
                cache=cached,
                fix_regex=fix_regex_value,
                workers=workers,
                cached_commits=cached_commits,
                blamed_commits=blamed_commits,
                worker_warning=combined_warning,
                total_start=total_start,
                blame_start=blame_start,
                complete=False,
            )
        )

    if workers <= 1:
        for commit in to_blame:
            fix_id = str(commit.get("commit") or "")
            try:
                cached[fix_id] = _szz_attributions_for_fix_commit(path, commit)
            except Exception as exc:
                cached[fix_id] = _failed_szz_commit_payload(
                    fix_id,
                    f"szz unavailable for {fix_id[:12]}: {exc}",
                )
            blamed_commits += 1
            emit_progress()
    elif to_blame:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="engram-szz") as pool:
            futures = {
                pool.submit(_szz_attributions_for_fix_commit, path, commit): str(
                    commit.get("commit") or ""
                )
                for commit in to_blame
            }
            for future in as_completed(futures):
                fix_id = futures[future]
                try:
                    cached[fix_id] = future.result()
                except Exception as exc:
                    cached[fix_id] = _failed_szz_commit_payload(
                        fix_id,
                        f"szz unavailable for {fix_id[:12]}: {exc}",
                    )
                blamed_commits += 1
                emit_progress()

    return _combine_szz_commit_cache(
        fix_commits=fix_commits,
        cache=cached,
        fix_regex=fix_regex_value,
        workers=workers,
        cached_commits=cached_commits,
        blamed_commits=blamed_commits,
        worker_warning=combined_warning,
        total_start=total_start,
        blame_start=blame_start,
        complete=True,
    )


def commit_log_with_status(
    root: str | Path,
    max_commits: int | None = None,
    rev_range: str | None = None,
    *,
    all_refs: bool = False,
    git_dir: bool = False,
    timeout_sec: float | None = None,
) -> dict:
    """Return a compact parsed git history or an unavailable status.

    This helper uses the same prompt/lock/timeout-proof process wrapper as the
    staleness helpers. It never raises for git failures.
    """

    if _staleness_disabled():
        return {
            "status": "unavailable",
            "warning": "git disabled by ENGRAM_GIT_STALENESS=0",
            "commits": [],
        }
    try:
        path = Path(root).resolve()
    except OSError as exc:
        return {"status": "unavailable", "warning": str(exc), "commits": []}
    max_commits = _coerce_max_commits(max_commits)
    if not git_dir and not (_git(path, "rev-parse", "--is-inside-work-tree") or "").strip():
        return {"status": "unavailable", "warning": "not a git worktree", "commits": []}
    fmt = f"{_LOG_RECORD_SEP}%H{_LOG_FIELD_SEP}%P{_LOG_FIELD_SEP}%ct{_LOG_FIELD_SEP}%an{_LOG_FIELD_SEP}%s"
    args = [
        "log",
        "--no-ext-diff",
        "--date-order",
        "-M",
        "--raw",
        "--numstat",
        f"--format={fmt}",
    ]
    if all_refs:
        args.append("--all")
    if max_commits is not None:
        args.extend(["-n", str(max_commits)])
    if rev_range:
        args.append(rev_range)
    args.append("--")
    timeout = _GIT_STALENESS_TIMEOUT_SEC if timeout_sec is None else timeout_sec
    out = (
        _git_dir(path, *args, timeout_sec=timeout)
        if git_dir
        else _git(path, *args, timeout_sec=timeout)
    )
    if out is None:
        return {"status": "unavailable", "warning": "git log unavailable", "commits": []}
    return {"status": "ready", "warning": "", "commits": _parse_log_output(out)}


def history_for_shared_store(
    root: str | Path,
    *,
    logical_project_id: str,
    checkout_kind: str = "",
    fingerprint: dict | None = None,
    fix_regex: str | None = None,
    max_commits: int | None = None,
    timeout_sec: float | None = None,
) -> dict:
    """Build the repo-wide git-history payload for the shared analytics store."""

    fp = fingerprint if isinstance(fingerprint, dict) else repo_ref_fingerprint(root)
    max_commits = _coerce_max_commits(max_commits)
    requested_fix_regex = requested_fix_regex_value(fix_regex)
    fix_regex_value, regex_warnings = regexsafe.pattern_or_default(
        fix_regex,
        DEFAULT_FIX_REGEX,
        label="git_fix_regex",
    )
    regex_warnings = _fix_regex_warnings(fix_regex, fix_regex_value, regex_warnings)
    common_dir = str(fp.get("common_dir") or "")
    base = {
        "schema_version": 1,
        "logical_project_id": logical_project_id,
        "checkout_kind": checkout_kind,
        "status": "unavailable",
        "max_commits": max_commits,
        "head_commit": str(fp.get("tip_commit") or ""),
        "git_common_dir": common_dir,
        "fingerprint": {
            "algorithm": str(fp.get("algorithm") or "git-refs-sha256-v1"),
            "hash": str(fp.get("hash") or ""),
            "refs": int(fp.get("refs", 0) or 0),
            "max_commit_ts": int(fp.get("max_commit_ts", 0) or 0),
            "tip_commit": str(fp.get("tip_commit") or ""),
        },
        "fix_regex": fix_regex_value,
        "requested_fix_regex": requested_fix_regex,
        "commits": [],
        "warning": "; ".join(regex_warnings),
        "warnings": regex_warnings,
    }
    if fp.get("status") != "ready" or not common_dir:
        warning = str(fp.get("warning") or "git refs unavailable")
        warnings = [*regex_warnings, warning] if warning else regex_warnings
        return base | {"warning": "; ".join(warnings), "warnings": warnings}
    status = commit_log_with_status(
        Path(common_dir),
        max_commits=max_commits,
        all_refs=True,
        git_dir=True,
        timeout_sec=_GIT_STALENESS_TIMEOUT_SEC if timeout_sec is None else timeout_sec,
    )
    if status.get("status") != "ready":
        warning = str(status.get("warning") or "git log --all unavailable")
        warnings = [*regex_warnings, warning] if warning else regex_warnings
        return base | {"warning": "; ".join(warnings), "warnings": warnings}
    commits = [c for c in (status.get("commits") or []) if isinstance(c, dict)]
    head_commit = str(fp.get("tip_commit") or (commits[0].get("commit") if commits else "") or "")
    # Resolve the caller regex against real commit messages exactly once here;
    # persist which commits matched so downstream counting/SZZ passes reuse
    # this identity instead of re-executing the (possibly unsafe) regex.
    fix_regex_value, fix_commits, warnings = _fix_commits_for_szz(commits, fix_regex)
    return base | {
        "status": "ready",
        "fix_regex": fix_regex_value,
        "requested_fix_regex": requested_fix_regex,
        "warning": "; ".join(warnings),
        "head_commit": head_commit,
        "commits": commits,
        "fix_commit_ids": [str(c.get("commit") or "") for c in fix_commits],
        "warnings": warnings,
    }


def snapshot(root: str | Path) -> dict:
    """Return git state for ``root`` without mutating the repository."""

    try:
        path = Path(root).resolve()
    except OSError:
        return _non_git_snapshot(None)
    rev_parse = _git(
        path,
        "rev-parse",
        "--path-format=absolute",
        "--show-toplevel",
        "--git-common-dir",
        "--git-dir",
    )
    if not rev_parse:
        rev_parse = _git(
            path,
            "rev-parse",
            "--show-toplevel",
            "--git-common-dir",
            "--git-dir",
        )
    if not rev_parse:
        return _non_git_snapshot(path)
    rev_lines = [line.strip() for line in rev_parse.splitlines()]
    if len(rev_lines) < 3 or not rev_lines[0]:
        return _non_git_snapshot(path)
    worktree = rev_lines[0]
    worktree_root = _canonical_git_path(worktree, base=path)
    common_dir = _canonical_git_path(rev_lines[1], base=path)
    git_dir = _canonical_git_path(rev_lines[2], base=path)
    checkout_kind = "main"
    if common_dir and git_dir and _path_key(common_dir) != _path_key(git_dir):
        checkout_kind = "worktree"
    logical_project_id = ""
    if common_dir:
        try:
            logical_project_id = paths.logical_project_id_for_common_dir(common_dir)
        except OSError:
            logical_project_id = ""
    if not logical_project_id and worktree_root:
        try:
            logical_project_id = paths.project_id_for(Path(worktree_root))
        except OSError:
            logical_project_id = ""
    ref, commit, dirty = _parse_status_v2_branch(
        _git(path, "status", "--porcelain=v2", "--branch")
    )
    return {
        "logical_project_id": logical_project_id,
        "checkout_kind": checkout_kind,
        "git_worktree_root": worktree_root,
        "indexed_ref": ref,
        "indexed_commit": commit,
        "indexed_dirty": dirty,
    }


def current_staleness(root: str | Path, indexed: dict) -> dict:
    """Compare current git state against manifest fields."""

    try:
        current = snapshot(root)
    except Exception as exc:
        return {
            "available": False,
            "git_stale": False,
            "reasons": [],
            "indexed": indexed,
            "current": {},
            "warning": f"git metadata unavailable: {exc}",
        }
    if not indexed.get("git_worktree_root") and not current.get("git_worktree_root"):
        return {
            "available": False,
            "git_stale": False,
            "reasons": [],
            "indexed": indexed,
            "current": current,
        }
    reasons: list[str] = []
    for key in ("git_worktree_root", "indexed_ref", "indexed_commit", "indexed_dirty"):
        if indexed.get(key) != current.get(key):
            reasons.append(key)
    return {
        "available": bool(current.get("git_worktree_root") or indexed.get("git_worktree_root")),
        "git_stale": bool(reasons),
        "reasons": reasons,
        "indexed": indexed,
        "current": current,
    }


def _snapshot_field(snapshot_data: dict, manifest_key: str, normalized_key: str, default=None):
    if manifest_key in snapshot_data:
        return snapshot_data.get(manifest_key)
    return snapshot_data.get(normalized_key, default)


def _normalize_revision_snapshot(snapshot_data: dict | None) -> dict:
    data = snapshot_data if isinstance(snapshot_data, dict) else {}
    dirty = _snapshot_field(data, "indexed_dirty", "dirty", False)
    return {
        "worktree_root": _snapshot_field(data, "git_worktree_root", "worktree_root", "") or "",
        "ref": _snapshot_field(data, "indexed_ref", "ref", "") or "",
        "commit": _snapshot_field(data, "indexed_commit", "commit", "") or "",
        "dirty": bool(dirty),
    }


def source_revision_from_staleness(git_status: dict | None) -> dict:
    """Normalize ``current_staleness`` output for search responses.

    This is a pure formatter: callers must pass an already-computed git status.
    It intentionally performs no repository inspection.
    """

    status = git_status if isinstance(git_status, dict) else {}
    available = bool(status.get("available"))
    indexed = _normalize_revision_snapshot(status.get("indexed"))
    current = _normalize_revision_snapshot(status.get("current"))
    reasons = [str(reason) for reason in (status.get("reasons") or [])]

    branch_mismatch = available and indexed["ref"] != current["ref"]
    commit_mismatch = available and indexed["commit"] != current["commit"]
    dirty_mismatch = available and indexed["dirty"] != current["dirty"]
    if not available:
        reasons = []
    else:
        for flag, reason in (
            (branch_mismatch, "indexed_ref"),
            (commit_mismatch, "indexed_commit"),
            (dirty_mismatch, "indexed_dirty"),
        ):
            if flag and reason not in reasons:
                reasons.append(reason)

    return {
        "available": available,
        "indexed": indexed,
        "current": current,
        "stale": available and bool(status.get("git_stale") or reasons),
        "branch_mismatch": bool(branch_mismatch),
        "commit_mismatch": bool(commit_mismatch),
        "dirty_mismatch": bool(dirty_mismatch),
        "reasons": reasons,
    }


def _revision_label(value: str, *, shorten: bool = False) -> str:
    text = value or "unknown"
    if shorten and len(text) > 12:
        return text[:12]
    return text


def source_revision_warning(
    source_revision: dict | None,
    *,
    include_commit_mismatch: bool = False,
) -> str | None:
    """Return a concise human/action warning for a formatted source revision."""

    revision = source_revision if isinstance(source_revision, dict) else {}
    if not revision.get("available"):
        return None
    indexed = revision.get("indexed") if isinstance(revision.get("indexed"), dict) else {}
    current = revision.get("current") if isinstance(revision.get("current"), dict) else {}
    if revision.get("branch_mismatch"):
        return (
            f"results are from indexed ref '{_revision_label(indexed.get('ref') or '')}', "
            f"not current ref '{_revision_label(current.get('ref') or '')}'; "
            "reindex the worktree to search it"
        )
    if include_commit_mismatch and revision.get("commit_mismatch"):
        return (
            f"results are from indexed commit "
            f"'{_revision_label(indexed.get('commit') or '', shorten=True)}', "
            f"not current commit '{_revision_label(current.get('commit') or '', shorten=True)}'; "
            "reindex the worktree to search it"
        )
    return None
