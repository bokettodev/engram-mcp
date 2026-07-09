"""Read-only git metadata helpers.

The metadata is diagnostic only. Engram records commit/ref/dirty state so tools
can report staleness, but it never uses churn or recency as a relevance prior.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

from engram_mcp import paths

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
_GIT_TIMEOUT_SEC = 3.0
_LOG_RECORD_SEP = "\x1e"
_LOG_FIELD_SEP = "\x1f"
DEFAULT_FIX_REGEX = r"(?i)\b(fix(e[sd])?|bug|hotfix|patch|close[sd]?\s+#\d+)\b"
_HUNK_RE = re.compile(r"@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? \+\d+(?:,\d+)? @@")


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


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


def _git(root: Path, *args: str) -> str | None:
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
        out, _ = proc.communicate(timeout=_GIT_TIMEOUT_SEC)
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
    )
    commits: Counter[str] = Counter()
    if out is None:
        return commits
    for line in out.splitlines():
        token = line.split(" ", 1)[0]
        if not re.fullmatch(r"[0-9a-f]{40}", token):
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


def _fix_commits_for_szz(commits: Iterable[dict], fix_rx: re.Pattern[str]) -> list[dict]:
    out: list[dict] = []
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
        if not fix_rx.search(str(commit.get("message") or "")):
            continue
        seen.add(commit_id)
        out.append(commit)
    return out


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
    previous: dict | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Return SZZ-style fix-to-introducer blame attributions.

    Failures are reported as ``partial``/``unavailable`` instead of raised so
    index jobs do not fail solely because optional VCS analytics degraded.
    """

    total_start = time.perf_counter()
    workers, worker_warning = _szz_worker_config()
    fix_regex_value = fix_regex or DEFAULT_FIX_REGEX
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
    try:
        fix_rx = re.compile(fix_regex_value)
    except re.error as exc:
        return {
            **_szz_empty_payload(
                status="unavailable",
                warning=f"invalid fix regex: {exc}",
                fix_regex=fix_regex_value,
                workers=workers,
            ),
            "timings_ms": {"total": _ms_since(total_start), "blame": 0.0},
        }

    fix_commits = _fix_commits_for_szz(commits, fix_rx)
    cache = _szz_cache_from_previous(previous)
    fix_ids = {str(commit.get("commit") or "") for commit in fix_commits}
    cached = {fix_id: payload for fix_id, payload in cache.items() if fix_id in fix_ids}
    to_blame = [commit for commit in fix_commits if str(commit.get("commit") or "") not in cached]
    cached_commits = len(cached)
    blamed_commits = 0
    blame_start = time.perf_counter()

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
                worker_warning=worker_warning,
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
        worker_warning=worker_warning,
        total_start=total_start,
        blame_start=blame_start,
        complete=True,
    )


def commit_log_with_status(
    root: str | Path,
    max_commits: int | None = None,
    rev_range: str | None = None,
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
    if not (_git(path, "rev-parse", "--is-inside-work-tree") or "").strip():
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
    if max_commits is not None:
        args.extend(["-n", str(max_commits)])
    if rev_range:
        args.append(rev_range)
    args.append("--")
    out = _git(path, *args)
    if out is None:
        return {"status": "unavailable", "warning": "git log unavailable", "commits": []}
    return {"status": "ready", "warning": "", "commits": _parse_log_output(out)}


def commit_log(root: str | Path, max_commits: int | None = None) -> list[dict]:
    """Walk ``git log`` and return compact commit/file statistics.

    Failures return an empty list; callers that need to distinguish empty
    history from unavailable git should use ``commit_log_with_status``.
    """

    return list(commit_log_with_status(root, max_commits=max_commits).get("commits") or [])


def head_commit(root: str | Path) -> str:
    try:
        path = Path(root).resolve()
    except OSError:
        return ""
    return _git(path, "rev-parse", "HEAD") or ""


def is_ancestor(root: str | Path, ancestor: str, descendant: str) -> bool:
    if not ancestor or not descendant:
        return False
    try:
        path = Path(root).resolve()
    except OSError:
        return False
    out = _git(path, "merge-base", "--is-ancestor", ancestor, descendant)
    return out is not None


def history_for_catalog(
    root: str | Path,
    *,
    max_commits: int | None = None,
    previous: dict | None = None,
    head: str | None = None,
    fix_regex: str | None = None,
) -> dict:
    """Build/update the cheap raw git-history block stored in the catalog."""

    max_commits = _coerce_max_commits(max_commits)
    current_head = head or head_commit(root)
    base = {
        "schema_version": 1,
        "status": "unavailable",
        "max_commits": max_commits,
        "head_commit": current_head or "",
        "fix_regex": fix_regex or DEFAULT_FIX_REGEX,
        "commits": [],
    }
    if not current_head:
        return base | {"warning": "git head unavailable"}

    old_head = ""
    old_commits: list[dict] = []
    if isinstance(previous, dict) and previous.get("status") == "ready":
        old_head = str(previous.get("head_commit") or "")
        old_commits = [c for c in (previous.get("commits") or []) if isinstance(c, dict)]
    previous_max = _coerce_max_commits(previous.get("max_commits")) if isinstance(previous, dict) else None

    if old_head and old_head == current_head and previous_max == max_commits:
        return base | {
            "status": "ready",
            "commits": _limit_commits(old_commits, max_commits),
        }

    if old_head and previous_max == max_commits and is_ancestor(root, old_head, current_head):
        delta = commit_log_with_status(root, max_commits=max_commits, rev_range=f"{old_head}..{current_head}")
        if delta.get("status") != "ready":
            return base | {"warning": delta.get("warning") or "git log unavailable"}
        seen: set[str] = set()
        commits: list[dict] = []
        for item in list(delta.get("commits") or []) + old_commits:
            commit = str(item.get("commit") or "")
            if not commit or commit in seen:
                continue
            seen.add(commit)
            commits.append(item)
            if max_commits is not None and len(commits) >= max_commits:
                break
        return base | {
            "status": "ready",
            "commits": commits,
        }

    full = commit_log_with_status(root, max_commits=max_commits)
    if full.get("status") != "ready":
        return base | {"warning": full.get("warning") or "git log unavailable"}
    return base | {
        "status": "ready",
        "commits": _limit_commits(list(full.get("commits") or []), max_commits),
    }


def snapshot(root: str | Path) -> dict:
    """Return git state for ``root`` without mutating the repository."""

    try:
        path = Path(root).resolve()
    except OSError:
        return _non_git_snapshot(None)
    worktree = _git(path, "rev-parse", "--show-toplevel")
    if not worktree:
        return _non_git_snapshot(path)
    worktree_root = _canonical_git_path(worktree, base=path)
    common_dir = _canonical_git_path(
        _git(path, "rev-parse", "--git-common-dir"),
        base=path,
    )
    git_dir = _canonical_git_path(_git(path, "rev-parse", "--git-dir"), base=path)
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
    commit = _git(path, "rev-parse", "HEAD") or ""
    ref = _git(path, "symbolic-ref", "--short", "-q", "HEAD")
    if not ref:
        ref = _git(path, "rev-parse", "--short", "HEAD") or ""
    status = _git(path, "status", "--porcelain")
    return {
        "logical_project_id": logical_project_id,
        "checkout_kind": checkout_kind,
        "git_worktree_root": worktree_root,
        "indexed_ref": ref,
        "indexed_commit": commit,
        "indexed_dirty": bool(status),
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
