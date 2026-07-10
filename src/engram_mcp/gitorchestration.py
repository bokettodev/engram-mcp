"""Shared git-analytics orchestration and SZZ background task state."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from engram_mcp import gitmeta, gitstore, paths, regexsafe

_SZZ_TASKS: dict[tuple[str, str, str, str], threading.Thread] = {}
_SZZ_TASKS_LOCK = threading.Lock()


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


#: Server-side maximum for the caller-supplied ``git_max_commits`` budget. A
#: request above this is clamped down (never rejected); see README
#: "Server-side limits" and structure_service._attach_git_analytics's warning surfacing.
MAX_GIT_MAX_COMMITS = 1_000_000


def coerce_git_max_commits(value: int | None) -> int | None:
    """Validate and clamp the shared git-history commit budget."""

    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return min(parsed, MAX_GIT_MAX_COMMITS)


# Compatibility alias for callers that predate the public repository seam.
_coerce_git_max_commits = coerce_git_max_commits


def _limit_commits(commits: list[dict], max_commits: int | None) -> list[dict]:
    limit = coerce_git_max_commits(max_commits)
    if limit is None:
        return list(commits)
    return list(commits)[:limit]


def _safe_fix_regex(fix_regex: str | None) -> tuple[str, list[str]]:
    return regexsafe.pattern_or_default(
        fix_regex,
        gitmeta.DEFAULT_FIX_REGEX,
        label="git_fix_regex",
    )


def _requested_fix_regex(fix_regex: str | None) -> str:
    return gitmeta.requested_fix_regex_value(fix_regex)


def _source_requested_fix_regex(payload: dict | None) -> str:
    if not isinstance(payload, dict):
        return gitmeta.DEFAULT_FIX_REGEX
    return str(payload.get("requested_fix_regex") or _source_fix_regex(payload))


def _payload_warnings(payload: dict | None) -> list[str]:
    if not isinstance(payload, dict):
        return []
    warnings = [str(w) for w in (payload.get("warnings") or []) if str(w)]
    warning = str(payload.get("warning") or "")
    if warning:
        warnings.append(warning)
    return warnings


def _merge_warnings(*groups: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for warning in group:
            text = str(warning)
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
    return out[:50]


def fix_commit_count(
    commits: list[dict],
    fix_regex: str,
    regex_cache: regexsafe.RegexRequestCache | None = None,
) -> int:
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
    runner = regex_cache.search_many_or_default if regex_cache is not None else regexsafe.search_many_or_default
    _fix_regex, hits, _warnings = runner(
        fix_regex,
        gitmeta.DEFAULT_FIX_REGEX,
        messages,
        label="git_fix_regex",
    )
    return sum(1 for hit in hits if hit)


def _fingerprint_from_payload(payload: dict | None) -> dict:
    if not isinstance(payload, dict):
        return {}
    fp = payload.get("fingerprint")
    return fp if isinstance(fp, dict) else {}


def _fingerprint_hash(payload: dict | None) -> str:
    return str(_fingerprint_from_payload(payload).get("hash") or "")


def _source_fix_regex(payload: dict | None) -> str:
    return str((payload or {}).get("fix_regex") or gitmeta.DEFAULT_FIX_REGEX)


def _source_max_commits(payload: dict | None) -> int | None:
    return coerce_git_max_commits((payload or {}).get("max_commits"))


def _shared_szz_ready(
    szz: dict | None,
    *,
    fingerprint: dict,
    fix_regex: str,
    max_commits: int | None,
) -> bool:
    if not isinstance(szz, dict):
        return False
    if str(szz.get("status") or "") != "ready":
        return False
    if _fingerprint_hash(szz) != str(fingerprint.get("hash") or ""):
        return False
    return _source_fix_regex(szz) == fix_regex and _source_max_commits(szz) == coerce_git_max_commits(max_commits)


def _shared_szz_same_source(
    szz: dict | None,
    *,
    fingerprint: dict,
    fix_regex: str,
    max_commits: int | None,
) -> bool:
    if not isinstance(szz, dict):
        return False
    if _fingerprint_hash(szz) != str(fingerprint.get("hash") or ""):
        return False
    return _source_fix_regex(szz) == fix_regex and _source_max_commits(szz) == coerce_git_max_commits(max_commits)


def _szz_task_key(
    logical_project_id: str,
    fingerprint: dict,
    fix_regex: str,
    max_commits: int | None,
) -> tuple[str, str, str, str]:
    limit = coerce_git_max_commits(max_commits)
    return logical_project_id, str(fingerprint.get("hash") or ""), fix_regex, "" if limit is None else str(limit)


def _szz_task_running(
    logical_project_id: str,
    fingerprint: dict,
    fix_regex: str,
    max_commits: int | None,
) -> bool:
    key = _szz_task_key(logical_project_id, fingerprint, fix_regex, max_commits)
    with _SZZ_TASKS_LOCK:
        thread = _SZZ_TASKS.get(key)
        if thread is None:
            return False
        if thread.is_alive():
            return True
        _SZZ_TASKS.pop(key, None)
        return False


def _wrap_shared_szz(
    *,
    logical_project_id: str,
    history: dict,
    szz: dict,
) -> dict:
    payload = dict(szz)
    warnings = _merge_warnings(_payload_warnings(history), _payload_warnings(payload))
    payload.update(
        {
            "schema_version": gitstore.SCHEMA_VERSION,
            "logical_project_id": logical_project_id,
            "head_commit": str(history.get("head_commit") or ""),
            "max_commits": history.get("max_commits"),
            "git_common_dir": str(history.get("git_common_dir") or ""),
            "fingerprint": dict(_fingerprint_from_payload(history)),
            "fix_regex": str(payload.get("fix_regex") or history.get("fix_regex") or gitmeta.DEFAULT_FIX_REGEX),
            "requested_fix_regex": str(
                payload.get("requested_fix_regex")
                or history.get("requested_fix_regex")
                or history.get("fix_regex")
                or gitmeta.DEFAULT_FIX_REGEX
            ),
            "warnings": warnings,
        }
    )
    if warnings and not str(payload.get("warning") or ""):
        payload["warning"] = "; ".join(warnings[:5])
    return payload


def _write_shared_szz_if_current(
    *,
    logical_project_id: str,
    expected_history: dict,
    payload: dict,
) -> bool:
    expected_fp = _fingerprint_from_payload(expected_history)
    expected_fix_regex = _source_fix_regex(expected_history)
    expected_max_commits = _source_max_commits(expected_history)
    if not expected_fp.get("hash"):
        return False
    # Guard against a payload whose own fix_regex disagrees with the history
    # it was computed against (e.g. a regex-resolution pass that fell back to
    # the default independently of the one baked into `expected_history`).
    # Writing it would desync the persisted identity and wedge SZZ reuse.
    if _source_fix_regex(payload) != expected_fix_regex:
        return False
    with paths.git_analytics_lock(logical_project_id):
        current = gitstore.load_history(logical_project_id)
        current_fp = _fingerprint_from_payload(current)
        if not current_fp.get("hash"):
            return False
        if current_fp.get("hash") != expected_fp.get("hash"):
            return False
        if _source_fix_regex(current) != expected_fix_regex:
            return False
        if _source_max_commits(current) != expected_max_commits:
            return False
        gitstore.save_szz(
            logical_project_id,
            _wrap_shared_szz(
                logical_project_id=logical_project_id,
                history=expected_history,
                szz=payload,
            ),
        )
    return True


def computing_szz_payload(history: dict, previous_szz: dict | None) -> dict:
    base = dict(previous_szz) if isinstance(previous_szz, dict) else {}
    warnings = [str(w) for w in (base.get("warnings") or []) if str(w)]
    return {
        **base,
        "status": "computing",
        "warning": "",
        "fix_regex": str(history.get("fix_regex") or base.get("fix_regex") or gitmeta.DEFAULT_FIX_REGEX),
        "requested_fix_regex": _source_requested_fix_regex(history),
        "fix_commits": int(base.get("fix_commits", 0) or 0),
        "blamed_lines": int(base.get("blamed_lines", 0) or 0),
        "attributions": list(base.get("attributions") or []),
        "commit_attributions": dict(base.get("commit_attributions") or {}),
        "workers": gitmeta.szz_worker_count(),
        "cached_commits": int(base.get("cached_commits", 0) or 0),
        "blamed_commits": 0,
        "timings_ms": {"total": 0.0, "blame": 0.0},
        "warnings": _merge_warnings(_payload_warnings(history), warnings),
    }


def unavailable_szz_payload(history: dict | None, warning: str) -> dict:
    return {
        "status": "unavailable",
        "warning": warning,
        "fix_regex": str((history or {}).get("fix_regex") or gitmeta.DEFAULT_FIX_REGEX),
        "requested_fix_regex": _source_requested_fix_regex(history),
        "max_commits": (history or {}).get("max_commits"),
        "fix_commits": 0,
        "blamed_lines": 0,
        "attributions": [],
        "commit_attributions": {},
        "workers": gitmeta.szz_worker_count(),
        "cached_commits": 0,
        "blamed_commits": 0,
        "timings_ms": {"total": 0.0, "blame": 0.0},
        "warnings": _merge_warnings(_payload_warnings(history), [warning] if warning else []),
    }


def szz_summary(
    szz: dict,
    *,
    commits: list[dict] | None = None,
    regex_cache: regexsafe.RegexRequestCache | None = None,
) -> dict:
    fix_regex, regex_warnings = _safe_fix_regex(str(szz.get("fix_regex") or gitmeta.DEFAULT_FIX_REGEX))
    fix_commits = int(szz.get("fix_commits", 0) or 0)
    if not fix_commits and commits:
        fix_commits = fix_commit_count(commits, fix_regex, regex_cache=regex_cache)
    attributions = szz.get("attributions") or []
    warnings = [str(w) for w in (szz.get("warnings") or []) if str(w)]
    warnings = [*regex_warnings, *warnings]
    return {
        "status": str(szz.get("status") or "unavailable"),
        "workers": int(szz.get("workers", gitmeta.szz_worker_count()) or gitmeta.szz_worker_count()),
        "cached_commits": int(szz.get("cached_commits", 0) or 0),
        "blamed_commits": int(szz.get("blamed_commits", 0) or 0),
        "fix_commits": fix_commits,
        "blamed_lines": int(szz.get("blamed_lines", 0) or 0),
        "attributions": len(attributions),
        "timings_ms": dict(szz.get("timings_ms") or {}),
        "warnings": warnings[:50],
    }


def szz_for_source(
    *,
    logical_project_id: str,
    source: dict,
    history: dict | None,
    commits: list[dict],
    regex_cache: regexsafe.RegexRequestCache | None = None,
) -> dict:
    fix_regex, _regex_warnings = _safe_fix_regex(
        str(source.get("fix_regex") or (history or {}).get("fix_regex") or gitmeta.DEFAULT_FIX_REGEX)
    )
    max_commits = _source_max_commits(source) if isinstance(source, dict) else _source_max_commits(history)
    fingerprint = _fingerprint_from_payload(source)
    sidecar = gitstore.load_szz(logical_project_id)
    if _shared_szz_ready(
        sidecar,
        fingerprint=fingerprint,
        fix_regex=fix_regex,
        max_commits=max_commits,
    ):
        return sidecar or {}
    if _shared_szz_same_source(
        sidecar,
        fingerprint=fingerprint,
        fix_regex=fix_regex,
        max_commits=max_commits,
    ):
        return sidecar or {}
    if str(source.get("status") or "") == "unavailable":
        return unavailable_szz_payload(history, str(source.get("warning") or "git unavailable"))
    return {
        "status": "computing",
        "warning": "",
        "fix_regex": fix_regex,
        "requested_fix_regex": _source_requested_fix_regex(source),
        "max_commits": max_commits,
        "fix_commits": fix_commit_count(commits, fix_regex, regex_cache=regex_cache),
        "blamed_lines": 0,
        "attributions": [],
        "commit_attributions": {},
        "workers": gitmeta.szz_worker_count(),
        "cached_commits": 0,
        "blamed_commits": 0,
        "timings_ms": {"total": 0.0, "blame": 0.0},
        "warnings": [],
    }


def _run_shared_szz_task(
    *,
    root: Path,
    logical_project_id: str,
    history: dict,
    previous_szz: dict | None,
) -> None:
    commits = [c for c in (history.get("commits") or []) if isinstance(c, dict)]
    fix_regex = str(history.get("fix_regex") or gitmeta.DEFAULT_FIX_REGEX)
    # `history_for_shared_store` already resolved this exact regex over these
    # exact commit messages once; reuse that identity instead of asking the
    # SZZ pass to re-execute the (possibly unsafe) regex a second time.
    fix_commit_ids = history.get("fix_commit_ids")
    fix_commits = (
        gitmeta.fix_commits_from_ids(commits, fix_commit_ids)
        if isinstance(fix_commit_ids, list)
        else None
    )
    analysis_root = Path(str(history.get("git_common_dir") or root))

    def write_progress(payload: dict) -> None:
        _write_shared_szz_if_current(
            logical_project_id=logical_project_id,
            expected_history=history,
            payload=payload,
        )

    try:
        szz = gitmeta.szz_attributions_with_status(
            analysis_root,
            commits,
            fix_regex=fix_regex,
            fix_commits=fix_commits,
            previous=previous_szz,
            progress=write_progress,
        )
    except Exception as exc:
        szz = unavailable_szz_payload(history, f"szz unavailable: {exc}")
    _write_shared_szz_if_current(
        logical_project_id=logical_project_id,
        expected_history=history,
        payload=szz,
    )


def _start_shared_szz_task(
    *,
    root: Path,
    logical_project_id: str,
    git_history: dict,
    previous_szz: dict | None,
) -> None:
    history_fp = _fingerprint_from_payload(git_history)
    history_fix_regex = _source_fix_regex(git_history)
    history_max_commits = _source_max_commits(git_history)
    key = _szz_task_key(logical_project_id, history_fp, history_fix_regex, history_max_commits)

    def runner() -> None:
        try:
            _run_shared_szz_task(
                root=root,
                logical_project_id=logical_project_id,
                history=git_history,
                previous_szz=previous_szz,
            )
        finally:
            with _SZZ_TASKS_LOCK:
                _SZZ_TASKS.pop(key, None)

    suffix = logical_project_id[:32] or "project"
    thread = threading.Thread(target=runner, name=f"engram-szz-{suffix}", daemon=True)
    with _SZZ_TASKS_LOCK:
        current = _SZZ_TASKS.get(key)
        if current is not None and current.is_alive():
            return
        _SZZ_TASKS[key] = thread
        thread.start()


def _stale_shared_history(previous: dict, *, warning: str) -> dict:
    """Downgrade a previously-``ready`` shared history to ``status="stale"``.

    Used when the *current* repo fingerprint could not be computed (e.g. a
    transient ``git`` timeout or lock contention). The commits/fingerprint/
    head_commit already captured for the last known-good snapshot are still
    the best data available and must not be discarded just because this one
    probe failed -- only a fully successful new ``ready`` snapshot may ever
    replace it (see ``ensure_shared_git_analytics``). This mirrors the same
    "stale" degrade `analytics_source` already applies when a freshness
    re-check fails after a fingerprint mismatch.
    """
    warnings = _merge_warnings(_payload_warnings(previous), [warning] if warning else [])
    history = dict(previous)
    history.update(
        {
            "status": "stale",
            "warning": "; ".join(warnings[:5]),
            "warnings": warnings,
        }
    )
    return history


def _unavailable_shared_history(
    *,
    logical_project_id: str,
    checkout_kind: str,
    fingerprint: dict,
    fix_regex: str,
    requested_fix_regex: str,
    warnings: list[str] | None,
    git_max_commits: int | None,
) -> dict:
    warning_list = _merge_warnings(warnings or [], [str(fingerprint.get("warning") or "git unavailable")])
    return {
        "schema_version": gitstore.SCHEMA_VERSION,
        "logical_project_id": logical_project_id,
        "checkout_kind": checkout_kind,
        "status": "unavailable",
        "warning": "; ".join(warning_list[:5]),
        "max_commits": git_max_commits,
        "head_commit": str(fingerprint.get("tip_commit") or ""),
        "git_common_dir": str(fingerprint.get("common_dir") or ""),
        "fingerprint": {
            "algorithm": str(fingerprint.get("algorithm") or "git-refs-sha256-v1"),
            "hash": str(fingerprint.get("hash") or ""),
            "refs": int(fingerprint.get("refs", 0) or 0),
            "max_commit_ts": int(fingerprint.get("max_commit_ts", 0) or 0),
            "tip_commit": str(fingerprint.get("tip_commit") or ""),
        },
        "fix_regex": fix_regex,
        "requested_fix_regex": requested_fix_regex,
        "commits": [],
        "warnings": warning_list,
    }


def ensure_shared_git_analytics(
    *,
    root: Path,
    logical_project_id: str,
    checkout_kind: str,
    enabled: bool,
    fix_regex: str | None,
    git_max_commits: int | None = None,
) -> dict:
    requested_fix_regex = _requested_fix_regex(fix_regex)
    fix_regex_value, regex_warnings = _safe_fix_regex(fix_regex)
    result = {
        "fix_regex": fix_regex_value,
        "requested_fix_regex": requested_fix_regex,
        "warnings": regex_warnings,
    }
    if not enabled or not logical_project_id:
        return result
    git_max_commits = coerce_git_max_commits(git_max_commits)
    fingerprint = gitmeta.repo_ref_fingerprint(root)
    schedule_history: dict | None = None
    schedule_previous_szz: dict | None = None
    history_result: dict | None = None
    with paths.git_analytics_lock(logical_project_id):
        current_history = gitstore.load_history(logical_project_id)
        current_szz = gitstore.load_szz(logical_project_id)
        if fingerprint.get("status") != "ready":
            fp_warning = str(fingerprint.get("warning") or "git unavailable")
            if isinstance(current_history, dict) and current_history.get("status") == "ready":
                # A transient failure to compute the *current* fingerprint must
                # not throw away a previously-ready history/SZZ payload -- keep
                # serving it, just flagged stale with this failure attached.
                # The SZZ sidecar is left untouched: its fingerprint/fix_regex
                # still match the (unchanged) preserved history, so it stays
                # valid data for that snapshot instead of being blanked out.
                history = _stale_shared_history(current_history, warning=fp_warning)
                gitstore.save_history(logical_project_id, history)
                return {
                    "fix_regex": _source_fix_regex(history),
                    "requested_fix_regex": _source_requested_fix_regex(history),
                    "warnings": _payload_warnings(history),
                }
            # Nothing usable was cached yet -- the original "nothing to lose"
            # behavior: persist the unavailable marker.
            history = _unavailable_shared_history(
                logical_project_id=logical_project_id,
                checkout_kind=checkout_kind,
                fingerprint=fingerprint,
                fix_regex=fix_regex_value,
                requested_fix_regex=requested_fix_regex,
                warnings=regex_warnings,
                git_max_commits=git_max_commits,
            )
            gitstore.save_history(logical_project_id, history)
            gitstore.save_szz(
                logical_project_id,
                _wrap_shared_szz(
                    logical_project_id=logical_project_id,
                    history=history,
                    szz=unavailable_szz_payload(history, history.get("warning") or "git unavailable"),
                ),
            )
            return {
                "fix_regex": _source_fix_regex(history),
                "requested_fix_regex": _source_requested_fix_regex(history),
                "warnings": _payload_warnings(history),
            }

        current_fp_hash = str(fingerprint.get("hash") or "")
        current_requested_matches = (
            isinstance(current_history, dict)
            and (
                (
                    "requested_fix_regex" in current_history
                    and _source_requested_fix_regex(current_history) == requested_fix_regex
                )
                or (
                    "requested_fix_regex" not in current_history
                    and str(current_history.get("fix_regex") or gitmeta.DEFAULT_FIX_REGEX) == fix_regex_value
                )
            )
        )
        current_history_is_fresh = (
            isinstance(current_history, dict)
            and current_history.get("status") == "ready"
            and _fingerprint_hash(current_history) == current_fp_hash
            and coerce_git_max_commits(current_history.get("max_commits")) == git_max_commits
            and current_requested_matches
        )
        if current_history_is_fresh:
            history = current_history
        else:
            history = gitmeta.history_for_shared_store(
                root,
                logical_project_id=logical_project_id,
                checkout_kind=checkout_kind,
                fingerprint=fingerprint,
                fix_regex=fix_regex,
                max_commits=git_max_commits,
                timeout_sec=gitmeta.git_index_timeout_seconds(),
            )
            gitstore.save_history(logical_project_id, history)
        history_result = {
            "fix_regex": _source_fix_regex(history),
            "requested_fix_regex": _source_requested_fix_regex(history),
            "warnings": _payload_warnings(history),
        }

        history_fp = _fingerprint_from_payload(history)
        history_fix_regex = _source_fix_regex(history)
        history_max_commits = _source_max_commits(history)
        if history.get("status") != "ready" or not history_fp.get("hash"):
            gitstore.save_szz(
                logical_project_id,
                _wrap_shared_szz(
                    logical_project_id=logical_project_id,
                    history=history,
                    szz=unavailable_szz_payload(
                        history,
                        str(history.get("warning") or "git history unavailable"),
                    ),
                ),
            )
            return history_result

        if _shared_szz_ready(
            current_szz,
            fingerprint=history_fp,
            fix_regex=history_fix_regex,
            max_commits=history_max_commits,
        ):
            wrapped_szz = _wrap_shared_szz(
                logical_project_id=logical_project_id,
                history=history,
                szz=current_szz or {},
            )
            if wrapped_szz != current_szz:
                gitstore.save_szz(logical_project_id, wrapped_szz)
            return history_result
        if (
            _shared_szz_same_source(
                current_szz,
                fingerprint=history_fp,
                fix_regex=history_fix_regex,
                max_commits=history_max_commits,
            )
            and _szz_task_running(
                logical_project_id,
                history_fp,
                history_fix_regex,
                history_max_commits,
            )
        ):
            return history_result

        computing = computing_szz_payload(history, current_szz)
        # Reuse the fix-commit identity `history_for_shared_store` already
        # resolved instead of re-running the (possibly unsafe) regex here —
        # a second, independent resolution could disagree with the first.
        fix_commit_ids = history.get("fix_commit_ids")
        if isinstance(fix_commit_ids, list):
            computing["fix_commits"] = len(fix_commit_ids)
        else:
            computing["fix_commits"] = fix_commit_count(
                [c for c in (history.get("commits") or []) if isinstance(c, dict)],
                str(computing.get("fix_regex") or gitmeta.DEFAULT_FIX_REGEX),
            )
        gitstore.save_szz(
            logical_project_id,
            _wrap_shared_szz(
                logical_project_id=logical_project_id,
                history=history,
                szz=computing,
            ),
        )
        schedule_history = history
        schedule_previous_szz = current_szz

    if schedule_history is not None:
        _start_shared_szz_task(
            root=root,
            logical_project_id=logical_project_id,
            git_history=schedule_history,
            previous_szz=schedule_previous_szz,
        )
    return history_result or result


def wait_for_szz_tasks(timeout: float | None = None) -> None:
    """Wait for currently scheduled background SZZ tasks."""

    deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
    while True:
        with _SZZ_TASKS_LOCK:
            tasks = [task for task in _SZZ_TASKS.values() if task.is_alive()]
        if not tasks:
            return
        for task in tasks:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining is not None and remaining <= 0:
                return
            task.join(remaining)


def _cache_satisfies_limit(cached_limit: int | None, requested_limit: int | None) -> bool:
    if cached_limit is None:
        return True
    return requested_limit is not None and cached_limit >= requested_limit


def analytics_source(
    root: Path,
    *,
    logical_project_id: str,
    checkout_kind: str,
    git_max_commits: int | None,
) -> dict:
    t0 = time.perf_counter()
    git_max_commits = coerce_git_max_commits(git_max_commits)
    history = gitstore.load_history(logical_project_id)
    cached_ready = isinstance(history, dict) and history.get("status") == "ready"
    cached_limit = coerce_git_max_commits(history.get("max_commits")) if isinstance(history, dict) else None
    cached_commits_all = [c for c in ((history or {}).get("commits") or []) if isinstance(c, dict)]
    cached_commits = _limit_commits(cached_commits_all, git_max_commits)
    cache_head = str((history or {}).get("head_commit") or "") if isinstance(history, dict) else ""
    cache_fp = _fingerprint_from_payload(history)
    current_fp = gitmeta.repo_ref_fingerprint(root)
    current_head = str(current_fp.get("tip_commit") or "")
    fix_regex, regex_warnings = _safe_fix_regex(
        str((history or {}).get("fix_regex") or gitmeta.DEFAULT_FIX_REGEX)
    )
    history_warnings = _payload_warnings(history)
    base = {
        "commits": [],
        "status": "unavailable",
        "warning": "",
        "cache_head": cache_head,
        "current_head": current_head,
        "cache_fingerprint": str(cache_fp.get("hash") or ""),
        "current_fingerprint": str(current_fp.get("hash") or ""),
        "fingerprint": dict(current_fp) if current_fp.get("status") == "ready" else dict(cache_fp),
        "max_commits": git_max_commits,
        "fix_regex": fix_regex,
        "requested_fix_regex": _source_requested_fix_regex(history),
        "freshened_commits": 0,
        "source_counts": {
            "cached_commits": len(cached_commits_all) if isinstance(history, dict) else 0,
            "scanned_commits": 0,
        },
        "log_ms": 0.0,
    }
    if current_fp.get("status") != "ready":
        warning = (
            str(history.get("warning") or "git head unavailable")
            if isinstance(history, dict)
            else str(current_fp.get("warning") or "git refs unavailable")
        )
        if cached_ready:
            # The current fingerprint probe failed (e.g. a transient `git`
            # timeout), but there IS a previously-ready cached history --
            # serving its commits as "stale" beats throwing away known-good
            # data and reporting nothing just because *this* freshness check
            # couldn't run. Mirrors the freshen-failure branch below.
            base.update(
                {
                    "commits": cached_commits,
                    "status": "stale",
                    "warning": warning,
                    "source_counts": {
                        "cached_commits": len(cached_commits_all),
                        "scanned_commits": 0,
                    },
                    "log_ms": _ms_since(t0),
                }
            )
            return base
        base["warning"] = warning
        base["log_ms"] = _ms_since(t0)
        return base

    if cached_ready:
        if (
            str(cache_fp.get("hash") or "") == str(current_fp.get("hash") or "")
            and _cache_satisfies_limit(cached_limit, git_max_commits)
        ):
            base.update(
                {
                    "commits": cached_commits,
                    "status": "ready",
                    "warning": "; ".join(_merge_warnings(regex_warnings, history_warnings)),
                    "source_counts": {"cached_commits": len(cached_commits_all)},
                    "log_ms": _ms_since(t0),
                }
            )
            return base
        fresh = gitmeta.history_for_shared_store(
            root,
            logical_project_id=logical_project_id,
            checkout_kind=checkout_kind,
            fingerprint=current_fp,
            fix_regex=fix_regex,
            max_commits=git_max_commits,
            timeout_sec=gitmeta.GIT_STALENESS_TIMEOUT_SEC,
        )
        if fresh.get("status") != "ready":
            base.update(
                {
                    "commits": cached_commits,
                    "status": "stale",
                    "warning": str(fresh.get("warning") or "git log --all unavailable"),
                    "source_counts": {
                        "cached_commits": len(cached_commits_all),
                        "scanned_commits": 0,
                    },
                    "log_ms": _ms_since(t0),
                }
            )
            return base
        commits = [c for c in (fresh.get("commits") or []) if isinstance(c, dict)]
        cached_ids = {str(c.get("commit") or "") for c in cached_commits_all}
        freshened = sum(1 for c in commits if str(c.get("commit") or "") not in cached_ids)
        fresh_warnings = [str(w) for w in (fresh.get("warnings") or []) if str(w)]
        warning = "cached git fingerprint differs from current refs"
        if fresh_warnings:
            warning = warning + "; " + "; ".join(fresh_warnings)
        base.update(
            {
                "commits": commits,
                "status": "freshened",
                "warning": warning,
                "freshened_commits": freshened,
                "source_counts": {
                    "cached_commits": len(cached_commits_all),
                    "scanned_commits": len(commits),
                },
                "fingerprint": dict(_fingerprint_from_payload(fresh)),
                "fix_regex": _source_fix_regex(fresh),
                "requested_fix_regex": _source_requested_fix_regex(fresh),
                "current_head": str(fresh.get("head_commit") or current_head),
                "log_ms": _ms_since(t0),
            }
        )
        return base

    live = gitmeta.history_for_shared_store(
        root,
        logical_project_id=logical_project_id,
        checkout_kind=checkout_kind,
        fingerprint=current_fp,
        fix_regex=fix_regex,
        max_commits=git_max_commits,
        timeout_sec=gitmeta.GIT_STALENESS_TIMEOUT_SEC,
    )
    commits = [c for c in (live.get("commits") or []) if isinstance(c, dict)]
    if live.get("status") != "ready":
        base["warning"] = str(live.get("warning") or "git log --all unavailable")
        base["log_ms"] = _ms_since(t0)
        return base
    base.update(
        {
            "commits": commits,
            "status": "uncached",
            "warning": str(live.get("warning") or ""),
            "source_counts": {"scanned_commits": len(commits)},
            "fingerprint": dict(_fingerprint_from_payload(live)),
            "fix_regex": _source_fix_regex(live),
            "requested_fix_regex": _source_requested_fix_regex(live),
            "current_head": str(live.get("head_commit") or current_head),
            "log_ms": _ms_since(t0),
        }
    )
    return base
