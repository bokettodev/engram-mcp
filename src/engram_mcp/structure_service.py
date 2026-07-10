"""Body-free project map assembly and git-analytics annotation."""

from __future__ import annotations

import time
from pathlib import Path

from engram_mcp import (
    catalog,
    gitanalytics,
    gitmeta,
    gitorchestration,
    regexsafe,
)
from engram_mcp.index_repository import load_project_catalog


MAX_GIT_MAX_FILES_PER_CHANGE = 500
MAX_GIT_COCHANGE_LIMIT = 100
MAX_GIT_HOTSPOTS_LIMIT = 500

_SZZ_DEFECT_KEYS = {
    "defect_introducing_commits",
    "defect_introducing_lines",
    "defect_hotspot_score",
}


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


def _unavailable_git_analytics(
    *,
    group_by: str,
    source_counts: dict | None = None,
    log_ms: float,
    total_start: float,
    warning: str | None = None,
    cache_head: str = "",
    current_head: str = "",
    freshened_commits: int = 0,
) -> dict:
    warnings = [warning] if warning else []
    return {
        "available": False,
        "status": "unavailable",
        "group_by": group_by,
        "analyzed_changes": 0,
        "skipped_changes": 0,
        "cache_head": cache_head,
        "current_head": current_head,
        "freshened_commits": freshened_commits,
        **(source_counts or {}),
        "timings_ms": {
            "log": log_ms,
            "group": 0.0,
            "cochange": 0.0,
            "churn": 0.0,
            "total": _ms_since(total_start),
        },
        "warnings": warnings,
        "szz": gitorchestration.szz_summary(
            gitorchestration.unavailable_szz_payload(None, warning or "git unavailable")
        ),
        "hotspots": [],
    }


def _strip_defect_signal(hotspot_result: dict) -> dict:
    files = hotspot_result.get("files")
    if isinstance(files, dict):
        for row in files.values():
            if isinstance(row, dict):
                for key in _SZZ_DEFECT_KEYS:
                    row.pop(key, None)
    for item in hotspot_result.get("hotspots") or []:
        if isinstance(item, dict):
            for key in _SZZ_DEFECT_KEYS:
                item.pop(key, None)
    return hotspot_result


def _file_git_payload(
    path: str,
    *,
    churn_by_file: dict[str, dict],
    cochanges_by_file: dict[str, list[dict]],
    hotspot_by_file: dict[str, dict],
    include_defects: bool,
) -> dict:
    churn_row = churn_by_file.get(path) or {}
    hotspot = hotspot_by_file.get(path) or {}
    raw_fix_density = churn_row.get("fix_density", 0.0)
    payload = {
        "changes": int(churn_row.get("changes", 0) or 0),
        "churn_lines": int(churn_row.get("churn_lines", 0) or 0),
        "last_touched_ts": int(churn_row.get("last_touched_ts", 0) or 0),
        "fix_density": (
            None
            if raw_fix_density is None
            else float(raw_fix_density or 0.0)
        ),
        "complexity": float(hotspot.get("complexity", 0.0) or 0.0),
        "indent_complexity": float(hotspot.get("indent_complexity", 0.0) or 0.0),
        "hotspot_quadrant": hotspot.get("hotspot_quadrant") or "low_churn_low_complexity",
        "cochanges": list(cochanges_by_file.get(path) or []),
    }
    if include_defects:
        payload.update(
            {
                "defect_introducing_commits": int(
                    hotspot.get("defect_introducing_commits", 0) or 0
                ),
                "defect_introducing_lines": int(hotspot.get("defect_introducing_lines", 0) or 0),
                "defect_hotspot_score": int(hotspot.get("defect_hotspot_score", 0) or 0),
            }
        )
    return payload


def _attach_git_analytics(
    out: dict,
    *,
    root: Path,
    logical_project_id: str,
    checkout_kind: str,
    data: dict,
    group_by: str,
    ticket_regex: str | None,
    window_hours: float,
    git_max_commits: int | None,
    recent_days: int,
    max_files_per_change: int,
    cochange_limit: int,
    hotspots_limit: int,
) -> dict:
    total_start = time.perf_counter()
    warnings: list[str] = []
    if max_files_per_change > MAX_GIT_MAX_FILES_PER_CHANGE:
        warnings.append(
            f"max_files_per_change clamped to server maximum {MAX_GIT_MAX_FILES_PER_CHANGE} "
            f"(requested {max_files_per_change})"
        )
        max_files_per_change = MAX_GIT_MAX_FILES_PER_CHANGE
    if cochange_limit > MAX_GIT_COCHANGE_LIMIT:
        warnings.append(
            f"cochange_limit clamped to server maximum {MAX_GIT_COCHANGE_LIMIT} "
            f"(requested {cochange_limit})"
        )
        cochange_limit = MAX_GIT_COCHANGE_LIMIT
    if hotspots_limit > MAX_GIT_HOTSPOTS_LIMIT:
        warnings.append(
            f"hotspots_limit clamped to server maximum {MAX_GIT_HOTSPOTS_LIMIT} "
            f"(requested {hotspots_limit})"
        )
        hotspots_limit = MAX_GIT_HOTSPOTS_LIMIT
    coerced_git_max_commits = gitorchestration.coerce_git_max_commits(git_max_commits)
    if (
        git_max_commits is not None
        and int(git_max_commits) > 0
        and coerced_git_max_commits != int(git_max_commits)
    ):
        warnings.append(
            f"git_max_commits clamped to server maximum {coerced_git_max_commits} "
            f"(requested {git_max_commits})"
        )
    regex_cache = regexsafe.RegexRequestCache()
    source = gitorchestration.analytics_source(
        root,
        logical_project_id=logical_project_id,
        checkout_kind=checkout_kind,
        git_max_commits=git_max_commits,
    )
    source_warning = str(source.get("warning") or "")
    if source.get("status") == "unavailable":
        payload = _unavailable_git_analytics(
            group_by=group_by,
            source_counts=source.get("source_counts") or {},
            log_ms=float(source.get("log_ms", 0.0) or 0.0),
            total_start=total_start,
            warning=source_warning,
            cache_head=str(source.get("cache_head") or ""),
            current_head=str(source.get("current_head") or ""),
            freshened_commits=int(source.get("freshened_commits", 0) or 0),
        )
        # Clamp warnings (recorded above, before the source turned out to be
        # unavailable) must still reach the caller instead of being dropped.
        payload["warnings"] = warnings + list(payload.get("warnings") or [])
        out["git_analytics"] = payload
        return out
    if source_warning:
        warnings.append(source_warning)
    commits = [c for c in (source.get("commits") or []) if isinstance(c, dict)]
    szz = gitorchestration.szz_for_source(
        logical_project_id=logical_project_id,
        source=source,
        history=None,
        commits=commits,
        regex_cache=regex_cache,
    )
    szz_status = str(szz.get("status") or "unavailable")
    szz_ready = szz_status == "ready"
    fix_regex = str(szz.get("fix_regex") or source.get("fix_regex") or gitmeta.DEFAULT_FIX_REGEX)

    t_group = time.perf_counter()
    grouped = gitanalytics.group_changes_result(
        commits,
        group_by=group_by,
        ticket_regex=ticket_regex,
        window_hours=window_hours,
        max_files_per_change=max_files_per_change,
        regex_cache=regex_cache,
    )
    change_sets = grouped["change_sets"]
    warnings.extend(str(w) for w in (grouped.get("regex_warnings") or []) if str(w))
    group_ms = _ms_since(t_group)

    t_cochange = time.perf_counter()
    cochanges_by_file = gitanalytics.cochange(change_sets, limit=cochange_limit)
    cochange_ms = _ms_since(t_cochange)

    t_churn = time.perf_counter()
    churn_result = gitanalytics.churn_result(
        change_sets,
        now_ts=int(time.time()),
        recent_days=recent_days,
        fix_regex=fix_regex,
        regex_cache=regex_cache,
        fix_density_ready=szz_ready,
    )
    churn_by_file = churn_result["files"]
    warnings.extend(str(w) for w in (churn_result.get("regex_warnings") or []) if str(w))
    defect_by_file = (
        gitanalytics.defect_introductions(szz.get("attributions") or []) if szz_ready else {}
    )
    hotspot_result = gitanalytics.hotspots(
        churn_by_file,
        data.get("files") or [],
        limit=hotspots_limit,
        defect_by_file=defect_by_file,
    )
    if not szz_ready:
        hotspot_result = _strip_defect_signal(hotspot_result)
    churn_ms = _ms_since(t_churn)
    hotspot_by_file = hotspot_result.get("files") or {}

    for row in out.get("files") or []:
        path = (row.get("path") or "").replace("\\", "/")
        row["git"] = _file_git_payload(
            path,
            churn_by_file=churn_by_file,
            cochanges_by_file=cochanges_by_file,
            hotspot_by_file=hotspot_by_file,
            include_defects=szz_ready,
        )

    out["git_analytics"] = {
        "available": True,
        "status": source.get("status") or "ready",
        "group_by": grouped.get("group_by") or group_by,
        "analyzed_changes": len(change_sets),
        "skipped_changes": int(grouped.get("skipped_changes", 0) or 0),
        "cache_head": str(source.get("cache_head") or ""),
        "current_head": str(source.get("current_head") or ""),
        "freshened_commits": int(source.get("freshened_commits", 0) or 0),
        "cache_fingerprint": str(source.get("cache_fingerprint") or ""),
        "current_fingerprint": str(source.get("current_fingerprint") or ""),
        **(source.get("source_counts") or {}),
        "szz": gitorchestration.szz_summary(szz, commits=commits, regex_cache=regex_cache),
        "timings_ms": {
            "log": float(source.get("log_ms", 0.0) or 0.0),
            "group": group_ms,
            "cochange": cochange_ms,
            "churn": churn_ms,
            "total": _ms_since(total_start),
        },
        "warnings": warnings,
        "hotspots": list(hotspot_result.get("hotspots") or []),
    }
    return out


def _disabled_git_analytics(*, group_by: str) -> dict:
    warning = (
        "git analytics is disabled for this project's index (indexed with "
        "git_analytics=false, e.g. `--no-git-analytics` or ENGRAM_GIT_ANALYTICS=0); "
        "re-index with git analytics enabled (`engram index --git-analytics <path>` "
        "or the MCP index_project git_analytics=true) to compute it"
    )
    return {
        "available": False,
        "status": "disabled",
        "group_by": group_by,
        "analyzed_changes": 0,
        "skipped_changes": 0,
        "cache_head": "",
        "current_head": "",
        "freshened_commits": 0,
        "timings_ms": {"log": 0.0, "group": 0.0, "cochange": 0.0, "churn": 0.0, "total": 0.0},
        "warnings": [warning],
        "szz": gitorchestration.szz_summary(
            gitorchestration.unavailable_szz_payload(None, warning)
        ),
        "hotspots": [],
    }


def project_map(
    root: str | Path,
    depth: int = 2,
    sort: str = "path",
    dirs_limit: int | None = 200,
    dirs_offset: int = 0,
    include_files: bool = False,
    files_limit: int | None = 50,
    files_offset: int = 0,
    include_symbols: bool = False,
    symbols_limit: int | None = 20,
    code_only: bool = False,
    languages: list[str] | None = None,
    chunk_roles: list[str] | None = None,
    kinds: list[str] | None = None,
    path_prefix: str | None = None,
    path_glob: str | None = None,
    symbol_kinds: list[str] | None = None,
    min_symbols: int = 0,
    non_empty: bool = True,
    include_git: bool | None = None,
    group_by: str = "commit",
    ticket_regex: str | None = None,
    window_hours: float = 2.0,
    git_max_commits: int | None = None,
    recent_days: int = 90,
    max_files_per_change: int = 50,
    cochange_limit: int = 5,
    hotspots_limit: int = 25,
) -> dict:
    """Return a body-free project map from the catalog sidecar.

    ``include_git`` omitted (``None``) defers to ``ENGRAM_GIT_ANALYTICS``
    (default: on). Regardless of ``include_git``, a project indexed with git
    analytics disabled (``git_analytics_enabled=False`` in its manifest, e.g.
    via ``--no-git-analytics`` or ``ENGRAM_GIT_ANALYTICS=0`` at index time)
    never performs a live request-path git walk here: it reports
    ``git_analytics.status == "disabled"`` instead. ``git_max_commits``
    omitted defers to the cap recorded on the project's manifest at index
    time rather than defaulting to unlimited history.
    """

    qi, data = load_project_catalog(root)
    out = catalog.project_map(
        data,
        depth=depth,
        sort=sort,
        dirs_limit=dirs_limit,
        dirs_offset=dirs_offset,
        include_files=include_files,
        files_limit=files_limit,
        files_offset=files_offset,
        include_symbols=include_symbols,
        symbols_limit=symbols_limit,
        code_only=code_only,
        languages=languages,
        chunk_roles=chunk_roles,
        kinds=kinds,
        path_prefix=path_prefix,
        path_glob=path_glob,
        symbol_kinds=symbol_kinds,
        min_symbols=min_symbols,
        non_empty=non_empty,
    )
    effective_include_git = gitmeta.git_analytics_default() if include_git is None else include_git
    if not effective_include_git:
        return out
    if not bool(getattr(qi.manifest, "git_analytics_enabled", True)):
        # The project was indexed with analytics disabled: never let an
        # `include_git=True` request (explicit or env-defaulted) trigger a
        # live request-path git walk. Report the recorded policy instead.
        out["git_analytics"] = _disabled_git_analytics(group_by=group_by)
        return out
    effective_git_max_commits = (
        git_max_commits if git_max_commits is not None else getattr(qi.manifest, "git_max_commits", None)
    )
    return _attach_git_analytics(
        out,
        root=qi.root,
        logical_project_id=qi.manifest.logical_project_id,
        checkout_kind=qi.manifest.checkout_kind,
        data=data,
        group_by=group_by,
        ticket_regex=ticket_regex,
        window_hours=window_hours,
        git_max_commits=effective_git_max_commits,
        recent_days=recent_days,
        max_files_per_change=max_files_per_change,
        cochange_limit=cochange_limit,
        hotspots_limit=hotspots_limit,
    )
