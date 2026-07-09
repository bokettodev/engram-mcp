from __future__ import annotations

import argparse
import copy
import json
import threading
import time
from pathlib import Path

from engram_mcp import catalog, gitanalytics, gitmeta, manifest, paths, pipeline


def _size_bytes(payload: dict) -> int:
    return len(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def _timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, round((time.perf_counter() - start) * 1000.0, 3)


def _load_catalog_with_timing(root: Path) -> tuple[dict | None, float]:
    pdir = paths.project_dir(root, create=False)
    m = manifest.load_project(pdir)
    if m is None:
        return None, 0.0
    return _timed(lambda: catalog.load_catalog(pdir, m.generation))


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None
    except OSError:
        return None


def _indentation_scan(root: Path, sidecar: dict | None) -> dict:
    if not isinstance(sidecar, dict):
        return {"files": 0, "ms": 0.0}

    def run() -> dict:
        files = 0
        total = 0.0
        for item in sidecar.get("files") or []:
            rel = str(item.get("path") or "")
            if not rel:
                continue
            text = _read_text(root / rel)
            if text is None:
                continue
            files += 1
            total += gitanalytics.indentation_complexity(text)
        return {"files": files, "total_indent_complexity": round(total, 3)}

    result, ms = _timed(run)
    return result | {"ms": ms}


def _sidecar_size_delta(sidecar: dict | None) -> dict:
    if not isinstance(sidecar, dict):
        return {"with_analytics_bytes": 0, "without_analytics_bytes": 0, "delta_bytes": 0}
    stripped = copy.deepcopy(sidecar)
    stripped.pop("git_history", None)
    for item in stripped.get("files") or []:
        if isinstance(item, dict):
            item.pop("indent_complexity", None)
    with_analytics = _size_bytes(sidecar)
    without_analytics = _size_bytes(stripped)
    return {
        "with_analytics_bytes": with_analytics,
        "without_analytics_bytes": without_analytics,
        "delta_bytes": with_analytics - without_analytics,
    }


def _freshen_probe(root: Path, sidecar: dict | None, max_commits: int | None) -> dict:
    if not isinstance(sidecar, dict):
        return {"available": False, "reason": "catalog unavailable"}
    current = gitmeta.head_commit(root)
    if not current:
        return {"available": False, "reason": "HEAD unavailable"}
    parent = gitmeta._git(root, "rev-parse", "HEAD~1")
    if not parent:
        return {"available": False, "reason": "HEAD~1 unavailable"}
    simulated = copy.deepcopy(sidecar)
    simulated["git_history"] = {
        "schema_version": 1,
        "status": "ready",
        "max_commits": max_commits,
        "head_commit": parent,
        "fix_regex": gitmeta.DEFAULT_FIX_REGEX,
        "commits": [],
    }
    source, ms = _timed(
        lambda: pipeline._analytics_source(root, simulated, git_max_commits=max_commits)
    )
    return {
        "available": True,
        "ms": ms,
        "status": source.get("status"),
        "cache_head": source.get("cache_head"),
        "current_head": source.get("current_head"),
        "freshened_commits": source.get("freshened_commits"),
        "warning": source.get("warning"),
    }


def _load_szz_sidecar(root: Path, sidecar: dict | None) -> tuple[dict | None, float]:
    if not isinstance(sidecar, dict):
        return None, 0.0
    pdir = paths.project_dir(root, create=False)
    generation = int(sidecar.get("generation", 0) or 0)
    return _timed(lambda: catalog.load_szz(pdir, generation))


def _first_pass_szz_with_search_probe(root: Path, commits: list[dict]) -> dict:
    result: dict = {}

    def run_szz() -> None:
        szz, ms = _timed(lambda: gitmeta.szz_attributions_with_status(root, commits))
        result["szz"] = szz
        result["ms"] = ms

    thread = threading.Thread(target=run_szz, name="bench-szz-first-pass")
    thread.start()
    try:
        _qi, searchable_ms = _timed(lambda: pipeline.load_query_index(root))
        searchable = True
        search_error = ""
    except Exception as exc:
        searchable_ms = 0.0
        searchable = False
        search_error = str(exc)
    szz_running_during_probe = thread.is_alive()
    thread.join()
    szz = result.get("szz") if isinstance(result.get("szz"), dict) else {}
    return {
        "szz": szz,
        "ms": float(result.get("ms", 0.0) or 0.0),
        "searchability_probe": {
            "searchable": searchable,
            "load_query_index_ms": searchable_ms,
            "szz_running_during_probe": szz_running_during_probe,
            "error": search_error,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure PR-3 git analytics baseline cost.")
    parser.add_argument("project_path", help="indexed project root")
    parser.add_argument("--git-max-commits", type=int, default=None)
    parser.add_argument("--recent-days", type=int, default=90)
    args = parser.parse_args()

    root = Path(args.project_path).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    commits, log_ms = _timed(lambda: gitmeta.commit_log(root, max_commits=args.git_max_commits))
    history, history_ms = _timed(
        lambda: gitmeta.history_for_catalog(root, max_commits=args.git_max_commits)
    )
    first_szz = _first_pass_szz_with_search_probe(root, commits)
    incremental_szz, incremental_szz_ms = _timed(
        lambda: gitmeta.szz_attributions_with_status(
            root,
            commits,
            previous=first_szz.get("szz") if isinstance(first_szz.get("szz"), dict) else None,
        )
    )
    base, base_ms = _timed(lambda: pipeline.project_map(root, include_git=False))

    commit_map, commit_ms = _timed(
        lambda: pipeline.project_map(
            root,
            include_files=True,
            include_git=True,
            group_by="commit",
            git_max_commits=args.git_max_commits,
            recent_days=args.recent_days,
        )
    )
    ticket_map, ticket_ms = _timed(
        lambda: pipeline.project_map(
            root,
            include_files=True,
            include_git=True,
            group_by="ticket",
            git_max_commits=args.git_max_commits,
            recent_days=args.recent_days,
        )
    )

    sidecar, parse_ms = _load_catalog_with_timing(root)
    szz_sidecar, szz_sidecar_ms = _load_szz_sidecar(root, sidecar)
    has_history = bool(isinstance(sidecar, dict) and isinstance(sidecar.get("git_history"), dict))
    base_with_history_ms = None
    if has_history:
        _base_with_history, base_with_history_ms = _timed(lambda: pipeline.project_map(root, include_git=False))

    out = {
        "project_path": str(root),
        "git_max_commits": args.git_max_commits,
        "szz_workers_chosen": gitmeta.szz_worker_count(),
        "commit_log": {
            "ms": log_ms,
            "commits": len(commits),
        },
        "index_time_analytics_build": {
            "git_history_cheap_ms": history_ms,
            "status": history.get("status"),
            "commits": len(history.get("commits") or []),
            "indentation_scan": _indentation_scan(root, sidecar),
        },
        "szz_measurement": {
            "first_pass_parallel_ms": first_szz.get("ms"),
            "first_pass_status": (first_szz.get("szz") or {}).get("status"),
            "first_pass_workers": (first_szz.get("szz") or {}).get("workers"),
            "first_pass_fix_commits": (first_szz.get("szz") or {}).get("fix_commits"),
            "first_pass_blamed_commits": (first_szz.get("szz") or {}).get("blamed_commits"),
            "first_pass_cached_commits": (first_szz.get("szz") or {}).get("cached_commits"),
            "first_pass_attributions": len((first_szz.get("szz") or {}).get("attributions") or []),
            "incremental_second_run_ms": incremental_szz_ms,
            "incremental_status": incremental_szz.get("status"),
            "incremental_blamed_commits": incremental_szz.get("blamed_commits"),
            "incremental_cached_commits": incremental_szz.get("cached_commits"),
            "searchability_probe": first_szz.get("searchability_probe"),
        },
        "project_map_base": {
            "ms": base_ms,
            "output_bytes": _size_bytes(base),
        },
        "project_map_git_commit": {
            "wall_ms": commit_ms,
            "timings_ms": (commit_map.get("git_analytics") or {}).get("timings_ms"),
            "output_bytes": _size_bytes(commit_map),
            "status": (commit_map.get("git_analytics") or {}).get("status"),
            "analyzed_changes": (commit_map.get("git_analytics") or {}).get("analyzed_changes"),
            "szz": (commit_map.get("git_analytics") or {}).get("szz"),
        },
        "project_map_git_ticket": {
            "wall_ms": ticket_ms,
            "timings_ms": (ticket_map.get("git_analytics") or {}).get("timings_ms"),
            "output_bytes": _size_bytes(ticket_map),
            "status": (ticket_map.get("git_analytics") or {}).get("status"),
            "analyzed_changes": (ticket_map.get("git_analytics") or {}).get("analyzed_changes"),
            "szz": (ticket_map.get("git_analytics") or {}).get("szz"),
        },
        "sidecar_git_history": {
            "present": has_history,
            "parse_ms": parse_ms if has_history else None,
            "status": (sidecar.get("git_history") or {}).get("status") if isinstance(sidecar, dict) else None,
            "commits": len((sidecar.get("git_history") or {}).get("commits") or [])
            if isinstance(sidecar, dict)
            else 0,
            "base_map_ms_with_history_present": base_with_history_ms,
        },
        "sidecar_szz": {
            "present": isinstance(szz_sidecar, dict),
            "parse_ms": szz_sidecar_ms if isinstance(szz_sidecar, dict) else None,
            "status": szz_sidecar.get("status") if isinstance(szz_sidecar, dict) else None,
            "workers": szz_sidecar.get("workers") if isinstance(szz_sidecar, dict) else None,
            "cached_commits": szz_sidecar.get("cached_commits") if isinstance(szz_sidecar, dict) else None,
            "blamed_commits": szz_sidecar.get("blamed_commits") if isinstance(szz_sidecar, dict) else None,
            "attributions": len(szz_sidecar.get("attributions") or [])
            if isinstance(szz_sidecar, dict)
            else 0,
        },
        "sidecar_size_delta": _sidecar_size_delta(sidecar),
        "read_time_freshen_probe": _freshen_probe(root, sidecar, args.git_max_commits),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
