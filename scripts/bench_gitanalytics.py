from __future__ import annotations

import argparse
import copy
import json
import threading
import time
from pathlib import Path

from engram_mcp import catalog, gitanalytics, gitmeta, gitstore, manifest, paths, pipeline


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


def _freshen_probe(root: Path, m: manifest.ProjectManifest | None) -> dict:
    if m is None:
        return {"available": False, "reason": "manifest unavailable"}
    source, ms = _timed(
        lambda: pipeline._analytics_source(
            root,
            logical_project_id=m.logical_project_id,
            checkout_kind=m.checkout_kind,
            git_max_commits=None,
        )
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


def _load_shared_store(root: Path) -> tuple[manifest.ProjectManifest | None, dict | None, float, dict | None, float]:
    pdir = paths.project_dir(root, create=False)
    m = manifest.load_project(pdir)
    if m is None:
        return None, None, 0.0, None, 0.0
    history, history_ms = _timed(lambda: gitstore.load_history(m.logical_project_id))
    szz, szz_ms = _timed(lambda: gitstore.load_szz(m.logical_project_id))
    return m, history, history_ms, szz, szz_ms


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
    parser.add_argument(
        "--second-project-path",
        default=None,
        help="optional second checkout/worktree to verify shared-store reuse",
    )
    args = parser.parse_args()

    root = Path(args.project_path).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    fingerprint = gitmeta.repo_ref_fingerprint(root)
    common_dir = fingerprint.get("common_dir") or root
    commits, log_ms = _timed(
        lambda: gitmeta.commit_log_with_status(common_dir, all_refs=True, git_dir=bool(fingerprint.get("common_dir"))).get(
            "commits",
            [],
        )
    )
    pdir = paths.project_dir(root, create=False)
    manifest_data = manifest.load_project(pdir)
    logical_project_id = manifest_data.logical_project_id if manifest_data is not None else ""
    checkout_kind = manifest_data.checkout_kind if manifest_data is not None else ""
    history, history_ms = _timed(
        lambda: gitmeta.history_for_shared_store(
            root,
            logical_project_id=logical_project_id,
            checkout_kind=checkout_kind,
            fingerprint=fingerprint,
        )
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
    store_manifest, shared_history, shared_history_ms, shared_szz, shared_szz_ms = _load_shared_store(root)
    has_history = isinstance(shared_history, dict)
    base_with_history_ms = None
    if has_history:
        _base_with_history, base_with_history_ms = _timed(lambda: pipeline.project_map(root, include_git=False))

    reuse_probe = None
    if args.second_project_path:
        second = Path(args.second_project_path).expanduser().resolve()
        if not second.is_dir():
            raise SystemExit(f"not a directory: {second}")
        second_manifest = manifest.load_project(paths.project_dir(second, create=False))
        first_ensure, first_ensure_ms = _timed(
            lambda: pipeline._ensure_shared_git_analytics(
                root=root,
                logical_project_id=logical_project_id,
                checkout_kind=checkout_kind,
                enabled=True,
                fix_regex=None,
            )
        )
        pipeline.wait_for_szz_tasks(timeout=None)
        second_logical = second_manifest.logical_project_id if second_manifest is not None else ""
        second_kind = second_manifest.checkout_kind if second_manifest is not None else ""
        second_ensure, second_ensure_ms = _timed(
            lambda: pipeline._ensure_shared_git_analytics(
                root=second,
                logical_project_id=second_logical,
                checkout_kind=second_kind,
                enabled=True,
                fix_regex=None,
            )
        )
        pipeline.wait_for_szz_tasks(timeout=None)
        first_szz = gitstore.load_szz(logical_project_id) if logical_project_id else None
        second_szz = gitstore.load_szz(second_logical) if second_logical else None
        reuse_probe = {
            "second_project_path": str(second),
            "same_logical_project_id": bool(logical_project_id and logical_project_id == second_logical),
            "first_ensure_ms": first_ensure_ms,
            "second_ensure_ms": second_ensure_ms,
            "first_ensure_return": first_ensure,
            "second_ensure_return": second_ensure,
            "first_szz_blamed_commits": (first_szz or {}).get("blamed_commits"),
            "second_szz_blamed_commits": (second_szz or {}).get("blamed_commits"),
            "second_reused_without_reblame": bool(
                logical_project_id
                and logical_project_id == second_logical
                and second_ensure_ms < max(100.0, first_ensure_ms * 0.25)
            ),
        }

    out = {
        "project_path": str(root),
        "git_max_commits": None,
        "szz_workers_chosen": gitmeta.szz_worker_count(),
        "shared_store": {
            "logical_project_id": logical_project_id,
            "checkout_kind": checkout_kind,
            "history_path": str(gitstore.history_path(logical_project_id, create=False))
            if logical_project_id
            else "",
            "szz_path": str(gitstore.szz_path(logical_project_id, create=False))
            if logical_project_id
            else "",
            "reuse_probe": reuse_probe,
        },
        "commit_log": {
            "ms": log_ms,
            "commits": len(commits),
        },
        "index_time_analytics_build": {
            "git_history_shared_ms": history_ms,
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
        "shared_git_history": {
            "present": has_history,
            "parse_ms": shared_history_ms if has_history else None,
            "status": shared_history.get("status") if isinstance(shared_history, dict) else None,
            "fingerprint": (shared_history.get("fingerprint") or {}).get("hash")
            if isinstance(shared_history, dict)
            else "",
            "commits": len((shared_history or {}).get("commits") or []),
            "base_map_ms_with_history_present": base_with_history_ms,
        },
        "shared_szz": {
            "present": isinstance(shared_szz, dict),
            "parse_ms": shared_szz_ms if isinstance(shared_szz, dict) else None,
            "status": shared_szz.get("status") if isinstance(shared_szz, dict) else None,
            "workers": shared_szz.get("workers") if isinstance(shared_szz, dict) else None,
            "cached_commits": shared_szz.get("cached_commits") if isinstance(shared_szz, dict) else None,
            "blamed_commits": shared_szz.get("blamed_commits") if isinstance(shared_szz, dict) else None,
            "attributions": len(shared_szz.get("attributions") or [])
            if isinstance(shared_szz, dict)
            else 0,
        },
        "sidecar_size_delta": _sidecar_size_delta(sidecar),
        "read_time_freshen_probe": _freshen_probe(root, store_manifest),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
