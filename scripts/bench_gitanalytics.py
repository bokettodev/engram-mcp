from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from engram_mcp import catalog, gitmeta, manifest, paths, pipeline


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure PR-3 git analytics baseline cost.")
    parser.add_argument("project_path", help="indexed project root")
    parser.add_argument("--git-max-commits", type=int, default=1000)
    parser.add_argument("--recent-days", type=int, default=90)
    args = parser.parse_args()

    root = Path(args.project_path).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    commits, log_ms = _timed(lambda: gitmeta.commit_log(root, max_commits=args.git_max_commits))
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
    has_history = bool(isinstance(sidecar, dict) and isinstance(sidecar.get("git_history"), dict))
    base_with_history_ms = None
    if has_history:
        _base_with_history, base_with_history_ms = _timed(lambda: pipeline.project_map(root, include_git=False))

    out = {
        "project_path": str(root),
        "git_max_commits": args.git_max_commits,
        "commit_log": {
            "ms": log_ms,
            "commits": len(commits),
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
        },
        "project_map_git_ticket": {
            "wall_ms": ticket_ms,
            "timings_ms": (ticket_map.get("git_analytics") or {}).get("timings_ms"),
            "output_bytes": _size_bytes(ticket_map),
            "status": (ticket_map.get("git_analytics") or {}).get("status"),
            "analyzed_changes": (ticket_map.get("git_analytics") or {}).get("analyzed_changes"),
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
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
