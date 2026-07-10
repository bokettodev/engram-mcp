"""Deterministic VCS analytics over compact git history.

This module is intentionally pure: no git, no filesystem, no model calls. It
consumes compact records returned by ``gitmeta.commit_log_with_status`` or
stored in the shared git-analytics sidecar.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median
from typing import Iterable

from engram_mcp import regexsafe

DEFAULT_TICKET_REGEX = r"(?P<ticket>[A-Z][A-Z0-9]+-\d+|#[0-9]+)"
DEFAULT_FIX_REGEX = r"(?i)\b(fix(e[sd])?|bug|hotfix|patch|close[sd]?\s+#\d+)\b"
_COMMENTISH_PREFIXES = ("#", "//", "/*", "*", "*/", "--", "<!--")


def _norm_path(value: str | None) -> str:
    text = (value or "").replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text.strip("/")


def indentation_complexity(
    text: str | None,
    *,
    tab_width: int = 4,
    max_chars: int = 1_000_000,
) -> float:
    """Return Hindle-style indentation complexity for one text file.

    The metric is the sum of leading indentation depth over non-blank,
    non-comment-ish logical lines. One depth unit is ``tab_width`` columns;
    tabs advance to the next tab stop.
    """

    if not text or "\x00" in text or len(text) > max_chars:
        return 0.0
    tab_width = max(1, int(tab_width or 4))
    total = 0.0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if line.lstrip().startswith(_COMMENTISH_PREFIXES):
            continue
        columns = 0
        for ch in line:
            if ch == " ":
                columns += 1
            elif ch == "\t":
                columns += tab_width - (columns % tab_width)
            elif ch in "\r\f\v":
                columns += 1
            else:
                break
        total += columns / tab_width
    return round(total, 3)


def _is_merge(commit: dict) -> bool:
    parents = commit.get("parents") or []
    return isinstance(parents, list) and len(parents) > 1


def _commit_path_entries(commit: dict) -> list[dict]:
    if _is_merge(commit):
        return []
    out: list[dict] = []
    for item in commit.get("paths") or []:
        if not isinstance(item, dict):
            continue
        path = _norm_path(item.get("path"))
        if not path:
            continue
        try:
            added = max(0, int(item.get("added", 0) or 0))
        except (TypeError, ValueError):
            added = 0
        try:
            deleted = max(0, int(item.get("deleted", 0) or 0))
        except (TypeError, ValueError):
            deleted = 0
        entry = {
            "path": path,
            "status": str(item.get("status") or "M")[:1],
            "added": added,
            "deleted": deleted,
        }
        old_path = _norm_path(item.get("old_path"))
        if old_path and old_path != path:
            entry["old_path"] = old_path
        out.append(entry)
    return out


def _change_set(change_id: str, commits: list[dict]) -> dict | None:
    entries: list[dict] = []
    commit_ids: list[str] = []
    messages: list[str] = []
    authors: set[str] = set()
    ts_values: list[int] = []
    for commit in commits:
        commit_id = str(commit.get("commit") or "")
        if commit_id:
            commit_ids.append(commit_id)
        message = str(commit.get("message") or "")
        if message:
            messages.append(message)
        author = str(commit.get("author") or "")
        if author:
            authors.add(author)
        try:
            ts_values.append(int(commit.get("ts", 0) or 0))
        except (TypeError, ValueError):
            ts_values.append(0)
        for entry in _commit_path_entries(commit):
            item = dict(entry)
            item["commit"] = commit_id
            item["author"] = author
            entries.append(item)
    files = sorted({entry["path"] for entry in entries})
    if not files:
        return None
    return {
        "id": change_id,
        "commits": commit_ids,
        "commit_count": len(commit_ids),
        "ts": max(ts_values) if ts_values else 0,
        "authors": sorted(authors),
        "message": messages[0] if messages else "",
        "messages": messages,
        "paths": entries,
        "files": files,
    }


def _ticket_ids(
    commits: list[dict],
    ticket_regex: str | None,
    regex_cache: regexsafe.RegexRequestCache | None = None,
) -> tuple[list[str], list[str]]:
    messages = [str(commit.get("message") or "") for commit in commits]
    runner = regex_cache.extract_first_or_default if regex_cache is not None else regexsafe.extract_first_or_default
    _pattern, extracted, warnings = runner(
        ticket_regex,
        DEFAULT_TICKET_REGEX,
        messages,
        label="ticket_regex",
    )
    ids: list[str] = []
    for commit, ticket in zip(commits, extracted, strict=False):
        ids.append(str(ticket or commit.get("commit") or ""))
    return ids, warnings


def _append_filtered(
    change_sets: list[dict],
    skipped: Counter,
    change: dict | None,
    *,
    max_files_per_change: int,
) -> None:
    if change is None:
        skipped["empty"] += 1
        return
    if len(change.get("files") or []) > max_files_per_change:
        skipped["large"] += 1
        return
    change_sets.append(change)


def group_changes_result(
    commits: Iterable[dict],
    group_by: str = "commit",
    ticket_regex: str | None = None,
    window_hours: float = 2.0,
    max_files_per_change: int = 50,
    regex_cache: regexsafe.RegexRequestCache | None = None,
) -> dict:
    """Group commits into logical changes and report skipped/noise counts."""

    items = [c for c in commits if isinstance(c, dict)]
    mode = (group_by or "commit").lower()
    if mode == "pr":
        mode = "merge"
    if mode not in {"commit", "ticket", "merge", "window"}:
        mode = "commit"
    max_files = max(1, int(max_files_per_change or 50))
    change_sets: list[dict] = []
    skipped: Counter = Counter()
    skipped["merge"] = sum(1 for commit in items if _is_merge(commit))

    if mode == "commit":
        for commit in items:
            change_id = str(commit.get("commit") or "")
            _append_filtered(
                change_sets,
                skipped,
                _change_set(change_id, [commit]),
                max_files_per_change=max_files,
            )
    elif mode == "ticket":
        ticket_ids, regex_warnings = _ticket_ids(items, ticket_regex, regex_cache=regex_cache)
        grouped: dict[str, list[dict]] = {}
        order: list[str] = []
        for commit, key in zip(items, ticket_ids, strict=False):
            if not key:
                key = str(commit.get("commit") or f"commit-{len(order)}")
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(commit)
        for key in order:
            _append_filtered(
                change_sets,
                skipped,
                _change_set(key, grouped[key]),
                max_files_per_change=max_files,
            )
    elif mode == "merge":
        group: list[dict] = []
        newest = ""
        for commit in items:
            if _is_merge(commit):
                if group:
                    _append_filtered(
                        change_sets,
                        skipped,
                        _change_set(f"merge-window:{newest}", group),
                        max_files_per_change=max_files,
                    )
                    group = []
                    newest = ""
                continue
            if not group:
                newest = str(commit.get("commit") or "")
            group.append(commit)
        if group:
            _append_filtered(
                change_sets,
                skipped,
                _change_set(f"merge-window:{newest}", group),
                max_files_per_change=max_files,
            )
    else:
        window_sec = max(0.0, float(window_hours or 2.0)) * 3600.0
        ordered = sorted(items, key=lambda c: int(c.get("ts", 0) or 0))
        group: list[dict] = []
        current_author = ""
        current_end = 0
        group_seq = 0
        for commit in ordered:
            author = str(commit.get("author") or "")
            try:
                ts = int(commit.get("ts", 0) or 0)
            except (TypeError, ValueError):
                ts = 0
            starts_new = (
                not group
                or author != current_author
                or (window_sec > 0 and ts - current_end > window_sec)
            )
            if starts_new and group:
                group_seq += 1
                _append_filtered(
                    change_sets,
                    skipped,
                    _change_set(f"window:{group_seq}", group),
                    max_files_per_change=max_files,
                )
                group = []
            group.append(commit)
            current_author = author
            current_end = ts
        if group:
            group_seq += 1
            _append_filtered(
                change_sets,
                skipped,
                _change_set(f"window:{group_seq}", group),
                max_files_per_change=max_files,
            )

    return {
        "change_sets": change_sets,
        "skipped_changes": int(skipped["large"] + skipped["merge"]),
        "skipped_large_changes": int(skipped["large"]),
        "skipped_merge_commits": int(skipped["merge"]),
        "skipped_empty_changes": int(skipped["empty"]),
        "group_by": mode,
        "regex_warnings": regex_warnings if mode == "ticket" else [],
    }


def group_changes(
    commits: Iterable[dict],
    group_by: str = "commit",
    ticket_regex: str | None = None,
    window_hours: float = 2.0,
    max_files_per_change: int = 50,
    regex_cache: regexsafe.RegexRequestCache | None = None,
) -> list[dict]:
    """Return logical change-sets after merge/noise filtering."""

    return list(
        group_changes_result(
            commits,
            group_by=group_by,
            ticket_regex=ticket_regex,
            window_hours=window_hours,
            max_files_per_change=max_files_per_change,
            regex_cache=regex_cache,
        )["change_sets"]
    )


def cochange(change_sets: Iterable[dict], limit: int | None = 5) -> dict[str, list[dict]]:
    """Compute temporal coupling as association rules ranked by lift."""

    changes = [c for c in change_sets if isinstance(c, dict)]
    total = len(changes)
    if total <= 0:
        return {}
    file_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    for change in changes:
        files = sorted({_norm_path(path) for path in (change.get("files") or []) if _norm_path(path)})
        if not files:
            continue
        file_counts.update(files)
        for idx, left in enumerate(files):
            for right in files[idx + 1 :]:
                pair_counts[(left, right)] += 1

    by_file: dict[str, list[dict]] = defaultdict(list)
    for (left, right), support in pair_counts.items():
        for source, target in ((left, right), (right, left)):
            source_count = file_counts[source]
            target_count = file_counts[target]
            if source_count <= 0 or target_count <= 0:
                continue
            confidence = support / source_count
            lift = (support * total) / (source_count * target_count)
            by_file[source].append(
                {
                    "path": target,
                    "support": int(support),
                    "confidence": round(confidence, 6),
                    "lift": round(lift, 6),
                }
            )
    top_n = None if limit is None else max(0, int(limit))
    out: dict[str, list[dict]] = {}
    for path, items in by_file.items():
        items.sort(key=lambda item: (-item["lift"], -item["support"], -item["confidence"], item["path"]))
        out[path] = items if top_n is None else items[:top_n]
    return dict(sorted(out.items()))


def churn_result(
    change_sets: Iterable[dict],
    now_ts: int | None = None,
    recent_days: int = 90,
    fix_regex: str | None = DEFAULT_FIX_REGEX,
    regex_cache: regexsafe.RegexRequestCache | None = None,
    fix_density_ready: bool = True,
) -> dict:
    """Compute per-file churn/recency/fix-density counts and regex warnings."""

    changes = [c for c in change_sets if isinstance(c, dict)]
    if now_ts is None:
        now_ts = max([int(c.get("ts", 0) or 0) for c in changes] or [0])
    recent_cutoff = int(now_ts) - max(0, int(recent_days or 0)) * 86400
    messages_by_change: list[list[str]] = []
    flat_messages: list[str] = []
    for change in changes:
        messages = [str(message or "") for message in (change.get("messages") or [])]
        if not messages:
            messages = [str(change.get("message") or "")]
        messages_by_change.append(messages)
        flat_messages.extend(messages)
    runner = regex_cache.search_many_or_default if regex_cache is not None else regexsafe.search_many_or_default
    _fix_pattern, message_hits, regex_warnings = runner(
        fix_regex,
        DEFAULT_FIX_REGEX,
        flat_messages,
        label="fix_regex",
    )
    fix_by_change: list[bool] = []
    offset = 0
    for messages in messages_by_change:
        width = len(messages)
        fix_by_change.append(any(message_hits[offset : offset + width]))
        offset += width

    stats: dict[str, dict] = {}
    fix_hits: Counter[str] = Counter()
    for idx, change in enumerate(changes):
        files = sorted({_norm_path(path) for path in (change.get("files") or []) if _norm_path(path)})
        if not files:
            continue
        try:
            ts = int(change.get("ts", 0) or 0)
        except (TypeError, ValueError):
            ts = 0
        is_fix = fix_by_change[idx] if idx < len(fix_by_change) else False
        churn_by_path: Counter[str] = Counter()
        for entry in change.get("paths") or []:
            if not isinstance(entry, dict):
                continue
            path = _norm_path(entry.get("path"))
            if not path:
                continue
            churn_by_path[path] += int(entry.get("added", 0) or 0) + int(entry.get("deleted", 0) or 0)
        for path in files:
            row = stats.setdefault(
                path,
                {
                    "changes": 0,
                    "churn_lines": 0,
                    "last_touched_ts": 0,
                    "recent_changes": 0,
                    "fix_density": 0.0 if fix_density_ready else None,
                },
            )
            row["changes"] += 1
            row["churn_lines"] += int(churn_by_path.get(path, 0))
            row["last_touched_ts"] = max(int(row["last_touched_ts"]), ts)
            if ts >= recent_cutoff:
                row["recent_changes"] += 1
            if is_fix:
                fix_hits[path] += 1
    if fix_density_ready:
        for path, row in stats.items():
            changes = max(1, int(row["changes"]))
            row["fix_density"] = round(fix_hits[path] / changes, 6)
    return {
        "files": dict(sorted(stats.items())),
        "regex_warnings": regex_warnings,
        "fix_regex": _fix_pattern,
    }


def churn(
    change_sets: Iterable[dict],
    now_ts: int | None = None,
    recent_days: int = 90,
    fix_regex: str | None = DEFAULT_FIX_REGEX,
    regex_cache: regexsafe.RegexRequestCache | None = None,
) -> dict[str, dict]:
    """Compute per-file churn/recency/fix-density counts."""

    return churn_result(
        change_sets,
        now_ts=now_ts,
        recent_days=recent_days,
        fix_regex=fix_regex,
        regex_cache=regex_cache,
    )["files"]


def _complexity_for(file_row: dict) -> dict:
    path = _norm_path(file_row.get("path"))
    try:
        chunks = max(0, int(file_row.get("chunks", 0) or 0))
    except (TypeError, ValueError):
        chunks = 0
    if "symbols_count" in file_row:
        try:
            symbols_count = max(0, int(file_row.get("symbols_count", 0) or 0))
        except (TypeError, ValueError):
            symbols_count = 0
    else:
        symbols_count = len(file_row.get("symbols") or [])
    raw_indent = file_row.get("indent_complexity")
    try:
        indent_complexity = max(0.0, float(raw_indent))
    except (TypeError, ValueError):
        indent_complexity = 0.0
    fallback_complexity = float(chunks + symbols_count)
    complexity = indent_complexity if raw_indent is not None else fallback_complexity
    return {
        "path": path,
        "chunks": chunks,
        "symbols_count": symbols_count,
        "indent_complexity": indent_complexity,
        "complexity": complexity,
    }


def _median_positive(values: Iterable[float]) -> float:
    positives = [float(value) for value in values if float(value) > 0]
    return float(median(positives)) if positives else 0.0


def defect_introductions(attributions: Iterable[dict]) -> dict[str, dict]:
    """Summarize SZZ blame attributions by file.

    Each attribution is expected to name the later fix commit, the blamed
    introducing commit, the file path in the parent, and an optional line count.
    """

    by_file: dict[str, dict] = {}
    introducing: dict[str, set[str]] = defaultdict(set)
    fixes: dict[str, set[str]] = defaultdict(set)
    for item in attributions:
        if not isinstance(item, dict):
            continue
        path = _norm_path(item.get("path"))
        fix_commit = str(item.get("fix_commit") or "")
        introducing_commit = str(item.get("introducing_commit") or "")
        if not path or not fix_commit or not introducing_commit:
            continue
        try:
            lines = max(1, int(item.get("lines", 1) or 1))
        except (TypeError, ValueError):
            lines = 1
        row = by_file.setdefault(
            path,
            {
                "defect_introducing_lines": 0,
                "defect_introducing_commits": 0,
                "fix_commits_with_szz": 0,
                "introducing_commits": [],
            },
        )
        row["defect_introducing_lines"] += lines
        introducing[path].add(introducing_commit)
        fixes[path].add(fix_commit)
    for path, row in by_file.items():
        row["introducing_commits"] = sorted(introducing[path])
        row["defect_introducing_commits"] = len(introducing[path])
        row["fix_commits_with_szz"] = len(fixes[path])
    return dict(sorted(by_file.items()))


def hotspots(
    churn_by_file: dict[str, dict],
    catalog_files: Iterable[dict],
    limit: int = 25,
    defect_by_file: dict[str, dict] | None = None,
) -> dict:
    """Combine churn, indentation complexity, and SZZ defect signals."""

    complexity = {
        item["path"]: item
        for item in (_complexity_for(file_row) for file_row in catalog_files)
        if item["path"]
    }
    defect_by_file = defect_by_file or {}
    change_threshold = _median_positive(
        int((churn_by_file.get(path) or {}).get("changes", 0) or 0)
        for path in set(complexity) | set(churn_by_file)
    )
    complexity_threshold = _median_positive(item["complexity"] for item in complexity.values())
    per_file: dict[str, dict] = {}
    ranked: list[dict] = []
    for path in sorted(set(complexity) | set(churn_by_file) | set(defect_by_file)):
        churn_row = churn_by_file.get(path) or {}
        comp = complexity.get(path) or {"chunks": 0, "symbols_count": 0, "complexity": 0}
        defect = defect_by_file.get(path) or {}
        changes = int(churn_row.get("changes", 0) or 0)
        complexity_score = float(comp.get("complexity", 0.0) or 0.0)
        high_churn = bool(change_threshold and changes >= change_threshold)
        high_complexity = bool(complexity_threshold and complexity_score >= complexity_threshold)
        quadrant = (
            ("high_churn" if high_churn else "low_churn")
            + "_"
            + ("high_complexity" if high_complexity else "low_complexity")
        )
        defect_commits = int(defect.get("defect_introducing_commits", 0) or 0)
        defect_lines = int(defect.get("defect_introducing_lines", 0) or 0)
        defect_hotspot_score = defect_commits * max(1, changes) + defect_lines
        raw_fix_density = churn_row.get("fix_density", 0.0)
        item = {
            "path": path,
            "changes": changes,
            "churn_lines": int(churn_row.get("churn_lines", 0) or 0),
            "recent_changes": int(churn_row.get("recent_changes", 0) or 0),
            "last_touched_ts": int(churn_row.get("last_touched_ts", 0) or 0),
            "fix_density": (
                None
                if raw_fix_density is None
                else float(raw_fix_density or 0.0)
            ),
            "defect_introducing_commits": defect_commits,
            "defect_introducing_lines": defect_lines,
            "defect_hotspot_score": defect_hotspot_score,
            "chunks": int(comp.get("chunks", 0) or 0),
            "symbols_count": int(comp.get("symbols_count", 0) or 0),
            "indent_complexity": float(comp.get("indent_complexity", 0.0) or 0.0),
            "complexity": complexity_score,
            "hotspot_quadrant": quadrant,
            "hotspot_score": changes * max(1, complexity_score),
        }
        per_file[path] = {
            "hotspot_quadrant": quadrant,
            "complexity": complexity_score,
            "indent_complexity": item["indent_complexity"],
            "symbols_count": item["symbols_count"],
            "chunks": item["chunks"],
            "defect_introducing_commits": defect_commits,
            "defect_introducing_lines": defect_lines,
            "defect_hotspot_score": defect_hotspot_score,
        }
        if changes > 0 or defect_hotspot_score > 0:
            ranked.append(item)
    ranked.sort(
        key=lambda item: (
            item["hotspot_quadrant"] != "high_churn_high_complexity",
            -item["hotspot_score"],
            -item["defect_hotspot_score"],
            -item["changes"],
            -item["churn_lines"],
            item["path"],
        )
    )
    return {"files": per_file, "hotspots": ranked[: max(0, int(limit))]}
