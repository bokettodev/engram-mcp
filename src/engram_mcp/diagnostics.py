"""Read-only index health and indexed-content diagnostics."""

from __future__ import annotations

import multiprocessing as mp
import os
import re
import threading
from pathlib import Path
from typing import Callable

from engram_mcp import catalog, config, errors, gcreclaim, gitmeta, manifest, paths
from engram_mcp.embeddings import cache as embedding_cache
from engram_mcp.grepworker import grep_rows_worker
from engram_mcp.index_repository import (
    ProjectNotIndexedError,
    QueryIndex,
    catalog_deep_validation_error as _catalog_deep_validation_error,
    load_query_index,
)
from engram_mcp.query_service import freshness_for_hit
from engram_mcp.store.lancedb_store import LanceStore


DEFAULT_GREP_REGEX_TIMEOUT_SEC = 2.0
GREP_WORKER_BATCH_ROWS = 200
MAX_GREP_LIMIT = 200
MAX_GREP_MAX_MATCHES = 5000
MAX_GREP_SCAN_CHUNKS = 100000

GrepRunner = Callable[..., dict]


def doctor_project(root: str | Path, *, check_git: bool = True) -> dict:
    """Read-only index health check. Does not load the embedding model."""

    root = Path(root).expanduser().resolve()
    pdir = paths.project_dir(root, create=False)
    issues: list[dict] = []
    if not pdir.exists():
        raise ProjectNotIndexedError(f"project not indexed: {root}")

    def issue(code: str, severity: str, message: str, hint: str | None = None) -> None:
        item = {"code": code, "severity": severity, "message": message}
        if hint:
            item["hint"] = hint
        issues.append(item)

    try:
        m = manifest.load_project_strict(pdir)
    except errors.EngramError as exc:
        issue(exc.code, "error", str(exc), exc.hint)
        return {
            "project_path": str(root),
            "source_type": "static_indexed_source",
            "ok": False,
            "summary": {"issues": len(issues), "errors": 1, "warnings": 0},
            "issues": issues,
        }
    if m is None:
        raise ProjectNotIndexedError(f"project not indexed: {root}")

    if m.root_path and not Path(m.root_path).exists():
        issue("root_missing", "error", "manifest root_path no longer exists")
    # Deferred import: doctor_project must stay torch-free and never load the
    # embedding model, but comparing against the *current* canonical id (repo
    # + pinned revision + dim + pooling, see embeddings/factory.py) is a pure
    # string computation, not a model load.
    from engram_mcp.embeddings.factory import CANONICAL_EMBEDDER_ID

    if m.embedder_id != CANONICAL_EMBEDDER_ID:
        issue("model_drift", "error", f"manifest embedder_id is {m.embedder_id!r}")
    if m.chunker_version != config.CHUNKER_VERSION:
        issue("chunker_drift", "error", f"manifest chunker_version is {m.chunker_version!r}")
    if m.chunk_id_scheme != config.CHUNK_ID_SCHEME:
        issue("chunk_id_scheme_drift", "error", f"manifest chunk_id_scheme is {m.chunk_id_scheme!r}")
    if m.schema_version != manifest.SCHEMA_VERSION:
        issue("manifest_schema", "error", f"manifest schema_version is {m.schema_version!r}")

    try:
        files_meta = manifest.load_files_strict(
            pdir, generation=m.generation, active_table=m.active_table or ""
        )
    except errors.EngramError as exc:
        issue("files_manifest_corrupt", "error", str(exc), exc.hint)
        # Informational-only fallback so the rest of this read-only report
        # (e.g. stale-file detection below) still has something to work
        # with; never used to decide what's deleted.
        files_meta = manifest.load_files(pdir)
    if len(files_meta) != m.files:
        issue("files_manifest_mismatch", "warning", "files.json count differs from project manifest")

    table_rows = None
    store = LanceStore(pdir / "lancedb", max(1, m.dim or 1), table=m.active_table or "chunks")
    expected_schema = set(store.expected_schema_names())
    if not store.exists():
        issue("table_missing", "error", f"active table {m.active_table!r} is missing")
    else:
        try:
            table_rows = store.count()
            if table_rows != m.chunks:
                issue("table_count_mismatch", "error", "active table row count differs from manifest chunks")
        except Exception as exc:
            issue("table_unreadable", "error", str(exc))
        schema_names = set(store.schema_names())
        missing_cols = sorted(expected_schema - schema_names)
        if missing_cols:
            issue("table_schema_mismatch", "error", "active table is missing columns: " + ", ".join(missing_cols))
        _rows, fts_warning = store.search_text_with_status("engram", k=1)
        if fts_warning:
            issue("fts_unavailable", "warning", fts_warning)

    cat = catalog.load_catalog(pdir, m.generation)
    if cat is None:
        issue("catalog_missing", "error", f"catalog_g{m.generation}.json is missing or invalid")
    else:
        catalog_qi = QueryIndex(root=root, pdir=pdir, manifest=m, store=store, count=table_rows or 0)
        # doctor_project is a diagnostic tool, not the search hot path: it is
        # the one place that still pays for the full O(total chunks) id-set
        # comparison the search-time check no longer performs on every query.
        catalog_problem = _catalog_deep_validation_error(cat, catalog_qi)
        if catalog_problem:
            issue("catalog_count_mismatch", "error", catalog_problem)
        else:
            totals = cat.get("totals") or {}
            if totals.get("files") != m.files or totals.get("chunks") != m.chunks:
                issue("catalog_count_mismatch", "error", "catalog totals differ from manifest")

    stale_files = []
    for rel in files_meta:
        stale, reason = freshness_for_hit(root, files_meta, {"rel_path": rel})
        if stale:
            stale_files.append({"path": rel, "reason": reason})
    if stale_files:
        issue("index_stale", "warning", f"{len(stale_files)} indexed files differ from the working tree")

    git_status = None
    if check_git:
        indexed_git = {
            "git_worktree_root": m.git_worktree_root,
            "indexed_ref": m.indexed_ref,
            "indexed_commit": m.indexed_commit,
            "indexed_dirty": m.indexed_dirty,
        }
        try:
            git_status = gitmeta.current_staleness(root, indexed_git)
        except Exception as exc:
            git_status = {
                "available": False,
                "git_stale": False,
                "reasons": [],
                "indexed": indexed_git,
                "current": {},
                "warning": f"git metadata unavailable: {exc}",
            }
            issue("git_unavailable", "warning", f"git metadata unavailable: {exc}")
        if git_status.get("git_stale"):
            issue("git_stale", "warning", "current git state differs from indexed git state")

    errors_count = sum(1 for i in issues if i["severity"] == "error")
    warnings_count = sum(1 for i in issues if i["severity"] == "warning")

    # Storage reporting is pure filesystem stat-ing (no LanceDB connect, no
    # directory creation, no writes) so it is safe on this read path. See
    # `gcreclaim.project_storage_report` and `embeddings.cache.read_only_stats`
    # for the read-only guarantees each one makes.
    storage_report = gcreclaim.project_storage_report(pdir)
    cache_stats = embedding_cache.read_only_stats(embedding_cache.global_cache_path())
    storage = {
        "active_generation_bytes": storage_report.get("active_table_bytes", 0),
        "active_catalog_bytes": storage_report.get("active_catalog_bytes", 0),
        "stale_generation_bytes": storage_report.get("stale_table_bytes", 0),
        "stale_catalog_bytes": storage_report.get("stale_catalog_bytes", 0),
        "stale_generations": storage_report.get("stale_tables", []),
        "reclaim_hint": (
            "run `engram gc --prune` to reclaim stale generations across all indexed projects"
            if storage_report.get("stale_tables")
            else None
        ),
        "global_cache_path": cache_stats.get("path"),
        "global_cache_bytes": cache_stats.get("bytes", 0),
        "global_cache_rows": cache_stats.get("rows"),
    }

    return {
        "project_path": str(root),
        "project_id": m.project_id,
        "logical_project_id": m.logical_project_id,
        "checkout_kind": m.checkout_kind,
        "indexed_ref": m.indexed_ref,
        "index_generation": m.generation,
        "source_type": "static_indexed_source",
        "ok": errors_count == 0,
        "summary": {
            "issues": len(issues),
            "errors": errors_count,
            "warnings": warnings_count,
            "manifest_files": m.files,
            "manifest_chunks": m.chunks,
            "table_rows": table_rows,
            "stale_files": len(stale_files),
        },
        "git": git_status,
        "storage": storage,
        "issues": issues,
    }


def grep_regex_timeout_seconds() -> float:
    raw = os.environ.get("ENGRAM_GREP_REGEX_TIMEOUT_SEC", "").strip()
    if not raw:
        value = DEFAULT_GREP_REGEX_TIMEOUT_SEC
    else:
        try:
            value = float(raw)
        except ValueError:
            value = DEFAULT_GREP_REGEX_TIMEOUT_SEC
    return max(0.05, min(value, 30.0))


def grep_rows_with_timeout(
    *,
    pattern: str,
    flags: int,
    rows: list[dict],
    include_lines: bool,
    max_matches: int,
    timeout_sec: float,
    worker: Callable = grep_rows_worker,
) -> dict:
    """Run the regex over ``rows`` in an isolated, timed-out subprocess.

    ``rows`` is never handed to ``Process(args=...)``: on ``spawn`` (the only
    start method on Windows) that would serialize and copy the whole corpus
    into the child during ``proc.start()``, before ``timeout_sec`` even starts
    being measured below. Instead the child is started with only the small,
    fixed-size arguments (pattern/flags/include_lines/max_matches) plus its
    end of a data pipe, and a background feeder thread streams ``rows`` to it
    in ``GREP_WORKER_BATCH_ROWS``-sized batches while the main thread waits on
    the *same* ``timeout_sec`` for a result -- so both the transfer and the
    regex work are inside one bounded window, and peak memory is one batch,
    not the whole corpus.
    """
    def degraded(reason: str) -> dict:
        warning = str(reason or "regex execution unavailable")
        return {"by_path": {}, "total_matches": 0, "stopped": True, "warning": warning}

    # Context/pipe/process construction and start() live inside the guarded
    # region below: an OSError while allocating the pipe or spawning the
    # process must degrade to a partial result, not escape as a server error.
    proc = None
    result_recv = None
    result_send = None
    data_recv = None
    data_send = None
    started = False
    feeder = None
    try:
        try:
            ctx = mp.get_context("spawn")
            data_recv, data_send = ctx.Pipe(duplex=False)
            result_recv, result_send = ctx.Pipe(duplex=False)
            proc = ctx.Process(
                target=worker,
                args=(data_recv, result_send, pattern, flags, include_lines, max_matches),
            )
            proc.start()
            started = True
            # Only the child needs to read the data channel or write the
            # result channel; drop our copies so the child holds the only
            # live ends (mirrors the original single-pipe close-after-start
            # pattern, generalized to the two channels below).
            data_recv.close()
            data_recv = None
            result_send.close()
            result_send = None

            def feed() -> None:
                try:
                    for i in range(0, len(rows), GREP_WORKER_BATCH_ROWS):
                        data_send.send(rows[i : i + GREP_WORKER_BATCH_ROWS])
                    data_send.send(None)
                except (BrokenPipeError, OSError, EOFError):
                    pass

            feeder = threading.Thread(target=feed, daemon=True)
            feeder.start()
            if result_recv.poll(timeout_sec):
                status, payload = result_recv.recv()
                proc.join(timeout=1)
                if status == "ok":
                    return payload
                return degraded(f"regex execution failed: {payload}")
            return degraded(f"regex execution timed out after {timeout_sec:.2f}s")
        except Exception as exc:
            return degraded(str(exc) or repr(exc))
    finally:
        # Each cleanup step is independently best-effort: a failure in one
        # (e.g. terminate() raising) must not skip the rest of the sequence.
        if started and proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.join(timeout=1)
            except Exception:
                pass
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.join(timeout=1)
            except Exception:
                pass
        # Closing our end of the data pipe here unblocks a feeder thread
        # still mid-send (broken pipe) after the process above has been
        # terminated/killed, instead of leaving it parked on a syscall
        # indefinitely.
        if data_send is not None:
            try:
                data_send.close()
            except OSError:
                pass
        if feeder is not None:
            feeder.join(timeout=1)
        if data_recv is not None:
            try:
                data_recv.close()
            except OSError:
                pass
        if result_send is not None:
            try:
                result_send.close()
            except OSError:
                pass
        if result_recv is not None:
            try:
                result_recv.close()
            except OSError:
                pass
        if proc is not None:
            try:
                proc.close()
            except Exception:
                pass


def grep_index(
    root: str | Path,
    pattern: str,
    *,
    ignore_case: bool = False,
    limit: int = 50,
    offset: int = 0,
    max_matches: int = 500,
    max_scan_chunks: int = 10000,
    include_lines: bool = False,
    _grep_runner: GrepRunner | None = None,
) -> dict:
    """Bounded regex/count probe over indexed chunk text."""

    if not pattern or not pattern.strip():
        raise ValueError("pattern must not be empty")
    if len(pattern) > 500:
        raise ValueError("pattern must be at most 500 characters")
    flags = re.IGNORECASE if ignore_case else 0
    try:
        re.compile(pattern, flags)
    except re.error as exc:
        raise ValueError(f"invalid regex: {exc}") from exc
    clamp_warnings: list[str] = []

    def _clamp(value: int, *, name: str, minimum: int, maximum: int) -> int:
        coerced = max(minimum, min(int(value), maximum))
        if int(value) > maximum:
            clamp_warnings.append(
                f"{name} clamped to server maximum {maximum} (requested {int(value)})"
            )
        return coerced

    limit = _clamp(limit, name="limit", minimum=1, maximum=MAX_GREP_LIMIT)
    offset = max(0, int(offset))
    max_matches = _clamp(max_matches, name="max_matches", minimum=1, maximum=MAX_GREP_MAX_MATCHES)
    max_scan_chunks = _clamp(
        max_scan_chunks, name="max_scan_chunks", minimum=1, maximum=MAX_GREP_SCAN_CHUNKS
    )

    qi = load_query_index(root)
    rows = qi.store.metadata_rows(
        columns=("rel_path", "start_line", "content"),
        limit=min(qi.count, max_scan_chunks),
    )
    runner = _grep_runner or grep_rows_with_timeout
    regex_result = runner(
        pattern=pattern,
        flags=flags,
        rows=rows,
        include_lines=include_lines,
        max_matches=max_matches,
        timeout_sec=grep_regex_timeout_seconds(),
    )
    by_path = regex_result["by_path"]
    total_matches = int(regex_result["total_matches"])
    stopped = bool(regex_result["stopped"])
    warning = str(regex_result.get("warning") or "")
    items = []
    for item in by_path.values():
        item["line_numbers"] = sorted(item["line_numbers"])
        if not include_lines:
            item.pop("lines", None)
        items.append(item)
    items.sort(key=lambda r: (-r["match_count"], r["path"]))
    page = items[offset : offset + limit]
    all_warnings = clamp_warnings + ([warning] if warning else [])
    return {
        "project_path": str(qi.root),
        "project_id": qi.manifest.project_id,
        "index_generation": qi.manifest.generation,
        "source_type": "static_indexed_source",
        "pattern": pattern,
        "status": "partial" if warning else "ready",
        "ignore_case": ignore_case,
        "limit": limit,
        "offset": offset,
        "count": len(page),
        "total_paths": len(items),
        "total_matches": total_matches,
        "max_matches": max_matches,
        "scanned_chunks": len(rows),
        "max_scan_chunks": max_scan_chunks,
        "truncated": stopped or len(rows) >= max_scan_chunks,
        "warning": warning,
        "warnings": all_warnings,
        "has_more": offset + limit < len(items),
        "results": page,
    }
