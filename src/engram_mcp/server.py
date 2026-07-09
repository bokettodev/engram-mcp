"""MCP server (stdio) exposing async indexing and static indexed-source search.

The tool surface is registered at module bottom in ``register_tools`` so
mutating tools can be withheld in read-only mode.

Indexing runs on a single-worker background thread pool so a tool call returns
immediately and concurrent index requests serialize on the one embedder.

Read-only mode: set ENGRAM_READONLY=1 (env) to expose ONLY the read tools
(search_code / get_chunk / find_definition / project_map / doctor_project /
grep_index / model_status / index_status / list_indexed_projects /
server_info). The
mutating tools (index_project / reindex_file / remove_project) are not
registered, so a client physically cannot alter an index. Indexing is then
driven out-of-band (e.g. the `engram` CLI/operator). Intended for hosts that
hand the server to untrusted callers (agents) while a separate process owns
indexing.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from threading import Lock, Thread
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from engram_mcp import config, errors, inventory, paths, pipeline
from engram_mcp.embeddings import factory
from engram_mcp.jobs import JobRegistry, snapshot
from engram_mcp.pipeline import (
    DEFAULT_RERANK_CANDIDATE_K,
    MAX_RERANK_CANDIDATES,
    MAX_SEARCH_K,
    ProjectNotIndexedError,
)
from engram_mcp.pipeline import doctor_project as _run_doctor_project
from engram_mcp.pipeline import find_definition as _run_find_def
from engram_mcp.pipeline import get_chunk as _run_get_chunk
from engram_mcp.pipeline import grep_index as _run_grep_index
from engram_mcp.pipeline import index_project as _run_index
from engram_mcp.pipeline import load_query_index
from engram_mcp.pipeline import plan_index as _plan_index
from engram_mcp.pipeline import project_map as _run_project_map
from engram_mcp.pipeline import reindex_file as _run_reindex_file
from engram_mcp.pipeline import remove_project as _run_remove
from engram_mcp.pipeline import search_project as _run_search

mcp = FastMCP("engram")
_registry = JobRegistry()
_index_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cidx-index")
_warmup_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cidx-warmup")
_model_loads = {}
_model_loads_lock = Lock()

_RESULT_FIELDS = (
    "chunk_id", "rel_path", "start_line", "end_line", "language", "symbol", "symbol_kind",
    "chunk_role", "score", "raw_score", "score_normalized", "relevance", "matched",
    "match_reason", "stale", "index_stale", "git_stale", "freshness_reason",
)
_CONTENT_MODES = {"none", "preview", "full"}
_DEFAULT_PREVIEW_CHARS = 800
_MAX_RESULT_CHARS = 20_000
_MODEL_RETRY_AFTER_SEC = 2
_DEFAULT_SEARCH_WAIT_SEC = 8.0
_DEFAULT_DELTA_CPU_MAX = 1024
_MAX_TOTAL_CHARS = 200_000


def _get_provider(index_device: str | None = None):
    """Index-time provider for the selected backend."""
    return factory.make_index_provider(index_device)


def _apply_index_progress_event(job_id: str, event: dict) -> None:
    fields = {}
    stage = event.get("stage")
    if isinstance(stage, str) and stage:
        fields["stage"] = stage
    if "unit" in event:
        fields["progress_unit"] = event.get("unit") or ""
    if "done" in event:
        fields["done_units"] = int(event.get("done") or 0)
    if "total" in event:
        total = event.get("total")
        fields["total_units"] = int(total) if total is not None else None
    for event_key, job_key in (
        ("files", "files"),
        ("chunks", "chunks"),
        ("embedded", "embedded"),
        ("reused", "reused"),
    ):
        if event_key in event and event.get(event_key) is not None:
            fields[job_key] = int(event[event_key])
    if fields:
        _registry.update(job_id, **fields)


def _stderr_tail_text(lines: deque[str]) -> str | None:
    text = "".join(lines).strip()
    return text[-1000:] or None


def _drain_stderr(stream, tail: deque[str]) -> None:
    if stream is None:
        return
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break
            tail.append(line)
    except Exception:
        return


def _subprocess_index(job_id: str, project_path: str, full_rebuild: bool, setting: str) -> None:
    """Run a GPU/auto index in a short-lived subprocess.

    Keeps torch/CUDA out of the long-lived server process entirely: the child
    resolves the device (auto prefers GPU), initializes CUDA if used, indexes,
    writes the canonical FastEmbed manifest, and its whole CUDA context is
    reclaimed when it exits — so the server stays 0-VRAM even after GPU jobs.
    Explicit CPU indexing runs in-process (it never touches CUDA); search never
    touches this path.
    """
    _registry.update(job_id, stage="loading_model", index_device=setting)
    cmd = [sys.executable, "-m", "engram_mcp.cli", "index", project_path,
           "--index-device", setting, "--json"]
    if full_rebuild:
        cmd.append("--rebuild")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except Exception as exc:
        _registry.update(
            job_id, status="error", stage="error",
            error=f"failed to launch GPU index subprocess: {exc!r}",
            code=errors.E_MODEL_LOAD_FAILED, finished_at=time.time(),
        )
        return

    stderr_tail: deque[str] = deque(maxlen=80)
    stderr_thread = Thread(
        target=_drain_stderr,
        args=(proc.stderr, stderr_tail),
        daemon=True,
        name="cidx-index-stderr",
    )
    stderr_thread.start()
    result = None
    stdout = proc.stdout
    if stdout is not None:
        for line in stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
            except ValueError:
                continue
            event = data.get("event")
            if event == "progress":
                _apply_index_progress_event(job_id, data)
            elif event == "result" or "ok" in data:
                result = data
            elif "stage" in data:
                _apply_index_progress_event(job_id, data)
    returncode = proc.wait()
    stderr_thread.join(timeout=1.0)

    data = result
    if data is None:
        # Back-compat for a mocked/older child that put a final object in a
        # buffered stdout string instead of streaming line iteration.
        buffered = getattr(proc, "stdout", None)
        text = getattr(buffered, "getvalue", lambda: "")() if buffered is not None else ""
        for line in reversed((text or "").splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                candidate = json.loads(line)
            except ValueError:
                continue
            if candidate.get("event") == "result" or "ok" in candidate:
                data = candidate
                break
    if data is None:
        hint = _stderr_tail_text(stderr_tail)
        if returncode == 0:
            hint = hint or "The subprocess exited successfully but did not emit a result event."
        _registry.update(
            job_id, status="error", stage="error",
            error=f"GPU index subprocess produced no result (exit {returncode})",
            hint=hint,
            code=errors.E_MODEL_LOAD_FAILED, finished_at=time.time(),
        )
        return
    if not data.get("ok"):
        _registry.update(
            job_id, status="error", stage="error", error=data.get("error"),
            code=data.get("code"), hint=data.get("hint") or _stderr_tail_text(stderr_tail),
            finished_at=time.time(),
        )
        return
    chunks = data.get("chunks")
    _registry.update(
        job_id, status="done", stage="done", error=None, code=None, hint=None,
        index_device=data.get("device") or setting,  # actual device the child used
        embedder_id=data.get("embedder_id"), backend_id=data.get("backend_id"),
        files=data.get("files"), chunks=data.get("chunks"),
        embedded=data.get("embedded_unique"), reused=data.get("reused_unique"),
        progress_unit="chunks",
        done_units=chunks or 0,
        total_units=chunks,
        finished_at=time.time(),
    )
    _schedule_query_model_warmup(data.get("embedder_id") or factory.CANONICAL_EMBEDDER_ID)


def _index_worker(job_id: str, project_path: str, full_rebuild: bool, index_device: str) -> None:
    _registry.update(
        job_id,
        status="running",
        stage="loading_model",
        index_device=index_device,
        started_at=time.time(),
    )
    if index_device in ("cuda", "auto"):
        # Isolate GPU/auto (which may load torch) in a child process so the
        # long-lived server never initializes CUDA and stays ~0 VRAM.
        _subprocess_index(job_id, project_path, full_rebuild, index_device)
        return
    provider = None
    try:
        provider = _get_provider(index_device)
        _registry.update(
            job_id,
            stage="loading_model",
            embedder_id=provider.model_id,
            backend_id=provider.backend_id,
        )

        def progress(event: dict) -> None:
            _apply_index_progress_event(job_id, event)

        stats = _run_index(project_path, provider, full_rebuild=full_rebuild, progress=progress)
        _registry.update(
            job_id,
            status="done",
            stage="done",
            error=None,
            code=None,
            hint=None,
            files=stats.files,
            chunks=stats.chunks,
            embedded=stats.embedded_unique,
            reused=stats.reused_unique,
            progress_unit="chunks",
            done_units=stats.chunks,
            total_units=stats.chunks,
            finished_at=time.time(),
        )
        _schedule_query_model_warmup(provider.model_id)
    except Exception as exc:  # surface failure via status, never crash the server
        payload = _error_payload(exc)
        _registry.update(
            job_id,
            status="error",
            stage="error",
            error=payload.get("error", str(exc)),
            code=payload.get("code"),
            hint=payload.get("hint"),
            finished_at=time.time(),
        )
    finally:
        if provider is not None:
            try:
                factory.release_index_provider(provider)
            except Exception:  # never let cleanup turn a finished job into a failure
                pass


# --- plain, testable logic (the MCP tools are thin wrappers over these) ---

def _delta_cpu_max() -> int:
    raw = os.environ.get("ENGRAM_DELTA_CPU_MAX", "").strip()
    if not raw:
        return _DEFAULT_DELTA_CPU_MAX
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_DELTA_CPU_MAX


def _rerank_candidate_k_default() -> int:
    raw = os.environ.get("ENGRAM_RERANK_CANDIDATE_K", "").strip()
    if not raw:
        value = DEFAULT_RERANK_CANDIDATE_K
    else:
        try:
            value = int(raw)
        except ValueError:
            value = DEFAULT_RERANK_CANDIDATE_K
    return max(1, min(value, MAX_RERANK_CANDIDATES))


def start_index_job(
    project_path: str,
    full_rebuild: bool = False,
    gpu: bool = False,
    index_device: str | None = None,
) -> dict:
    root = Path(project_path).expanduser()
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    device = factory.resolve_index_device(index_device, gpu=gpu)
    resolved = str(root.resolve())
    routed_device = device
    plan = None
    if device == "auto":
        plan = _plan_index(
            resolved,
            full_rebuild=full_rebuild,
            model_id=factory.CANONICAL_EMBEDDER_ID,
            dim=config.DEFAULT_EMBED_DIM,
        )
        if plan.missing_unique_chunks <= _delta_cpu_max():
            routed_device = "cpu"
    job = _registry.create(resolved)
    _registry.update(job.job_id, index_device=routed_device, embedder_id=factory.CANONICAL_EMBEDDER_ID)
    _index_pool.submit(_index_worker, job.job_id, resolved, full_rebuild, routed_device)
    plan_payload = None
    if plan is not None:
        plan_payload = {
            "mode": plan.mode,
            "files": plan.files,
            "chunks": plan.chunks,
            "added": plan.added,
            "changed": plan.changed,
            "deleted": plan.deleted,
            "unchanged": plan.unchanged,
            "missing_unique_chunks": plan.missing_unique_chunks,
            "delta_cpu_max": _delta_cpu_max(),
        }
    return {
        "job_id": job.job_id,
        "project_path": resolved,
        "status": job.status,
        "index_device": routed_device,
        "index_device_requested": device,
        "routing": "delta_cpu" if device == "auto" and routed_device == "cpu" else "requested",
        "plan": plan_payload,
        "embedder_id": factory.CANONICAL_EMBEDDER_ID,
    }


def _not_indexed_hint() -> str:
    if read_only_enabled():
        return "Index this project out of band via the engram CLI/operator."
    return "Call index_project first, or run `engram index <project_path>`."


def _error_payload(exc: BaseException, *, results: list | None = None) -> dict:
    if isinstance(exc, ProjectNotIndexedError):
        return errors.error_result(
            str(exc),
            errors.E_PROJECT_NOT_INDEXED,
            hint=_not_indexed_hint(),
            results=results,
        )
    if isinstance(exc, errors.EngramError):
        return errors.error_result(str(exc), exc.code, hint=exc.hint, results=results)
    if isinstance(exc, ImportError):
        return errors.error_result(str(exc), errors.E_EXTRA_MISSING, results=results)
    if isinstance(exc, ValueError):
        return errors.error_result(str(exc), errors.E_BAD_REQUEST, results=results)
    message = str(exc) or repr(exc)
    hint = None
    try:
        from engram_mcp import net

        if net.is_cert_error(exc) or "TLS certificate verification failed" in message:
            hint = net.CERT_HINT
    except Exception:
        pass
    return errors.error_result(message, errors.E_MODEL_LOAD_FAILED, hint=hint, results=results)


def _provider_load_worker(model_id: str):
    return factory.provider_for_model_id(model_id)


def _schedule_query_model_warmup(model_id: str) -> None:
    if not model_id:
        return
    try:
        backend_id = factory.query_backend_id_for_model_id(model_id)
    except Exception:
        return
    if factory.is_model_loaded(backend_id):
        return
    with _model_loads_lock:
        fut = _model_loads.get(model_id)
        if fut is None or (fut.done() and fut.exception() is not None):
            _model_loads[model_id] = _warmup_pool.submit(_provider_load_worker, model_id)


def _search_wait_budget_sec() -> float:
    raw = os.environ.get("ENGRAM_SEARCH_WAIT_SEC", "").strip()
    if not raw:
        return _DEFAULT_SEARCH_WAIT_SEC
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_SEARCH_WAIT_SEC
    return max(0.0, value)


def _model_loading_error(model_id: str) -> errors.EngramError:
    return errors.EngramError(
        f"model for this index is loading: {model_id}",
        errors.E_MODEL_LOADING,
        hint="Retry the search after the reported delay.",
    )


def _provider_for_query_model(model_id: str):
    """Return a loaded provider or wait briefly for the load future."""

    backend_id = factory.query_backend_id_for_model_id(model_id)
    if factory.is_model_loaded(backend_id):
        return factory.provider_for_model_id(model_id)
    with _model_loads_lock:
        fut = _model_loads.get(model_id)
        if fut is None:
            fut = _warmup_pool.submit(_provider_load_worker, model_id)
            _model_loads[model_id] = fut
    if fut.done():
        return fut.result()
    budget = _search_wait_budget_sec()
    if budget > 0:
        try:
            return fut.result(timeout=budget)
        except FutureTimeoutError:
            pass
    raise _model_loading_error(model_id)


def _model_status_for(model_id: str) -> dict:
    backend_id = factory.query_backend_id_for_model_id(model_id)
    base = {"model_id": model_id, "backend_id": backend_id}
    with _model_loads_lock:
        fut = _model_loads.get(model_id)
    if factory.is_model_loaded(backend_id):
        return base | {"status": "loaded"}
    if fut is None:
        return base | {"status": "not_loaded"}
    if not fut.done():
        return base | {"status": "loading", "retry_after_sec": _MODEL_RETRY_AFTER_SEC}
    exc = fut.exception()
    if exc is not None:
        payload = _error_payload(exc)
        return base | {"status": "error", **payload}
    return base | {"status": "loaded"}


def _check_k(k: int) -> None:
    if not isinstance(k, int) or not (1 <= k <= MAX_SEARCH_K):
        raise ValueError(f"k must be between 1 and {MAX_SEARCH_K}")


def _check_content(content: str, max_chars_per_result: int) -> None:
    if content not in _CONTENT_MODES:
        raise ValueError("content must be one of: none, preview, full")
    if (
        not isinstance(max_chars_per_result, int)
        or max_chars_per_result < 1
        or max_chars_per_result > _MAX_RESULT_CHARS
    ):
        raise ValueError(f"max_chars_per_result must be between 1 and {_MAX_RESULT_CHARS}")


def _clip_text(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


_SEARCH_TOKEN = re.compile(r"[A-Za-z0-9_]{2,}")


def _query_tokens(query: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in _SEARCH_TOKEN.findall(query.lower()):
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _matched_in(hit: dict, query: str) -> list[str]:
    toks = _query_tokens(query)
    fields = []
    checks = {
        "content": hit.get("content") or "",
        "symbol": hit.get("symbol") or "",
        "path": hit.get("rel_path") or "",
    }
    for name, value in checks.items():
        low = value.lower()
        if any(tok in low for tok in toks):
            fields.append(name)
    return fields


def _center_excerpt(text: str, query: str, limit: int, *, prefer_literal: bool) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    tokens = _query_tokens(query)
    low = text.lower()
    pos = None
    if prefer_literal:
        found = [
            low.find(tok) for tok in tokens
            if tok and low.find(tok) >= 0
        ]
        if found:
            pos = min(found)
    if pos is None:
        sentences = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
        if sentences:
            scored = []
            for sentence in sentences:
                stoks = set(_query_tokens(sentence))
                scored.append((len(stoks & set(tokens)), len(sentence), sentence))
            scored.sort(key=lambda item: (-item[0], item[1]))
            best = scored[0][2]
            if scored[0][0] > 0 and len(best) <= limit:
                return best, True
        pos = 0
    half = max(0, limit // 2)
    start = max(0, pos - half)
    end = min(len(text), start + limit)
    start = max(0, end - limit)
    excerpt = text[start:end]
    if start > 0:
        excerpt = "..." + excerpt.lstrip()
    if end < len(text):
        excerpt = excerpt.rstrip() + "..."
    if len(excerpt) > limit:
        excerpt = excerpt[:limit]
    return excerpt, True


def _format_search_hit(
    hit: dict,
    content: str,
    max_chars: int,
    *,
    query: str,
    mode_used: str,
    remaining_budget: int | None = None,
) -> tuple[dict, int, bool]:
    out = {field: hit.get(field) for field in _RESULT_FIELDS}
    out["span"] = {"start_line": hit.get("start_line"), "end_line": hit.get("end_line")}
    out["path"] = hit.get("rel_path")
    raw_content = hit.get("content") or ""
    budget = max_chars if remaining_budget is None else max(0, min(max_chars, remaining_budget))
    if budget <= 0 and content != "none":
        out["truncated"] = bool(raw_content)
        out["matched_in"] = _matched_in(hit, query)
        return out, 0, bool(raw_content)
    if content == "none":
        out["truncated"] = False
        out["matched_in"] = _matched_in(hit, query)
        return out, 0, False
    elif content == "preview":
        preview, truncated = _center_excerpt(
            raw_content,
            query,
            budget,
            prefer_literal=mode_used == "hybrid",
        )
        out["preview"] = preview
        out["excerpt"] = preview
        out["matched_in"] = _matched_in(hit, query)
        out["truncated"] = truncated
        return out, len(preview), truncated
    else:
        body, truncated = _clip_text(raw_content, budget)
        out["content"] = body
        out["matched_in"] = _matched_in(hit, query)
        out["truncated"] = truncated
        return out, len(body), truncated


def _result_map(results: list[dict]) -> list[dict]:
    return [
        {
            "rank": i,
            "chunk_id": r.get("chunk_id"),
            "path": r.get("path") or r.get("rel_path"),
            "span": r.get("span"),
            "score": r.get("score"),
            "relevance": r.get("relevance"),
            "symbol": r.get("symbol"),
        }
        for i, r in enumerate(results, 1)
    ]


def _validate_max_total_chars(value: int | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value < 1 or value > _MAX_TOTAL_CHARS:
        raise ValueError(f"max_total_chars must be between 1 and {_MAX_TOTAL_CHARS}")
    return value


def _search_hints(outcome: dict, results: list[dict], content: str, max_total_chars: int | None) -> list[str]:
    hints: list[str] = []
    if not results:
        hints.append("No hits: try mode='hybrid' for exact tokens or find_definition for known symbols.")
    if (outcome.get("dirty") or {}).get("stale"):
        hints.append("Some results are stale: rebuild the index or reindex changed files.")
    if outcome.get("git", {}).get("git_stale"):
        hints.append("Git state differs from the indexed commit/ref; rebuild for current checkout state.")
    if content != "none" and max_total_chars is None and len(results) >= 20:
        hints.append("Large result set: use content='none' or max_total_chars to keep output bounded.")
    return hints


def do_reindex_file(project_path: str, rel_path: str) -> dict:
    provider = None
    try:
        root = Path(project_path).expanduser().resolve()
        qi = load_query_index(root)
        provider = factory.provider_for_model_id(qi.manifest.embedder_id)
        return _run_reindex_file(qi.root, provider, rel_path)
    except Exception as exc:
        return _error_payload(exc)
    finally:
        if provider is not None:
            try:
                factory.release_index_provider(provider)
            except Exception:
                pass


def do_remove_project(project_path: str) -> dict:
    root = Path(project_path).expanduser().resolve()
    return {"removed": _run_remove(root), "project_path": str(root)}


def do_find_definition(project_path: str, symbol: str) -> dict:
    try:
        root = Path(project_path).expanduser().resolve()
        return _run_find_def(root, symbol, include_suggestions=True)
    except Exception as exc:
        return _error_payload(exc, results=[])


def get_status(job_id: str) -> dict:
    job = _registry.get(job_id)
    if job is None:
        return {
            "error": (
                f"unknown job_id in this server process: {job_id}. "
                "Index jobs are tracked only in the current process; list_indexed_projects "
                "shows completed on-disk indexes."
            ),
            "code": errors.E_BAD_REQUEST,
            "scope": "current_process",
        }
    return snapshot(job)


def do_search(
    project_path: str, query: str, k: int = 8, language: str | None = None,
    mode: str = "auto", rerank: bool = False, content: str = "preview",
    max_chars_per_result: int = _DEFAULT_PREVIEW_CHARS,
    max_total_chars: int | None = None,
    candidate_k: int | None = None,
    facets: list[str] | None = None,
    min_relevance: str | None = None,
) -> dict:
    try:
        _check_k(k)
        _check_content(content, max_chars_per_result)
        max_total_chars = _validate_max_total_chars(max_total_chars)
        if candidate_k is None:
            candidate_k = _rerank_candidate_k_default()
        if not isinstance(candidate_k, int) or candidate_k < 1 or candidate_k > MAX_RERANK_CANDIDATES:
            raise ValueError(f"candidate_k must be between 1 and {MAX_RERANK_CANDIDATES}")
        root = Path(project_path).expanduser().resolve()
        qi = load_query_index(root)
        provider = _provider_for_query_model(qi.manifest.embedder_id)
        outcome = _run_search(
            qi.root,
            provider,
            query,
            k=k,
            language=language,
            mode=mode,
            candidate_k=candidate_k,
            rerank=rerank,
            facets=facets,
            min_relevance=min_relevance,
            return_meta=True,
        )
        results = []
        used_chars = 0
        truncated = False
        for h in outcome.pop("hits"):
            remaining = None if max_total_chars is None else max_total_chars - used_chars
            formatted, consumed, item_truncated = _format_search_hit(
                h,
                content,
                max_chars_per_result,
                query=query,
                mode_used=outcome.get("mode_used") or mode,
                remaining_budget=remaining,
            )
            used_chars += consumed
            truncated = truncated or item_truncated
            results.append(formatted)
        return outcome | {
            "content": content,
            "max_chars_per_result": max_chars_per_result,
            "max_total_chars": max_total_chars,
            "body_chars": used_chars,
            "truncated": truncated,
            "count": len(results),
            "map": _result_map(results),
            "hints": _search_hints(outcome, results, content, max_total_chars),
            "results": results,
        }
    except errors.EngramError as exc:
        extra = {}
        if exc.code == errors.E_MODEL_LOADING:
            extra["retry_after_sec"] = _MODEL_RETRY_AFTER_SEC
            extra["hints"] = ["Model is still warming; retry after the reported delay."]
        return _error_payload(exc, results=[]) | extra
    except Exception as exc:
        return _error_payload(exc, results=[])


def list_projects(
    limit: int = inventory.DEFAULT_LIST_LIMIT,
    cursor: str | None = None,
    verbose: bool = False,
    prune_orphans: bool = True,
) -> dict:
    return inventory.list_indexed_projects(
        limit=limit,
        cursor=cursor,
        verbose=verbose,
        prune_orphans=prune_orphans,
    )


def do_project_map(
    project_path: str,
    depth: int = 2,
    sort: str = "path",
    limit: int | None = 200,
    dirs_limit: int | None = None,
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
) -> dict:
    try:
        return _run_project_map(
            Path(project_path).expanduser().resolve(),
            depth=depth,
            sort=sort,
            limit=limit,
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
    except Exception as exc:
        return _error_payload(exc)


def do_doctor_project(project_path: str, check_git: bool = True) -> dict:
    try:
        return _run_doctor_project(Path(project_path).expanduser().resolve(), check_git=check_git)
    except Exception as exc:
        return _error_payload(exc)


def do_grep_index(
    project_path: str,
    pattern: str,
    ignore_case: bool = False,
    limit: int = 50,
    offset: int = 0,
    max_matches: int = 500,
    max_scan_chunks: int = 10000,
    include_lines: bool = False,
) -> dict:
    try:
        return _run_grep_index(
            Path(project_path).expanduser().resolve(),
            pattern,
            ignore_case=ignore_case,
            limit=limit,
            offset=offset,
            max_matches=max_matches,
            max_scan_chunks=max_scan_chunks,
            include_lines=include_lines,
        )
    except Exception as exc:
        return _error_payload(exc, results=[])


def _clip_nested_chunk_content(row: dict, max_chars: int | None) -> None:
    if max_chars is None:
        return
    if "content" in row:
        row["content"], row["truncated"] = _clip_text(row.get("content") or "", max_chars)


def do_get_chunk(
    project_path: str,
    chunk_id: str,
    max_chars: int | None = None,
    include_neighbors: bool = False,
    neighbor_window: int = 1,
    include_parent: bool = False,
) -> dict:
    try:
        row = _run_get_chunk(
            Path(project_path).expanduser().resolve(),
            chunk_id,
            include_neighbors=include_neighbors,
            neighbor_window=neighbor_window,
            include_parent=include_parent,
        )
        content = row.get("content") or ""
        row["truncated"] = False
        if max_chars is not None:
            if not isinstance(max_chars, int) or max_chars < 1 or max_chars > _MAX_RESULT_CHARS:
                raise ValueError(f"max_chars must be between 1 and {_MAX_RESULT_CHARS}")
            row["content"], row["truncated"] = _clip_text(content, max_chars)
            for neighbor in row.get("neighbors") or []:
                _clip_nested_chunk_content(neighbor, max_chars)
            if isinstance(row.get("parent"), dict):
                _clip_nested_chunk_content(row["parent"], max_chars)
        row["span"] = {"start_line": row.get("start_line"), "end_line": row.get("end_line")}
        return row
    except Exception as exc:
        return _error_payload(exc)


def do_model_status(project_path: str | None = None) -> dict:
    try:
        base = {
            "data_home": str(paths.data_home(create=False)),
            "loaded_models": factory.loaded_model_ids(),
            "active_loads": sorted(_model_loads),
            "embedder_id": factory.CANONICAL_EMBEDDER_ID,
        }
        if project_path is None:
            return base
        qi = load_query_index(Path(project_path).expanduser().resolve())
        return base | {
            "project_path": str(qi.root),
            "project_id": qi.manifest.project_id,
            **_model_status_for(qi.manifest.embedder_id),
        }
    except Exception as exc:
        return _error_payload(exc)


def do_server_info() -> dict:
    default_error = None
    try:
        default_device = factory.default_index_device()
    except Exception as exc:
        default_device = factory.DEFAULT_INDEX_DEVICE
        default_error = _error_payload(exc)
    from engram_mcp.rerankers import DEFAULT_BACKEND, DEFAULT_ONNX_RERANKER

    onnx_available = False
    onnx_import_error = None
    try:
        if importlib.util.find_spec("fastembed.rerank.cross_encoder") is None:
            onnx_import_error = "fastembed.rerank.cross_encoder module not found"
        else:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            onnx_available = TextCrossEncoder is not None
            if not onnx_available:
                onnx_import_error = "TextCrossEncoder is unavailable"
    except Exception as exc:
        onnx_import_error = str(exc) or repr(exc)

    onnx_model = os.environ.get("ENGRAM_RERANKER_MODEL") or DEFAULT_ONNX_RERANKER
    return {
        "data_home": str(paths.data_home(create=False)),
        "data_home_source": paths.data_home_source(),
        "data_home_exists": paths.data_home(create=False).exists(),
        "read_only": read_only_enabled(),
        "embedder_id": factory.CANONICAL_EMBEDDER_ID,
        "default_index_device": default_device,
        "default_index_device_error": default_error,
        "supported_index_devices": list(factory.SUPPORTED_INDEX_DEVICES),
        "search_backend": "fastembed-cpu",
        "delta_cpu_max": _delta_cpu_max(),
        "search_warmup_executor": "single_worker",
        "reranker": {
            "enabled": pipeline.rerank_enabled(),
            "enable_env": "ENGRAM_RERANK_ENABLED",
            "gated_to_mode": "vector",
            "default_backend": DEFAULT_BACKEND,
            "onnx_model": onnx_model,
            "onnx_available": onnx_available,
            "onnx_import_error": onnx_import_error,
            "candidate_k_default": _rerank_candidate_k_default(),
            "license_note": (
                "jina-v2-multilingual is CC-BY-NC-4.0; fine for local/private use, "
                "revisit before commercial distribution"
            ),
            "torch_fallback_backend": "sentence_transformers (needs 'gpu' extra)",
        },
        "source_type": "static_indexed_source",
    }


# --- MCP tool surface ---
#
# Tools are defined as plain coroutines and registered explicitly at the bottom
# so the mutating ones can be withheld in read-only mode (ENGRAM_READONLY=1).


async def index_project(
    project_path: str, full_rebuild: bool = False, index_device: str | None = None
) -> dict:
    """Start a background index job and return job/routing details."""
    return start_index_job(project_path, full_rebuild, index_device=index_device)


async def index_status(job_id: str) -> dict:
    """Progress snapshot for an index job in this server process.

    Job tracking is in-memory and scoped to the current MCP process; completed
    on-disk indexes are visible through list_indexed_projects.
    """
    return get_status(job_id)


async def search_code(
    project_path: str, query: str, k: int = 8, language: str | None = None,
    mode: str = "auto", rerank: bool = False, content: str = "preview",
    max_chars_per_result: int = _DEFAULT_PREVIEW_CHARS,
    max_total_chars: int | None = None,
    candidate_k: int | None = None,
    facets: list[str] | None = None,
    min_relevance: str | None = None,
) -> dict:
    """Search indexed source with compact bodies, facets, min relevance, and opt-in rerank."""
    return await asyncio.to_thread(
        do_search,
        project_path=project_path,
        query=query,
        k=k,
        language=language,
        mode=mode,
        rerank=rerank,
        content=content,
        max_chars_per_result=max_chars_per_result,
        max_total_chars=max_total_chars,
        candidate_k=candidate_k,
        facets=facets,
        min_relevance=min_relevance,
    )


async def reindex_file(project_path: str, rel_path: str) -> dict:
    """Incrementally re-index (or drop, if missing) a single file on the index."""
    return await asyncio.to_thread(do_reindex_file, project_path, rel_path)


async def remove_project(project_path: str) -> dict:
    """Delete a project's on-disk index (vectors + manifests)."""
    return await asyncio.to_thread(do_remove_project, project_path)


async def find_definition(project_path: str, symbol: str) -> dict:
    """Exact symbol lookup over static indexed source, with miss suggestions.

    Use this when you already know the symbol name (`symbol` or `Parent.symbol`).
    It does not load an embedding model. On an exact miss, suggestions contains
    nearby symbols from the indexed inventory.
    """
    return await asyncio.to_thread(do_find_definition, project_path, symbol)


async def get_chunk(
    project_path: str,
    chunk_id: str,
    max_chars: int | None = None,
    include_neighbors: bool = False,
    neighbor_window: int = 1,
    include_parent: bool = False,
) -> dict:
    """Fetch one chunk, optionally with adjacent/parent context."""
    return await asyncio.to_thread(
        do_get_chunk,
        project_path,
        chunk_id,
        max_chars,
        include_neighbors,
        neighbor_window,
        include_parent,
    )


async def project_map(
    project_path: str,
    depth: int = 2,
    sort: str = "path",
    limit: int | None = 200,
    dirs_limit: int | None = None,
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
) -> dict:
    """Body-free dirs map; opt into compact, paginated files and filters."""
    return await asyncio.to_thread(
        do_project_map,
        project_path=project_path,
        depth=depth,
        sort=sort,
        limit=limit,
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


async def doctor_project(project_path: str, check_git: bool = True) -> dict:
    """Read-only index health check; does not load an embedding model."""
    return await asyncio.to_thread(do_doctor_project, project_path, check_git)


async def grep_index(
    project_path: str,
    pattern: str,
    ignore_case: bool = False,
    limit: int = 50,
    offset: int = 0,
    max_matches: int = 500,
    max_scan_chunks: int = 10000,
    include_lines: bool = False,
) -> dict:
    """Bounded regex/count probe over indexed text; bodies omitted by default."""
    return await asyncio.to_thread(
        do_grep_index,
        project_path,
        pattern,
        ignore_case,
        limit,
        offset,
        max_matches,
        max_scan_chunks,
        include_lines,
    )


async def model_status(project_path: str | None = None) -> dict:
    """Report whether a project's recorded query model is loaded in this process.

    This is read-only and does not start a model download. If a first search has
    scheduled warmup, status may be loading with retry_after_sec.
    """
    return await asyncio.to_thread(do_model_status, project_path)


async def server_info() -> dict:
    """Server configuration and data-home diagnostics for this process."""
    return do_server_info()


async def list_indexed_projects(
    limit: int = inventory.DEFAULT_LIST_LIMIT,
    cursor: str | None = None,
    verbose: bool = False,
    prune_orphans: bool = True,
) -> dict:
    """List indexed projects with compact pagination and optional health checks.

    Compact mode reads manifests only. ``verbose=true`` may open LanceDB to add
    table health. ``prune_orphans=true`` removes index dirs whose recorded root
    path no longer exists and reports them under ``gc``.
    """
    return list_projects(
        limit=limit,
        cursor=cursor,
        verbose=verbose,
        prune_orphans=prune_orphans,
    )


def read_only_enabled() -> bool:
    """True when ENGRAM_READONLY selects the read-only tool surface."""
    return os.environ.get("ENGRAM_READONLY", "").strip().lower() in ("1", "true", "yes", "on")


def register_tools(read_only: bool) -> None:
    """Register the MCP tool surface; mutating tools are withheld when read_only."""
    # Read tools — always available.
    mcp.tool()(search_code)
    mcp.tool()(find_definition)
    mcp.tool()(get_chunk)
    mcp.tool()(project_map)
    mcp.tool()(doctor_project)
    mcp.tool()(grep_index)
    mcp.tool()(model_status)
    mcp.tool()(index_status)
    mcp.tool()(list_indexed_projects)
    mcp.tool()(server_info)
    # Mutating tools — only when not read-only.
    if not read_only:
        mcp.tool()(index_project)
        mcp.tool()(reindex_file)
        mcp.tool()(remove_project)


register_tools(read_only_enabled())


def _maybe_warmup_on_start() -> None:
    if os.environ.get("ENGRAM_WARMUP_ON_START", "").strip().lower() in ("1", "true", "yes", "on"):
        _schedule_query_model_warmup(factory.CANONICAL_EMBEDDER_ID)


def main() -> None:
    # Set up TLS trust (OS store / CA bundle / insecure) before the background
    # index worker loads a model over HTTPS.
    from engram_mcp.net import configure_tls

    configure_tls()
    _maybe_warmup_on_start()
    mcp.run()


if __name__ == "__main__":
    main()
