"""MCP server (stdio) exposing async indexing and static indexed-source search.

The tool surface is registered at module bottom in ``register_tools`` so
mutating tools can be withheld in read-only mode.

Indexing runs on a single-worker background thread pool so an index_project
tool call returns immediately (before the project walk/plan even runs) and
concurrent index requests for the *same* project coalesce onto one job. The
embedding step itself always runs in a short-lived subprocess
(``_subprocess_index``): this long-lived server process never constructs an
embedding provider or embeds a passage for an index job, so a running index
can never contend with the query path's cached-provider inference lock. See
``_index_worker`` for the planning/routing/dispatch sequence and
``ENGRAM_INPROCESS_CPU_MAX`` for the optional, off-by-default in-process
small-delta fast path.

Read-only mode: set ENGRAM_READONLY=1 (env) to expose ONLY the read tools
(search_code / get_chunk / find_definition / project_map / doctor_project /
model_status / index_status / list_indexed_projects / server_info). The
mutating tools (index_project / cancel_index / reindex_file / remove_project)
are not registered, so a client physically cannot alter an index. Indexing is
then driven out-of-band (e.g. the `engram` CLI/operator). Intended for hosts
that hand the server to untrusted callers (agents) while a separate process
owns indexing.

Note: `grep_index` (approximate regex probe over overlapping chunks) is not
part of the MCP tool surface -- agents have better exact-search tools
available already. The capability still exists as a CLI diagnostic
(`engram grep`) and as `engram_mcp.diagnostics.grep_index`.
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

from engram_mcp import config, errors, gitmeta, inventory, paths, query_service
from engram_mcp.diagnostics import doctor_project as _run_doctor_project
from engram_mcp.embeddings import factory
from engram_mcp.index_repository import ProjectNotIndexedError
from engram_mcp.index_repository import index_project as _run_index
from engram_mcp.index_repository import load_query_index
from engram_mcp.index_repository import plan_index as _plan_index
from engram_mcp.index_repository import reindex_file as _run_reindex_file
from engram_mcp.index_repository import remove_project as _run_remove
from engram_mcp.jobs import JobRegistry, snapshot
from engram_mcp.query_service import (
    DEFAULT_RERANK_CANDIDATE_K,
    MAX_RERANK_CANDIDATES,
    MAX_SEARCH_K,
)
from engram_mcp.query_service import find_definition as _run_find_def
from engram_mcp.query_service import get_chunk as _run_get_chunk
from engram_mcp.query_service import search_project as _run_search
from engram_mcp.structure_service import project_map as _run_project_map

mcp = FastMCP("engram")
_registry = JobRegistry()
_index_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cidx-index")
_warmup_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cidx-warmup")
_model_loads = {}
_model_loads_lock = Lock()

# job_id -> the short-lived index subprocess currently running that job, so
# cancel_index can find and kill it. Populated/cleared by _subprocess_index.
_job_processes: dict[str, subprocess.Popen] = {}
_job_processes_lock = Lock()

_RESULT_FIELDS = (
    "chunk_id", "rel_path", "language", "symbol", "symbol_kind",
    "chunk_role", "raw_score", "score_normalized", "relevance", "matched",
    "match_reason", "stale", "index_stale", "freshness_reason",
)
_CONTENT_MODES = {"none", "preview", "full"}
_DEFAULT_PREVIEW_CHARS = 800
_MAX_RESULT_CHARS = 20_000
_MODEL_RETRY_AFTER_SEC = 2
_DEFAULT_SEARCH_WAIT_SEC = 8.0
_DEFAULT_DELTA_CPU_MAX = 1024
# Off by default: CPU indexing always runs in the short-lived subprocess (see
# _subprocess_index) so the server process never embeds. Setting
# ENGRAM_INPROCESS_CPU_MAX > 0 opts a deployment into an in-process fast path
# for deltas at or below this many missing unique chunk embeddings, trading
# the subprocess's fixed startup+model-load overhead for a small risk of
# briefly contending with a concurrent search (mitigated by always using a
# separate, uncached provider instance -- see factory.make_uncached_cpu_provider).
_DEFAULT_INPROCESS_CPU_MAX = 0
_MAX_TOTAL_CHARS = 200_000
_CANCEL_POLL_INTERVAL_SEC = 0.05
_CANCEL_WAIT_SEC = 5.0


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


def _register_job_process(job_id: str, proc: subprocess.Popen) -> None:
    with _job_processes_lock:
        _job_processes[job_id] = proc


def _unregister_job_process(job_id: str) -> None:
    with _job_processes_lock:
        _job_processes.pop(job_id, None)


def _job_cancel_requested(job_id: str) -> bool:
    job = _registry.get(job_id)
    return bool(job is not None and job.cancel_requested)


def _mark_cancelled(job_id: str, hint: str | None = "Index job cancelled.") -> None:
    _registry.update(
        job_id, status="cancelled", stage="cancelled", error=None, code=None,
        hint=hint, finished_at=time.time(),
    )


def _subprocess_index(
    job_id: str,
    project_path: str,
    full_rebuild: bool,
    setting: str,
    git_max_commits: int | None = None,
    git_analytics: bool | None = None,
) -> None:
    """Run an index job in a short-lived subprocess.

    Every index device (cpu/cuda/auto) runs here: the long-lived server
    process never constructs an embedding provider or embeds a passage for an
    index job, so it can never contend with the query path's inference lock
    and never has to initialize CUDA. The child resolves the device (auto
    prefers GPU), indexes, writes the canonical FastEmbed manifest, and its
    whole process (and CUDA context, if any) is reclaimed when it exits.

    ``git_analytics`` mirrors ``ENGRAM_GIT_ANALYTICS`` precedence: ``None``
    means "let the child CLI resolve its own default from the inherited env
    var", so no flag is passed at all. An explicit ``True``/``False`` always
    wins over the env var, so it is passed as an explicit CLI flag.
    """
    if _job_cancel_requested(job_id):
        _mark_cancelled(job_id)
        return
    _registry.update(job_id, stage="loading_model", index_device=setting)
    cmd = [sys.executable, "-m", "engram_mcp.cli", "index", project_path,
           "--index-device", setting, "--json"]
    if full_rebuild:
        cmd.append("--rebuild")
    if git_max_commits is not None:
        cmd.extend(["--git-max-commits", str(git_max_commits)])
    if git_analytics is not None:
        cmd.append("--git-analytics" if git_analytics else "--no-git-analytics")
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
            error=f"failed to launch index subprocess: {exc!r}",
            code=errors.E_MODEL_LOAD_FAILED, finished_at=time.time(),
        )
        return

    _register_job_process(job_id, proc)
    try:
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
    finally:
        _unregister_job_process(job_id)

    # A cancel may have killed the child mid-flight (no result emitted, or a
    # non-zero exit from the kill itself). Cancellation always wins that race:
    # a killed subprocess never reaches the atomic manifest/generation swap
    # (see index_repository._full_rebuild/_incremental), so the previously published
    # index is untouched and still searchable; there is nothing to "finish".
    # If the child *did* emit a real result (ok or error) despite a late
    # cancel request, the work already completed/failed on its own before the
    # kill could land -- report that outcome normally instead.
    cancelled = _job_cancel_requested(job_id)

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
        if cancelled:
            _mark_cancelled(job_id)
            return
        hint = _stderr_tail_text(stderr_tail)
        if returncode == 0:
            hint = hint or "The subprocess exited successfully but did not emit a result event."
        _registry.update(
            job_id, status="error", stage="error",
            error=f"index subprocess produced no result (exit {returncode})",
            hint=hint,
            code=errors.E_MODEL_LOAD_FAILED, finished_at=time.time(),
        )
        return
    if not data.get("ok"):
        if cancelled:
            _mark_cancelled(job_id)
            return
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


def _inprocess_cpu_index(
    job_id: str,
    project_path: str,
    full_rebuild: bool,
    git_max_commits: int | None = None,
    git_analytics: bool | None = None,
) -> None:
    """Optional in-process CPU fast path for very small deltas.

    Off by default (see ``ENGRAM_INPROCESS_CPU_MAX``/``_inprocess_cpu_max``).
    Only ``_index_worker`` decides to call this, after confirming the plan's
    ``missing_unique_chunks`` is at or below the configured threshold. Always
    builds a fresh, uncached provider (``factory.make_uncached_cpu_provider``)
    -- never the cached query provider -- so it cannot contend with the query
    path's inference lock. Unlike the subprocess path, a mid-batch cancel
    cannot interrupt this call; the job may finish (or fail) before a cancel
    request is observed. Callers that need reliable cancellation should leave
    this fast path disabled (the default).
    """
    provider = None
    try:
        provider = factory.make_uncached_cpu_provider()
        _registry.update(
            job_id,
            stage="loading_model",
            embedder_id=provider.model_id,
            backend_id=provider.backend_id,
        )

        def progress(event: dict) -> None:
            _apply_index_progress_event(job_id, event)

        resolved_git_analytics = (
            gitmeta.git_analytics_default() if git_analytics is None else bool(git_analytics)
        )
        stats = _run_index(
            project_path,
            provider,
            full_rebuild=full_rebuild,
            progress=progress,
            git_analytics=resolved_git_analytics,
            git_max_commits=git_max_commits,
        )
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


def _plan_payload(plan) -> dict:
    return {
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


def _index_worker(
    job_id: str,
    project_path: str,
    full_rebuild: bool,
    index_device: str,
    git_max_commits: int | None = None,
    git_analytics: bool | None = None,
) -> None:
    """Background body of an index job: plan, route, then embed.

    The walk/chunk/hash/cache-plan work (``plan_index``) used to run
    synchronously inside the ``index_project`` tool call, blocking it on a
    filesystem walk of the whole project before a job even existed. It now
    runs here instead, so ``start_index_job``/the MCP tool returns a job_id
    immediately and the plan (when computed) shows up as job progress.

    Embedding itself always happens in the short-lived index subprocess
    (``_subprocess_index``) unless the optional, off-by-default in-process CPU
    fast path (``ENGRAM_INPROCESS_CPU_MAX``) is enabled and the delta is small
    enough -- either way, this long-lived server process never embeds with
    the cached query provider.
    """
    _registry.update(
        job_id,
        status="running",
        stage="planning",
        index_device_requested=index_device,
        started_at=time.time(),
    )
    if _job_cancel_requested(job_id):
        _mark_cancelled(job_id)
        return

    routed_device = index_device
    plan = None
    inprocess_max = _inprocess_cpu_max()
    needs_plan = index_device == "auto" or (index_device == "cpu" and inprocess_max > 0)
    if needs_plan:
        try:
            plan = _plan_index(
                project_path,
                full_rebuild=full_rebuild,
                model_id=factory.CANONICAL_EMBEDDER_ID,
                dim=config.DEFAULT_EMBED_DIM,
            )
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
            return
        if index_device == "auto" and plan.missing_unique_chunks <= _delta_cpu_max():
            routed_device = "cpu"
        _registry.update(job_id, plan=_plan_payload(plan))

    if _job_cancel_requested(job_id):
        _mark_cancelled(job_id)
        return

    _registry.update(job_id, stage="loading_model", index_device=routed_device)

    use_inprocess = (
        routed_device == "cpu"
        and plan is not None
        and inprocess_max > 0
        and plan.missing_unique_chunks <= inprocess_max
    )
    if use_inprocess:
        _inprocess_cpu_index(job_id, project_path, full_rebuild, git_max_commits, git_analytics)
    else:
        _subprocess_index(job_id, project_path, full_rebuild, routed_device, git_max_commits, git_analytics)


# --- plain, testable logic (the MCP tools are thin wrappers over these) ---

def _delta_cpu_max() -> int:
    raw = os.environ.get("ENGRAM_DELTA_CPU_MAX", "").strip()
    if not raw:
        return _DEFAULT_DELTA_CPU_MAX
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_DELTA_CPU_MAX


def _inprocess_cpu_max() -> int:
    """Threshold (missing unique chunks) below which the optional in-process
    CPU fast path may run instead of the default subprocess. 0 (default)
    disables the fast path entirely. See ``ENGRAM_INPROCESS_CPU_MAX``."""
    raw = os.environ.get("ENGRAM_INPROCESS_CPU_MAX", "").strip()
    if not raw:
        return _DEFAULT_INPROCESS_CPU_MAX
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_INPROCESS_CPU_MAX


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
    git_max_commits: int | None = None,
    git_analytics: bool | None = None,
) -> dict:
    """Validate, coalesce, and hand off an index job; returns promptly.

    No filesystem walk happens here: the delta-aware plan (``plan_index``)
    that used to run synchronously in this function before a job existed now
    runs inside the background job body (``_index_worker``), so this call
    only ever does path validation plus in-memory registry bookkeeping before
    returning. Poll ``index_status`` for the plan/routing decision and
    progress.

    A second call for the same resolved project path while a job is already
    queued/running for it is coalesced onto that existing job (``coalesced``
    is ``True`` in the response) instead of starting a duplicate.
    """
    root = Path(project_path).expanduser()
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    device = factory.resolve_index_device(index_device, gpu=gpu)
    resolved = str(root.resolve())
    resolved_git_analytics = (
        gitmeta.git_analytics_default() if git_analytics is None else bool(git_analytics)
    )

    existing = _registry.find_active(resolved)
    if existing is not None:
        return {
            "job_id": existing.job_id,
            "project_path": resolved,
            "status": existing.status,
            "index_device": existing.index_device or device,
            "index_device_requested": device,
            "git_max_commits": git_max_commits,
            "git_analytics": resolved_git_analytics,
            "coalesced": True,
            "embedder_id": factory.CANONICAL_EMBEDDER_ID,
        }

    job = _registry.create(resolved)
    _registry.update(
        job.job_id,
        index_device=device,
        index_device_requested=device,
        embedder_id=factory.CANONICAL_EMBEDDER_ID,
    )
    _index_pool.submit(
        _index_worker,
        job.job_id,
        resolved,
        full_rebuild,
        device,
        git_max_commits,
        git_analytics,
    )
    return {
        "job_id": job.job_id,
        "project_path": resolved,
        "status": job.status,
        "index_device": device,
        "index_device_requested": device,
        "git_max_commits": git_max_commits,
        "git_analytics": resolved_git_analytics,
        "coalesced": False,
        "embedder_id": factory.CANONICAL_EMBEDDER_ID,
    }


def do_cancel_index(job_id: str) -> dict:
    """Cancel a running/queued index job; best-effort, idempotent.

    Kills the index subprocess's whole process tree (mirroring
    ``gitmeta.kill_process_tree``: a plain ``terminate()`` on the parent alone can
    leave grandchildren alive on Windows) so a killed child never reaches the
    atomic manifest/generation swap -- the previously published index is left
    untouched and searchable. A job already in a terminal state is reported
    as such rather than treated as an error.
    """
    job = _registry.get(job_id)
    if job is None:
        return {
            "error": (
                f"unknown job_id in this server process: {job_id}. "
                "Index jobs are tracked only in the current process."
            ),
            "code": errors.E_BAD_REQUEST,
            "scope": "current_process",
        }
    if job.is_terminal:
        return snapshot(job) | {"already_terminal": True}

    _registry.update(job_id, cancel_requested=True)
    with _job_processes_lock:
        proc = _job_processes.get(job_id)
    if proc is not None:
        gitmeta.kill_process_tree(proc)

    deadline = time.time() + _CANCEL_WAIT_SEC
    job = _registry.get(job_id)
    while job is not None and not job.is_terminal and time.time() < deadline:
        time.sleep(_CANCEL_POLL_INTERVAL_SEC)
        job = _registry.get(job_id)
    if job is None:
        return {
            "job_id": job_id,
            "error": "job disappeared from the registry while cancelling",
            "code": errors.E_BAD_REQUEST,
            "scope": "current_process",
        }
    return snapshot(job) | {"already_terminal": False}


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
    """Reject only malformed/sub-minimum ``k``.

    ``k`` above the server's ``MAX_SEARCH_K`` budget is not rejected here --
    ``query_service.search_project`` clamps it to that budget and reports the
    clamp as a warning in the response instead of failing the request. See
    ``query_service._validate_search_k``.
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be an integer >= 1 (server maximum is {MAX_SEARCH_K})")


def _check_content(content: str, max_chars_per_result: int) -> tuple[int, bool]:
    """Validate ``content``/``max_chars_per_result``, clamping only the over-budget side.

    Returns ``(effective_max_chars_per_result, clamped)``. A sub-minimum or
    malformed value is still rejected (degenerate request); a value above the
    server's ``_MAX_RESULT_CHARS`` response-character budget is clamped down
    instead, with the caller told via a response warning.
    """
    if content not in _CONTENT_MODES:
        raise ValueError("content must be one of: none, preview, full")
    if not isinstance(max_chars_per_result, int) or max_chars_per_result < 1:
        raise ValueError(
            f"max_chars_per_result must be an integer >= 1 (server maximum is {_MAX_RESULT_CHARS})"
        )
    if max_chars_per_result > _MAX_RESULT_CHARS:
        return _MAX_RESULT_CHARS, True
    return max_chars_per_result, False


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
        out["matched_in"] = _matched_in(hit, query)
        out["truncated"] = truncated
        return out, len(preview), truncated
    else:
        body, truncated = _clip_text(raw_content, budget)
        out["content"] = body
        out["matched_in"] = _matched_in(hit, query)
        out["truncated"] = truncated
        return out, len(body), truncated


def _validate_max_total_chars(value: int | None) -> tuple[int | None, bool]:
    """Validate ``max_total_chars``, clamping only the over-budget side.

    Returns ``(effective_value, clamped)``. See ``_check_content`` for the
    same sub-minimum-errors/over-budget-clamps split.
    """
    if value is None:
        return None, False
    if not isinstance(value, int) or value < 1:
        raise ValueError(
            f"max_total_chars must be an integer >= 1 (server maximum is {_MAX_TOTAL_CHARS})"
        )
    if value > _MAX_TOTAL_CHARS:
        return _MAX_TOTAL_CHARS, True
    return value, False


def _search_hints(outcome: dict, results: list[dict], content: str, max_total_chars: int | None) -> list[str]:
    hints: list[str] = []
    if not results:
        hints.append("No hits: try mode='hybrid' for exact tokens or find_definition for known symbols.")
    if (outcome.get("dirty") or {}).get("stale"):
        hints.append("Some results are stale: rebuild the index or reindex changed files.")
    if (outcome.get("source_revision") or {}).get("stale"):
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


def do_find_definition(project_path: str, symbol: str, ref: str | None = None) -> dict:
    try:
        root = Path(project_path).expanduser().resolve()
        return _run_find_def(root, symbol, include_suggestions=True, ref=ref)
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
    ref: str | None = None,
) -> dict:
    try:
        _check_k(k)
        requested_max_chars_per_result = max_chars_per_result
        max_chars_per_result, max_chars_per_result_clamped = _check_content(
            content, max_chars_per_result
        )
        requested_max_total_chars = max_total_chars
        max_total_chars, max_total_chars_clamped = _validate_max_total_chars(max_total_chars)
        if candidate_k is None:
            candidate_k = _rerank_candidate_k_default()
        # Sub-minimum/malformed candidate_k is still a bad request; a value
        # above MAX_RERANK_CANDIDATES is a budget question, not a validation
        # one -- query_service.search_project clamps it to the server maximum and
        # reports the clamp as a response warning instead of failing here.
        if not isinstance(candidate_k, int) or candidate_k < 1:
            raise ValueError(f"candidate_k must be an integer >= 1 (server maximum is {MAX_RERANK_CANDIDATES})")
        root = Path(project_path).expanduser().resolve()
        qi = load_query_index(root, ref=ref)
        provider = _provider_for_query_model(qi.manifest.embedder_id)
        outcome = _run_search(
            root,
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
            ref=ref,
            _query_index=qi,
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
        response_warnings = list(outcome.get("warnings") or [])
        if max_chars_per_result_clamped:
            response_warnings.append(
                f"max_chars_per_result clamped to server maximum {_MAX_RESULT_CHARS} "
                f"(requested {requested_max_chars_per_result})"
            )
        if max_total_chars_clamped:
            response_warnings.append(
                f"max_total_chars clamped to server maximum {_MAX_TOTAL_CHARS} "
                f"(requested {requested_max_total_chars})"
            )
        outcome["warnings"] = response_warnings
        return outcome | {
            "content": content,
            "max_chars_per_result": max_chars_per_result,
            "max_total_chars": max_total_chars,
            "body_chars": used_chars,
            "truncated": truncated,
            "count": len(results),
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
    prune_orphans: bool = False,
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
    try:
        return _run_project_map(
            Path(project_path).expanduser().resolve(),
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
            include_git=include_git,
            group_by=group_by,
            ticket_regex=ticket_regex,
            window_hours=window_hours,
            git_max_commits=git_max_commits,
            recent_days=recent_days,
            max_files_per_change=max_files_per_change,
            cochange_limit=cochange_limit,
            hotspots_limit=hotspots_limit,
        )
    except Exception as exc:
        return _error_payload(exc)


def do_doctor_project(project_path: str, check_git: bool = True) -> dict:
    try:
        return _run_doctor_project(Path(project_path).expanduser().resolve(), check_git=check_git)
    except Exception as exc:
        return _error_payload(exc)


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
    ref: str | None = None,
) -> dict:
    try:
        row = _run_get_chunk(
            Path(project_path).expanduser().resolve(),
            chunk_id,
            include_neighbors=include_neighbors,
            neighbor_window=neighbor_window,
            include_parent=include_parent,
            ref=ref,
        )
        content = row.get("content") or ""
        row["truncated"] = False
        if max_chars is not None:
            if not isinstance(max_chars, int) or max_chars < 1:
                raise ValueError(
                    f"max_chars must be an integer >= 1 (server maximum is {_MAX_RESULT_CHARS})"
                )
            # Over-budget max_chars is clamped (not rejected), same as
            # search_code's max_chars_per_result/max_total_chars.
            if max_chars > _MAX_RESULT_CHARS:
                row.setdefault("warnings", []).append(
                    f"max_chars clamped to server maximum {_MAX_RESULT_CHARS} (requested {max_chars})"
                )
                max_chars = _MAX_RESULT_CHARS
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
            "enabled": query_service.rerank_enabled(),
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
        },
        "source_type": "static_indexed_source",
    }


# --- MCP tool surface ---
#
# Tools are defined as plain coroutines and registered explicitly at the bottom
# so the mutating ones can be withheld in read-only mode (ENGRAM_READONLY=1).


async def index_project(
    project_path: str,
    full_rebuild: bool = False,
    index_device: str | None = None,
    git_max_commits: int | None = None,
    git_analytics: bool | None = None,
) -> dict:
    """Start a background index job and return its job_id immediately.

    Returns before the project walk/plan runs, not after it: planning and
    device routing happen inside the background job and show up via
    ``index_status`` (``stage="planning"``, then a ``plan`` object once
    computed). A second call for a project that already has a job
    queued/running returns that job's id with ``coalesced: true`` instead of
    starting a duplicate.

    ``git_max_commits`` optionally caps shared git-history analytics; omitted
    means full history. ``git_analytics`` controls whether git history/SZZ
    analytics are captured at all; omitted (``None``) defers to the
    ``ENGRAM_GIT_ANALYTICS`` env var (default: on). An explicit ``True``/
    ``False`` here always overrides the env var for this job.
    """
    return start_index_job(
        project_path,
        full_rebuild,
        index_device=index_device,
        git_max_commits=git_max_commits,
        git_analytics=git_analytics,
    )


async def index_status(job_id: str) -> dict:
    """Progress snapshot for an index job in this server process.

    Job tracking is in-memory and scoped to the current MCP process; completed
    on-disk indexes are visible through list_indexed_projects. In read-only
    mode no index jobs are produced, so this tool is inert except for jobs
    already started in this process before read-only registration.
    """
    return get_status(job_id)


async def cancel_index(job_id: str) -> dict:
    """Cancel a queued/running index job started in this process.

    Kills the index subprocess's whole process tree. A cancelled job ends in
    a terminal ``cancelled`` status; because the child is killed before it
    can reach the atomic manifest/generation swap, the previously published
    index (if any) is left fully intact and searchable. Calling this on a job
    that already finished (``done``/``error``/``cancelled``) is a no-op that
    reports the job's current terminal status rather than erroring.
    """
    return await asyncio.to_thread(do_cancel_index, job_id)


async def search_code(
    project_path: str, query: str, k: int = 8, language: str | None = None,
    mode: str = "auto", rerank: bool = False, content: str = "preview",
    max_chars_per_result: int = _DEFAULT_PREVIEW_CHARS,
    max_total_chars: int | None = None,
    candidate_k: int | None = None,
    facets: list[str] | None = None,
    min_relevance: str | None = None,
    ref: str | None = None,
) -> dict:
    """Search indexed source with compact bodies, facets, min relevance, and opt-in rerank.

    ``ref`` selects an already indexed git ref for the same logical project;
    a missing ref returns ``E_REF_NOT_INDEXED`` instead of falling back.
    """
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
        ref=ref,
    )


async def reindex_file(project_path: str, rel_path: str) -> dict:
    """Incrementally re-index (or drop, if missing) a single file on the index."""
    return await asyncio.to_thread(do_reindex_file, project_path, rel_path)


async def remove_project(project_path: str) -> dict:
    """Delete a project's on-disk index (vectors + manifests)."""
    return await asyncio.to_thread(do_remove_project, project_path)


async def find_definition(project_path: str, symbol: str, ref: str | None = None) -> dict:
    """Exact symbol lookup over static indexed source, with miss suggestions.

    Use this when you already know the symbol name (`symbol` or `Parent.symbol`).
    It does not load an embedding model. On an exact miss, suggestions contains
    nearby symbols from the indexed inventory. ``ref`` must already be indexed
    for the same logical project or the response is ``E_REF_NOT_INDEXED``.
    """
    return await asyncio.to_thread(do_find_definition, project_path, symbol, ref)


async def get_chunk(
    project_path: str,
    chunk_id: str,
    max_chars: int | None = None,
    include_neighbors: bool = False,
    neighbor_window: int = 1,
    include_parent: bool = False,
    ref: str | None = None,
) -> dict:
    """Fetch one chunk, optionally with adjacent/parent context.

    ``ref`` selects an already indexed git ref for the same logical project,
    matching ``search_code``/``find_definition``; a missing ref returns
    ``E_REF_NOT_INDEXED`` instead of silently hydrating against another index.
    """
    return await asyncio.to_thread(
        do_get_chunk,
        project_path,
        chunk_id,
        max_chars,
        include_neighbors,
        neighbor_window,
        include_parent,
        ref,
    )


async def project_map(
    project_path: str,
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
    """Body-free dirs map; opt into compact files, filters, and VCS analytics.

    ``include_git`` omitted defers to ``ENGRAM_GIT_ANALYTICS`` (default: on).
    A project indexed with git analytics disabled always reports
    ``git_analytics.status == "disabled"`` here rather than performing a live
    git walk, regardless of ``include_git``.
    """
    return await asyncio.to_thread(
        do_project_map,
        project_path=project_path,
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
        include_git=include_git,
        group_by=group_by,
        ticket_regex=ticket_regex,
        window_hours=window_hours,
        git_max_commits=git_max_commits,
        recent_days=recent_days,
        max_files_per_change=max_files_per_change,
        cochange_limit=cochange_limit,
        hotspots_limit=hotspots_limit,
    )


async def doctor_project(project_path: str, check_git: bool = True) -> dict:
    """Read-only index health check; does not load an embedding model."""
    return await asyncio.to_thread(do_doctor_project, project_path, check_git)


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
    prune_orphans: bool = False,
) -> dict:
    """List indexed projects with compact pagination and optional health checks.

    Compact mode reads manifests only. ``verbose=true`` may open LanceDB to add
    table health. ``prune_orphans`` defaults to false: a listing call must
    never delete an index merely because its recorded root looks momentarily
    missing (disconnected drive, unmounted share, renamed workspace). Passing
    ``prune_orphans=true`` here still works when explicitly requested, but the
    preferred way to delete orphaned index directories is the explicit
    operator path: `engram gc --prune` (or `engram gc --dry-run` to preview).
    """
    return list_projects(
        limit=limit,
        cursor=cursor,
        verbose=verbose,
        prune_orphans=prune_orphans,
    )


def read_only_enabled() -> bool:
    """True when ENGRAM_READONLY selects the read-only tool surface."""
    return paths.read_only_enabled()


def register_tools(read_only: bool) -> None:
    """Register the MCP tool surface; mutating tools are withheld when read_only."""
    # Read tools — always available.
    mcp.tool()(search_code)
    mcp.tool()(find_definition)
    mcp.tool()(get_chunk)
    mcp.tool()(project_map)
    mcp.tool()(doctor_project)
    mcp.tool()(model_status)
    mcp.tool()(index_status)
    mcp.tool()(list_indexed_projects)
    mcp.tool()(server_info)
    # Mutating tools — only when not read-only.
    if not read_only:
        mcp.tool()(index_project)
        mcp.tool()(cancel_index)
        mcp.tool()(reindex_file)
        mcp.tool()(remove_project)


register_tools(read_only_enabled())


def _maybe_warmup_on_start() -> None:
    if os.environ.get("ENGRAM_WARMUP_ON_START", "").strip().lower() in ("1", "true", "yes", "on"):
        _schedule_query_model_warmup(factory.CANONICAL_EMBEDDER_ID)


def _maybe_reclaim_on_start() -> None:
    """Kick off startup GC (stale generations + optional cache prune) in the background.

    Opt-in via ENGRAM_GC_ON_START=1; a no-op otherwise, and always under
    ENGRAM_READONLY=1. Backgrounded so a large ``ENGRAM_HOME`` (many indexed
    projects) never delays the stdio handshake. See `engram_mcp.startup` for
    why automatic reclaim is unsafe when several server processes share one
    ENGRAM_HOME.
    """

    def _run() -> None:
        from engram_mcp.startup import run_startup_maintenance

        try:
            run_startup_maintenance()
        except Exception:
            pass  # best-effort housekeeping; never let this affect server startup

    Thread(target=_run, name="engram-startup-gc", daemon=True).start()


def main() -> None:
    # Set up TLS trust (OS store / CA bundle / insecure) before the background
    # index worker loads a model over HTTPS.
    from engram_mcp.net import configure_tls

    configure_tls()
    _maybe_warmup_on_start()
    _maybe_reclaim_on_start()
    mcp.run()


if __name__ == "__main__":
    main()
