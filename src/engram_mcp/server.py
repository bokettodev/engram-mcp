"""MCP server (stdio) exposing async indexing + semantic search.

Tools:
  index_project(project_path)            -> {job_id, ...}   (async; poll status)
  index_status(job_id)                   -> progress snapshot
  search_code(project_path, query, ...)  -> ranked chunks
  get_chunk(project_path, chunk_id)      -> fetch one chunk's full content
  find_definition(project_path, symbol)  -> exact symbol lookup
  model_status(project_path?)            -> query-model load status
  reindex_file(project_path, rel_path)   -> incremental single-file re-index
  remove_project(project_path)           -> delete a project's index
  list_indexed_projects()                -> on-disk index inventory
  server_info()                          -> data-home/server diagnostics

Indexing runs on a single-worker background thread pool so a tool call returns
immediately and concurrent index requests serialize on the one embedder.

Read-only mode: set ENGRAM_READONLY=1 (env) to expose ONLY the read tools
(search_code / get_chunk / find_definition / model_status / index_status /
list_indexed_projects / server_info). The
mutating tools (index_project / reindex_file / remove_project) are not
registered, so a client physically cannot alter an index. Indexing is then
driven out-of-band (e.g. the `engram` CLI/operator). Intended for hosts that
hand the server to untrusted callers (agents) while a separate process owns
indexing.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from engram_mcp import errors
from engram_mcp import paths
from engram_mcp.embeddings import factory
from engram_mcp.jobs import JobRegistry, snapshot
from engram_mcp.pipeline import MAX_SEARCH_K, ProjectNotIndexedError
from engram_mcp.pipeline import find_definition as _run_find_def
from engram_mcp.pipeline import get_chunk as _run_get_chunk
from engram_mcp.pipeline import index_project as _run_index
from engram_mcp.pipeline import load_query_index
from engram_mcp.pipeline import reindex_file as _run_reindex_file
from engram_mcp.pipeline import remove_project as _run_remove
from engram_mcp.pipeline import search_project as _run_search

mcp = FastMCP("engram")
_registry = JobRegistry()
_index_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cidx-index")
_model_loads = {}
_model_loads_lock = Lock()

_RESULT_FIELDS = (
    "chunk_id", "rel_path", "start_line", "end_line", "language", "symbol", "symbol_kind",
    "chunk_role", "score", "raw_score", "score_normalized", "relevance", "matched",
    "match_reason", "stale", "freshness_reason",
)
_CONTENT_MODES = {"none", "preview", "full"}
_DEFAULT_PREVIEW_CHARS = 800
_MAX_RESULT_CHARS = 20_000
_MODEL_RETRY_AFTER_SEC = 2


def _get_provider(index_device: str | None = None):
    """Index-time provider for the selected backend."""
    return factory.make_index_provider(index_device)


def _index_worker(job_id: str, project_path: str, full_rebuild: bool, index_device: str) -> None:
    _registry.update(
        job_id,
        status="running",
        stage="loading-model",
        index_device=index_device,
        started_at=time.time(),
    )
    provider = None
    try:
        provider = _get_provider(index_device)
        _registry.update(
            job_id,
            stage="embedding",
            embedder_id=provider.model_id,
            backend_id=provider.backend_id,
        )

        def progress(done: int, total: int) -> None:
            _registry.update(job_id, done_units=done, total_units=total)

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
            finished_at=time.time(),
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
    finally:
        if provider is not None:
            try:
                factory.release_index_provider(provider)
            except Exception:  # never let cleanup turn a finished job into a failure
                pass


# --- plain, testable logic (the MCP tools are thin wrappers over these) ---

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
    job = _registry.create(resolved)
    _registry.update(job.job_id, index_device=device, embedder_id=factory.CANONICAL_EMBEDDER_ID)
    _index_pool.submit(_index_worker, job.job_id, resolved, full_rebuild, device)
    return {
        "job_id": job.job_id,
        "project_path": resolved,
        "status": job.status,
        "index_device": device,
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


def _provider_for_query_model(model_id: str):
    """Return a loaded provider or schedule warmup and raise E_MODEL_LOADING."""

    backend_id = factory.query_backend_id_for_model_id(model_id)
    if factory.is_model_loaded(backend_id):
        return factory.provider_for_model_id(model_id)
    with _model_loads_lock:
        fut = _model_loads.get(model_id)
        if fut is None:
            fut = _index_pool.submit(_provider_load_worker, model_id)
            _model_loads[model_id] = fut
    if fut.done():
        return fut.result()
    raise errors.EngramError(
        f"model for this index is loading: {model_id}",
        errors.E_MODEL_LOADING,
        hint="Retry the search after the reported delay.",
    )


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


def _format_search_hit(hit: dict, content: str, max_chars: int) -> dict:
    out = {field: hit.get(field) for field in _RESULT_FIELDS}
    out["span"] = {"start_line": hit.get("start_line"), "end_line": hit.get("end_line")}
    out["path"] = hit.get("rel_path")
    raw_content = hit.get("content") or ""
    if content == "none":
        out["truncated"] = False
    elif content == "preview":
        preview, truncated = _clip_text(raw_content, max_chars)
        out["preview"] = preview
        out["truncated"] = truncated
    else:
        body, truncated = _clip_text(raw_content, max_chars)
        out["content"] = body
        out["truncated"] = truncated
    return out


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
    min_relevance: str | None = None,
) -> dict:
    try:
        _check_k(k)
        _check_content(content, max_chars_per_result)
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
            rerank=rerank,
            min_relevance=min_relevance,
            return_meta=True,
        )
        results = [
            _format_search_hit(h, content, max_chars_per_result)
            for h in outcome.pop("hits")
        ]
        return outcome | {
            "content": content,
            "max_chars_per_result": max_chars_per_result,
            "count": len(results),
            "results": results,
        }
    except errors.EngramError as exc:
        extra = {}
        if exc.code == errors.E_MODEL_LOADING:
            extra["retry_after_sec"] = _MODEL_RETRY_AFTER_SEC
        return _error_payload(exc, results=[]) | extra
    except Exception as exc:
        return _error_payload(exc, results=[])


def list_projects() -> dict:
    home = paths.data_home(create=False)
    base = home / "projects"
    out = []
    errs = []
    if not home.exists():
        return {
            "data_home": str(home),
            "data_home_source": paths.data_home_source(),
            "home_exists": False,
            "projects_empty": True,
            "projects": [],
            "errors": [],
        }
    if not base.exists():
        return {
            "data_home": str(home),
            "data_home_source": paths.data_home_source(),
            "home_exists": True,
            "projects_empty": True,
            "projects": [],
            "errors": [],
        }
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        pj = d / "project.json"
        if not pj.is_file():
            errs.append(
                {
                    "project_id": d.name,
                    "manifest_path": str(pj),
                    "code": errors.E_INDEX_INVALID,
                    "error": "project manifest is missing",
                }
            )
            continue
        try:
            raw = json.loads(pj.read_text(encoding="utf-8"))
            root = raw.get("root_path")
            if not root:
                raise errors.EngramError(
                    "root_path is missing", errors.E_INDEX_INVALID
                )
            qi = load_query_index(root)
            out.append(raw | {
                "project_id": d.name,
                "table_rows": qi.count,
                "valid": True,
            })
        except (OSError, json.JSONDecodeError) as exc:
            errs.append(
                {
                    "project_id": d.name,
                    "manifest_path": str(pj),
                    "code": errors.E_INDEX_INVALID,
                    "error": f"invalid project manifest: {exc}",
                }
            )
        except Exception as exc:
            payload = _error_payload(exc)
            errs.append(
                {
                    "project_id": d.name,
                    "manifest_path": str(pj),
                    "code": payload.get("code", errors.E_INDEX_INVALID),
                    "error": payload.get("error", str(exc)),
                    **({"hint": payload["hint"]} if payload.get("hint") else {}),
                }
            )
    return {
        "data_home": str(home),
        "data_home_source": paths.data_home_source(),
        "home_exists": True,
        "projects_empty": not out and not errs,
        "projects": out,
        "errors": errs,
    }


def do_get_chunk(project_path: str, chunk_id: str, max_chars: int | None = None) -> dict:
    try:
        row = _run_get_chunk(Path(project_path).expanduser().resolve(), chunk_id)
        content = row.get("content") or ""
        row["truncated"] = False
        if max_chars is not None:
            if not isinstance(max_chars, int) or max_chars < 1 or max_chars > _MAX_RESULT_CHARS:
                raise ValueError(f"max_chars must be between 1 and {_MAX_RESULT_CHARS}")
            row["content"], row["truncated"] = _clip_text(content, max_chars)
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
        "source_type": "static_indexed_source",
    }


# --- MCP tool surface ---
#
# Tools are defined as plain coroutines and registered explicitly at the bottom
# so the mutating ones can be withheld in read-only mode (ENGRAM_READONLY=1).


async def index_project(
    project_path: str, full_rebuild: bool = False, gpu: bool = False
) -> dict:
    """Start a background index/re-index of a project directory.

    Returns a job_id immediately. Poll index_status until status == 'done'
    (or 'error'). Incremental by default (only changed files are touched);
    pass full_rebuild=true to rebuild the whole index atomically. By default
    indexing uses FastEmbed/ONNX on CPU. Pass gpu=true, or set
    ENGRAM_INDEX_DEVICE=cuda, to index once with sentence-transformers on CUDA;
    search still uses FastEmbed/ONNX on CPU.
    """
    return start_index_job(project_path, full_rebuild, gpu)


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
    min_relevance: str | None = None,
) -> dict:
    """Semantic search over static indexed source for choosing chunks to read.

    Use find_definition when you know a symbol. Use hybrid for identifiers,
    literals, action names, and paths; use vector for natural-language behavior
    questions. Returns compact hits by default: chunk_id, path/span, symbol,
    preview, relevance, freshness, mode metadata, and static-source warnings.
    content is "none", "preview" (default), or "full" bounded by
    max_chars_per_result; fetch exact full text with get_chunk(chunk_id). k is
    bounded to 1..50. rerank=true is best-effort and reports rerank_applied.
    """
    return await asyncio.to_thread(
        do_search,
        project_path,
        query,
        k,
        language,
        mode,
        rerank,
        content,
        max_chars_per_result,
        min_relevance,
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


async def get_chunk(project_path: str, chunk_id: str, max_chars: int | None = None) -> dict:
    """Fetch full static indexed source content for one search_code chunk_id."""
    return await asyncio.to_thread(do_get_chunk, project_path, chunk_id, max_chars)


async def model_status(project_path: str | None = None) -> dict:
    """Report whether a project's recorded query model is loaded in this process.

    This is read-only and does not start a model download. If a first search has
    scheduled warmup, status may be loading with retry_after_sec.
    """
    return await asyncio.to_thread(do_model_status, project_path)


async def server_info() -> dict:
    """Server configuration and data-home diagnostics for this process."""
    return do_server_info()


async def list_indexed_projects() -> dict:
    """List on-disk indexes, data_home, and broken manifest/table errors."""
    return list_projects()


def read_only_enabled() -> bool:
    """True when ENGRAM_READONLY selects the read-only tool surface."""
    return os.environ.get("ENGRAM_READONLY", "").strip().lower() in ("1", "true", "yes", "on")


def register_tools(read_only: bool) -> None:
    """Register the MCP tool surface; mutating tools are withheld when read_only."""
    # Read tools — always available.
    mcp.tool()(search_code)
    mcp.tool()(find_definition)
    mcp.tool()(get_chunk)
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


def main() -> None:
    # Set up TLS trust (OS store / CA bundle / insecure) before the background
    # index worker loads a model over HTTPS.
    from engram_mcp.net import configure_tls

    configure_tls()
    mcp.run()


if __name__ == "__main__":
    main()
