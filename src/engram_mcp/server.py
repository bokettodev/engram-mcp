"""MCP server (stdio) exposing async indexing + semantic search.

Tools:
  index_project(project_path)            -> {job_id, ...}   (async; poll status)
  index_status(job_id)                   -> progress snapshot
  search_code(project_path, query, ...)  -> ranked chunks
  find_definition(project_path, symbol)  -> exact symbol lookup
  reindex_file(project_path, rel_path)   -> incremental single-file re-index
  remove_project(project_path)           -> delete a project's index
  list_indexed_projects()                -> on-disk index inventory

Indexing runs on a single-worker background thread pool so a tool call returns
immediately and concurrent index requests serialize on the one embedder.

Read-only mode: set ENGRAM_READONLY=1 (env) to expose ONLY the read tools
(search_code / find_definition / index_status / list_indexed_projects). The
mutating tools (index_project / reindex_file / remove_project) are not
registered, so a client physically cannot alter an index. Indexing is then
driven out-of-band (e.g. the `engram` CLI). Intended for hosts that hand the
server to untrusted callers (agents) while a separate process owns indexing.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from engram_mcp import paths
from engram_mcp.embeddings import factory
from engram_mcp.jobs import JobRegistry, snapshot
from engram_mcp.pipeline import ProjectNotIndexedError
from engram_mcp.pipeline import find_definition as _run_find_def
from engram_mcp.pipeline import index_project as _run_index
from engram_mcp.pipeline import reindex_file as _run_reindex_file
from engram_mcp.pipeline import remove_project as _run_remove
from engram_mcp.pipeline import search_project as _run_search

mcp = FastMCP("engram")
_registry = JobRegistry()
_index_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cidx-index")

_RESULT_FIELDS = (
    "rel_path", "start_line", "end_line", "language", "symbol", "symbol_kind", "content",
)


def _get_provider(profile: str = factory.DEFAULT_PROFILE):
    """Index-time provider for a profile (model instances cached by the factory)."""
    return factory.make_provider(profile)


def _index_worker(job_id: str, project_path: str, full_rebuild: bool, profile: str) -> None:
    _registry.update(job_id, status="running", stage="loading-model", started_at=time.time())
    try:
        provider = _get_provider(profile)
        _registry.update(job_id, stage="embedding")

        def progress(done: int, total: int) -> None:
            _registry.update(job_id, done_units=done, total_units=total)

        stats = _run_index(project_path, provider, full_rebuild=full_rebuild, progress=progress)
        _registry.update(
            job_id,
            status="done",
            stage="done",
            files=stats.files,
            chunks=stats.chunks,
            embedded=stats.embedded_unique,
            reused=stats.reused_unique,
            finished_at=time.time(),
        )
    except Exception as exc:  # surface failure via status, never crash the server
        _registry.update(
            job_id, status="error", stage="error", error=repr(exc), finished_at=time.time()
        )


# --- plain, testable logic (the MCP tools are thin wrappers over these) ---

def start_index_job(
    project_path: str, full_rebuild: bool = False, profile: str | None = None
) -> dict:
    root = Path(project_path).expanduser()
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    profile = profile or factory.DEFAULT_PROFILE
    if profile not in factory.PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choices: {', '.join(factory.PROFILES)}")
    resolved = str(root.resolve())
    job = _registry.create(resolved)
    _index_pool.submit(_index_worker, job.job_id, resolved, full_rebuild, profile)
    return {"job_id": job.job_id, "project_path": resolved, "status": job.status}


def do_reindex_file(project_path: str, rel_path: str) -> dict:
    root = Path(project_path).expanduser().resolve()
    provider = factory.provider_for_project(root)
    try:
        return _run_reindex_file(root, provider, rel_path)
    except (ProjectNotIndexedError, ValueError) as exc:
        return {"error": str(exc)}


def do_remove_project(project_path: str) -> dict:
    root = Path(project_path).expanduser().resolve()
    return {"removed": _run_remove(root), "project_path": str(root)}


def do_find_definition(project_path: str, symbol: str) -> dict:
    root = Path(project_path).expanduser().resolve()
    try:
        rows = _run_find_def(root, symbol)
    except ProjectNotIndexedError as exc:
        return {"error": str(exc), "results": []}
    return {"symbol": symbol, "count": len(rows), "results": rows}


def get_status(job_id: str) -> dict:
    job = _registry.get(job_id)
    if job is None:
        return {"error": f"unknown job_id: {job_id}"}
    return snapshot(job)


def do_search(
    project_path: str, query: str, k: int = 8, language: str | None = None,
    mode: str = "auto", rerank: bool = False,
) -> dict:
    root = Path(project_path).expanduser().resolve()
    provider = factory.provider_for_project(root)
    try:
        hits = _run_search(root, provider, query, k=k, language=language, mode=mode, rerank=rerank)
    except ProjectNotIndexedError:
        return {"error": f"project not indexed: {root}. Call index_project first.", "results": []}
    except ValueError as exc:
        return {"error": str(exc), "results": []}
    results = [
        {field: h.get(field) for field in _RESULT_FIELDS} | {"score": h.get("score")}
        for h in hits
    ]
    return {"query": query, "count": len(results), "results": results}


def list_projects() -> dict:
    base = paths.data_home() / "projects"
    out = []
    if base.is_dir():
        for d in sorted(base.iterdir()):
            pj = d / "project.json"
            if pj.is_file():
                try:
                    out.append(json.loads(pj.read_text(encoding="utf-8")) | {"project_id": d.name})
                except (OSError, json.JSONDecodeError):
                    continue
    return {"projects": out}


# --- MCP tool surface ---
#
# Tools are defined as plain coroutines and registered explicitly at the bottom
# so the mutating ones can be withheld in read-only mode (ENGRAM_READONLY=1).


async def index_project(
    project_path: str, full_rebuild: bool = False, profile: str | None = None
) -> dict:
    """Start a background index/re-index of a project directory.

    Returns a job_id immediately. Poll index_status until status == 'done'
    (or 'error'). Incremental by default (only changed files are touched);
    pass full_rebuild=true to rebuild the whole index atomically. profile is
    one of local_fast (CPU, default), local_quality (bge-large), or the
    Qwen3 quality profiles local_qwen_small / local_qwen (need the gpu extra).
    """
    return start_index_job(project_path, full_rebuild, profile)


async def index_status(job_id: str) -> dict:
    """Progress snapshot for an index job: status, stage, counts, ETA."""
    return get_status(job_id)


async def search_code(
    project_path: str, query: str, k: int = 8, language: str | None = None,
    mode: str = "auto", rerank: bool = False,
) -> dict:
    """Semantic search over an indexed project.

    Returns up to k ranked code chunks (path, line range, symbol, content).
    mode is "auto" (default: identifier/literal queries use hybrid full-text +
    vector, natural-language queries use vector), or force "vector"/"hybrid".
    rerank=true applies a cross-encoder reranker (needs the `gpu` extra).
    The project must be indexed first via index_project.
    """
    return await asyncio.to_thread(do_search, project_path, query, k, language, mode, rerank)


async def reindex_file(project_path: str, rel_path: str) -> dict:
    """Incrementally re-index (or drop, if missing) a single file on the index."""
    return await asyncio.to_thread(do_reindex_file, project_path, rel_path)


async def remove_project(project_path: str) -> dict:
    """Delete a project's on-disk index (vectors + manifests)."""
    return await asyncio.to_thread(do_remove_project, project_path)


async def find_definition(project_path: str, symbol: str) -> dict:
    """Exact symbol lookup (no embedding): the definition(s) named `symbol`
    (or `Parent.symbol`), returned as whole-symbol chunks with line ranges."""
    return await asyncio.to_thread(do_find_definition, project_path, symbol)


async def list_indexed_projects() -> dict:
    """List projects that currently have an index on disk."""
    return list_projects()


def read_only_enabled() -> bool:
    """True when ENGRAM_READONLY selects the read-only tool surface."""
    return os.environ.get("ENGRAM_READONLY", "").strip().lower() in ("1", "true", "yes", "on")


def register_tools(read_only: bool) -> None:
    """Register the MCP tool surface; mutating tools are withheld when read_only."""
    # Read tools — always available.
    mcp.tool()(search_code)
    mcp.tool()(find_definition)
    mcp.tool()(index_status)
    mcp.tool()(list_indexed_projects)
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
