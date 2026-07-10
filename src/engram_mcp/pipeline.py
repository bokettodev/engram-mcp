"""Compatibility facade for Engram's focused repository and service seams."""

from __future__ import annotations

from pathlib import Path

from engram_mcp import diagnostics, index_repository as _index_repository
from engram_mcp.diagnostics import (
    DEFAULT_GREP_REGEX_TIMEOUT_SEC as DEFAULT_GREP_REGEX_TIMEOUT_SEC,
    GREP_WORKER_BATCH_ROWS as GREP_WORKER_BATCH_ROWS,
    MAX_GREP_LIMIT as MAX_GREP_LIMIT,
    MAX_GREP_MAX_MATCHES as MAX_GREP_MAX_MATCHES,
    MAX_GREP_SCAN_CHUNKS as MAX_GREP_SCAN_CHUNKS,
    doctor_project as doctor_project,
    grep_regex_timeout_seconds as grep_regex_timeout_seconds,
)
from engram_mcp.index_repository import (
    IndexPlan as IndexPlan,
    IndexStats as IndexStats,
    ProjectNotIndexedError as ProjectNotIndexedError,
    QueryIndex as QueryIndex,
    _full_rebuild as _full_rebuild,
    _incremental as _incremental,
    _is_compatible as _is_compatible,
    _load_files_for_indexing as _load_files_for_indexing,
    _rows as _rows,
    _save_catalog_from_rows as _save_catalog_from_rows,
    _search_text as _search_text,
    _strict_catalog_rows as _strict_catalog_rows,
    derive_chunk_role as derive_chunk_role,
    index_project as index_project,
    load_project_catalog as load_project_catalog,
    load_query_index as load_query_index,
    plan_index as plan_index,
    reindex_file as reindex_file,
    remove_project as remove_project,
)

from engram_mcp.query_service import (
    DEFAULT_RERANK_CANDIDATE_K as DEFAULT_RERANK_CANDIDATE_K,
    MAX_RERANK_CANDIDATES as MAX_RERANK_CANDIDATES,
    MAX_SEARCH_K as MAX_SEARCH_K,
    _search_count_metadata as _search_count_metadata,
    _validate_search_k as _validate_search_k,
    find_definition as find_definition,
    get_chunk as get_chunk,
    rerank_candidate_k_default as rerank_candidate_k_default,
    rerank_enabled as rerank_enabled,
    search_project as search_project,
)
from engram_mcp.structure_service import (
    MAX_GIT_COCHANGE_LIMIT as MAX_GIT_COCHANGE_LIMIT,
    MAX_GIT_HOTSPOTS_LIMIT as MAX_GIT_HOTSPOTS_LIMIT,
    MAX_GIT_MAX_FILES_PER_CHANGE as MAX_GIT_MAX_FILES_PER_CHANGE,
    _attach_git_analytics as _attach_git_analytics,
    project_map as project_map,
)

# Compatibility module/global names retained for existing callers and tests.
_catalog_deep_validation_error = _index_repository.catalog_deep_validation_error
_catalog_ref_count = _index_repository.catalog_ref_count
_catalog_validation_error = _index_repository.catalog_validation_error
_load_valid_catalog = _index_repository.load_valid_catalog
_digest_mismatch = _index_repository.digest_mismatch
_read_text = _index_repository.read_text


def _grep_rows_with_timeout(
    *,
    pattern: str,
    flags: int,
    rows: list[dict],
    include_lines: bool,
    max_matches: int,
    timeout_sec: float,
) -> dict:
    """Compatibility adapter preserving pipeline-level worker monkeypatching."""

    return diagnostics.grep_rows_with_timeout(
        pattern=pattern,
        flags=flags,
        rows=rows,
        include_lines=include_lines,
        max_matches=max_matches,
        timeout_sec=timeout_sec,
    )


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
) -> dict:
    """Compatibility adapter for the moved indexed-grep diagnostic."""

    return diagnostics.grep_index(
        root,
        pattern,
        ignore_case=ignore_case,
        limit=limit,
        offset=offset,
        max_matches=max_matches,
        max_scan_chunks=max_scan_chunks,
        include_lines=include_lines,
        _grep_runner=_grep_rows_with_timeout,
    )
