"""Read-side search, symbol lookup, and chunk hydration service."""

from __future__ import annotations

import os
import time
from difflib import SequenceMatcher
from pathlib import Path

from engram_mcp import catalog, config, errors, gitmeta, manifest, retrieval
from engram_mcp.embeddings.base import EmbeddingProvider
from engram_mcp.index_repository import (
    QueryIndex,
    derive_chunk_role,
    digest_mismatch,
    load_query_index,
    load_valid_catalog as _load_valid_catalog,
    read_text,
)
from engram_mcp.indexing.hash import sha256_text
from engram_mcp.indexing.languages import is_valid_language
from engram_mcp.store.lancedb_store import LanceStore


MAX_SEARCH_K = 50
MAX_RERANK_CANDIDATES = 50
DEFAULT_RERANK_CANDIDATE_K = 20
VECTOR_ESTIMATE_RELATIVE_FRACTION = 0.85


def _ensure_chunk_role(row: dict) -> dict:
    if row.get("chunk_role"):
        return row
    row["chunk_role"] = derive_chunk_role(
        row.get("rel_path"), row.get("language"), row.get("symbol_kind")
    )
    return row


def _validate_search_k(k: int) -> tuple[int, bool]:
    """Validate ``k``, clamping only the over-budget direction.

    A sub-minimum/malformed ``k`` (not an int, or < 1) is still rejected: it
    is a degenerate request, not a resource-budget concern. A ``k`` above
    ``MAX_SEARCH_K`` (the server's documented result-count budget) is instead
    clamped down to it -- the caller gets a bounded, still-useful result plus
    a warning explaining it was clamped, rather than an error. Returns
    ``(effective_k, clamped)``.
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be an integer >= 1 (server maximum is {MAX_SEARCH_K})")
    if k > MAX_SEARCH_K:
        return MAX_SEARCH_K, True
    return k, False


def rerank_enabled() -> bool:
    """Master switch for reranking. Off unless ENGRAM_RERANK_ENABLED is truthy.

    When off, a per-call ``rerank=true`` is ignored and NO reranker model is
    constructed or downloaded — a hard guarantee that the heavy ONNX cross-encoder
    never loads on the always-on server unless the operator opts in.
    """
    return os.environ.get("ENGRAM_RERANK_ENABLED", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def rerank_candidate_k_default() -> int:
    raw = os.environ.get("ENGRAM_RERANK_CANDIDATE_K", "").strip()
    if not raw:
        value = DEFAULT_RERANK_CANDIDATE_K
    else:
        try:
            value = int(raw)
        except ValueError:
            value = DEFAULT_RERANK_CANDIDATE_K
    return max(1, min(value, MAX_RERANK_CANDIDATES))


def freshness_for_hit(root: Path, files_meta: dict[str, dict], hit: dict) -> tuple[bool, str]:
    rel = hit.get("rel_path") or ""
    meta = files_meta.get(rel)
    if not meta:
        return True, "file is not present in files.json"
    abs_path = root / rel
    try:
        st = abs_path.stat()
    except OSError:
        return True, "file is missing from the working tree"
    if st.st_size == meta.get("size") and st.st_mtime_ns == meta.get("mtime_ns"):
        return False, "mtime and size match the indexed file"
    if st.st_size <= config.MAX_FILE_BYTES:
        text = read_text(abs_path)
        if text is not None and sha256_text(text) == meta.get("file_hash"):
            return False, "content hash matches despite metadata drift"
    return True, "working-tree file differs from the indexed file"


def _annotate_hits(
    root: Path,
    pdir: Path,
    hits: list[dict],
    query: str,
    mode_used: str,
    min_relevance: str | None,
) -> tuple[list[dict], dict, dict]:
    files_meta = manifest.load_files(pdir)
    annotated: list[dict] = []
    stale_paths: set[str] = set()
    for rank, hit in enumerate(hits, 1):
        h = _ensure_chunk_role(dict(hit))
        raw = float(h.get("score", 0.0))
        normalized = retrieval.normalize_score(
            raw, mode_used, rank, reranked=bool(h.get("reranked"))
        )
        relevance = retrieval.relevance_bucket(normalized)
        h["raw_score"] = raw
        h["score"] = raw
        h["score_normalized"] = normalized
        h["relevance"] = relevance
        h["matched"] = relevance in {"high", "medium"}
        h["match_reason"] = retrieval.match_reason(h, query, mode_used)
        stale, reason = freshness_for_hit(root, files_meta, h)
        h["stale"] = stale
        h["index_stale"] = stale
        h["freshness_reason"] = reason
        if stale:
            stale_paths.add(h.get("rel_path") or "")
        if retrieval.relevance_at_least(relevance, min_relevance):
            annotated.append(h)

    dirty = {
        "stale": bool(stale_paths),
        "stale_results": len(stale_paths),
        "stale_paths": sorted(p for p in stale_paths if p),
    }
    tail = {
        "tail_weak": any(
            h.get("relevance") in {"low", "uncertain"} for h in annotated[3:]
        ),
        "tail_weak_after_rank": 3 if len(annotated) > 3 else None,
    }
    if not tail["tail_weak"]:
        tail["tail_weak_after_rank"] = None
    return annotated, dirty, tail


def _parse_facets(facets: list[str] | tuple[str, ...] | None) -> list[str]:
    if not facets:
        return []
    out: list[str] = []
    for item in facets:
        if item not in catalog.SUPPORTED_FACETS:
            raise ValueError(
                "facets must be drawn from: " + ", ".join(sorted(catalog.SUPPORTED_FACETS))
            )
        if item not in out:
            out.append(item)
    return out


def _vector_similarity(hit: dict) -> float:
    if "_distance" in hit:
        distance = max(0.0, float(hit.get("_distance") or 0.0))
        return 1.0 / (1.0 + distance)
    raw = float(hit.get("score", 0.0) or 0.0)
    return max(0.0, raw)


def _vector_candidate_estimate(candidates: list[dict]) -> dict:
    if not candidates:
        return {
            "count": 0,
            "exact": False,
            "scope": "candidate_pool",
            "candidate_count": 0,
            "relative_score_fraction": VECTOR_ESTIMATE_RELATIVE_FRACTION,
        }
    sims = [_vector_similarity(h) for h in candidates]
    top = max(sims)
    threshold = top * VECTOR_ESTIMATE_RELATIVE_FRACTION
    return {
        "count": sum(1 for s in sims if s >= threshold),
        "exact": False,
        "scope": "candidate_pool",
        "candidate_count": len(candidates),
        "relative_score_fraction": VECTOR_ESTIMATE_RELATIVE_FRACTION,
        "top_similarity": round(top, 6),
        "threshold": round(threshold, 6),
    }


def _metadata_facet_counts(rows: list[dict], catalog_data: dict | None, requested: list[str]) -> dict:
    if not requested:
        return {}
    counts: dict[str, dict[str, int]] = {f: {} for f in requested}
    by_path = catalog.file_by_path(catalog_data) if catalog_data else {}

    def inc(field: str, value: str, n: int = 1) -> None:
        if field not in counts:
            return
        bucket = value or "(none)"
        counts[field][bucket] = counts[field].get(bucket, 0) + n

    seen_paths: set[str] = set()
    for row in rows:
        rel = row.get("rel_path") or ""
        file_meta = by_path.get(rel, {})
        if "chunk_role" in counts:
            inc("chunk_role", row.get("chunk_role") or "(none)")
        if rel in seen_paths:
            continue
        seen_paths.add(rel)
        if "dir" in counts:
            inc("dir", file_meta.get("dir") or catalog.directory_for(rel))
        if "language" in counts:
            inc("language", row.get("language") or file_meta.get("language") or "(none)")
        if "kind" in counts:
            for kind in file_meta.get("kinds") or catalog.derive_file_kinds(rel, row.get("language")):
                value = kind.get("kind") or ""
                if value:
                    inc("kind", value)
    return {field: dict(sorted(values.items())) for field, values in counts.items()}


def _search_count_metadata(
    qi: QueryIndex,
    query: str,
    where: str | None,
    mode_used: str,
    vector_candidates: list[dict],
    requested_facets: list[str],
    catalog_data: dict | None,
) -> tuple[dict, dict | None, list[str]]:
    warnings: list[str] = []
    total = {
        "fts_exact": {
            "available": False,
            "count": None,
            "exact": True,
            "capped": False,
            "method": "lancedb_0_33_fts_metadata_scan",
        },
        "vector_estimate": _vector_candidate_estimate(vector_candidates),
    }
    facet_payload: dict | None = None
    fts_rows: list[dict] = []
    fts_exact = False
    # The FTS metadata scan below is a second full pass over up to
    # ENGRAM_FTS_COUNT_MAX_SCAN matches (see lancedb_store.fts_metadata). It
    # exists to produce two things: total_matches.fts_exact (an exact/near-
    # exact match count) and, when requested_facets is non-empty, exact facet
    # counts. Neither is used unless the caller actually asked for facets --
    # so run the scan only then. When facets were not requested,
    # total_matches.fts_exact stays explicitly absent (available=False, with
    # a reason) rather than silently paying for a scan nobody reads; when
    # facets ARE requested this is byte-for-byte the same computation (and
    # the same total_matches.fts_exact numbers) as before this change.
    if mode_used == "hybrid" and requested_facets:
        fts_rows, warning, fts_meta = qi.store.fts_metadata(
            query,
            columns=("chunk_id", "rel_path", "language", "chunk_role"),
            where=where,
        )
        if warning:
            warnings.append(warning)
        else:
            capped = bool(fts_meta.get("capped"))
            fts_exact = not capped
            total["fts_exact"] = {
                "available": True,
                "count": len(fts_rows),
                "exact": fts_exact,
                "capped": capped,
                "method": "lancedb_0_33_fts_metadata_scan",
            }
            if capped:
                cap = fts_meta.get("cap", len(fts_rows))
                warnings.append(
                    f"FTS metadata scan hit ENGRAM_FTS_COUNT_MAX_SCAN={cap}; "
                    "total_matches.fts_exact.count is a lower bound."
                )
    elif mode_used == "hybrid":
        total["fts_exact"] = {
            "available": False,
            "count": None,
            "exact": False,
            "capped": False,
            "method": "lancedb_0_33_fts_metadata_scan",
            "reason": "not computed: pass facets=[...] to request an exact FTS match count",
        }
    if requested_facets:
        if mode_used == "hybrid" and total["fts_exact"]["available"]:
            facet_payload = {
                "scope": "fts_exact" if fts_exact else "fts_capped_lower_bound",
                "exact": fts_exact,
                "fields": _metadata_facet_counts(fts_rows, catalog_data, requested_facets),
            }
        else:
            estimated_rows = [
                row for row in vector_candidates
                if _vector_similarity(row) >= (
                    (total["vector_estimate"].get("threshold") or 0.0)
                )
            ]
            facet_payload = {
                "scope": "vector_candidate_estimate",
                "exact": False,
                "fields": _metadata_facet_counts(estimated_rows, catalog_data, requested_facets),
            }
    return total, facet_payload, warnings


def search_project(
    root: str | Path,
    provider: EmbeddingProvider,
    query: str,
    k: int = 8,
    language: str | None = None,
    mode: str = "auto",
    candidate_k: int | None = None,
    rerank: bool = False,
    facets: list[str] | tuple[str, ...] | None = None,
    min_relevance: str | None = None,
    return_meta: bool = False,
    ref: str | None = None,
    _query_index: QueryIndex | None = None,
) -> list[dict] | dict:
    """Search an indexed project.

    If ``ref`` is supplied, the ref must already have a matching index for the
    same logical git project; otherwise ``E_REF_NOT_INDEXED`` is raised.
    """

    requested_root = Path(root).resolve()
    requested_k = k
    k, k_clamped = _validate_search_k(k)
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if language is not None and not is_valid_language(language):
        raise ValueError(f"unknown language filter: {language!r}")
    if mode not in ("auto", "hybrid", "vector"):
        raise ValueError(f"unknown search mode: {mode!r}")
    requested_facets = _parse_facets(facets)
    if min_relevance is not None and min_relevance not in retrieval.RELEVANCE_ORDER:
        retrieval.relevance_at_least("low", min_relevance)  # raises the canonical message
    resolved_mode = retrieval.classify_query(query) if mode == "auto" else mode
    qi = _query_index or load_query_index(requested_root, ref=ref)
    root = qi.root
    m = qi.manifest
    if m.embedder_id and m.embedder_id != provider.model_id:
        raise errors.EngramError(
            f"index built with a different embedder ({m.embedder_id}); rebuild the index",
            errors.E_INDEX_INVALID,
            hint="Rebuild the index with `engram index --rebuild <project_path>`.",
        )
    if m.dim != provider.dim:
        raise errors.EngramError(
            f"index dimension {m.dim} does not match provider dimension {provider.dim}",
            errors.E_INDEX_INVALID,
            hint="Rebuild the index with the recorded embedder.",
        )
    if digest_mismatch(m.embedder_artifact_digest, getattr(provider, "artifact_digest", "")):
        raise errors.EngramError(
            "embedder artifact digest does not match the index's recorded digest "
            f"(index: {m.embedder_artifact_digest!r}, loaded: {provider.artifact_digest!r})",
            errors.E_INDEX_INVALID,
            hint="Rebuild the index with `engram index --rebuild <project_path>`.",
        )

    where = f"language = '{language}'" if language else None  # language is whitelisted above
    qv = provider.embed_queries([query])[0]
    if candidate_k is None:
        candidate_k = rerank_candidate_k_default()
    if not isinstance(candidate_k, int):
        raise ValueError("candidate_k must be an integer")
    requested_candidate_k = candidate_k
    # Over-budget candidate_k is clamped (not rejected) to the server's
    # documented MAX_RERANK_CANDIDATES budget; a below-k value is raised up to
    # k since a candidate pool smaller than the requested result count would
    # silently under-fill the response. Only the MAX_RERANK_CANDIDATES clamp
    # is reported as a "budget" warning below -- being raised to k is normal,
    # expected behavior, not a bounding action.
    candidate_k = max(k, min(candidate_k, MAX_RERANK_CANDIDATES))
    n = candidate_k
    warnings: list[str] = list(qi.resolution_warnings)
    if k_clamped:
        warnings.append(
            f"k clamped to server maximum {MAX_SEARCH_K} (requested {requested_k})"
        )
    if requested_candidate_k > MAX_RERANK_CANDIDATES:
        warnings.append(
            f"candidate_k clamped to server maximum {MAX_RERANK_CANDIDATES} "
            f"(requested {requested_candidate_k})"
        )
    mode_used = resolved_mode
    vector_candidates: list[dict] = []
    if resolved_mode == "hybrid":
        hits, meta = retrieval.hybrid_search(
            qi.store, query, qv, k=n, where=where, candidate_k=candidate_k, return_meta=True
        )
        warnings.extend(meta["warnings"])
        mode_used = meta["mode_used"]
        vector_candidates = [dict(h) for h in meta.get("vector_candidates", [])]
    else:
        hits = qi.store.search(qv, k=n, where=where)
        vector_candidates = [dict(h) for h in hits]
        for h in hits:
            _ensure_chunk_role(h)
            h["score"] = -float(h.get("_distance", 0.0)) + retrieval.role_boost(
                h.get("chunk_role")
            )
        hits = sorted(hits, key=lambda h: h.get("score", 0.0), reverse=True)
    rerank_applied = False
    rerank_model = None
    rerank_latency_ms = None
    rerank_skipped_reason = None
    if rerank and not rerank_enabled():
        # Master switch: reranking must be explicitly enabled by the operator.
        # Off by default so a stray rerank=true never loads the ONNX model.
        rerank_skipped_reason = "reranking disabled (set ENGRAM_RERANK_ENABLED=1 to enable)"
        warnings.append(f"rerank skipped: {rerank_skipped_reason}")
        hits = hits[:k]
    elif rerank and mode_used != "vector":
        # Skip hybrid rerank because measured quality regresses for exact
        # identifier/literal queries; callers can force mode="vector".
        rerank_skipped_reason = f"mode_used={mode_used} (rerank applies to vector mode only)"
        warnings.append(f"rerank skipped: {rerank_skipped_reason}")
        hits = hits[:k]
    elif rerank:
        t_rerank = time.time()
        try:
            from engram_mcp.rerankers import get_reranker

            reranker = get_reranker(backend="fastembed")
            hits = reranker.rerank(query, hits, top_k=k)
            rerank_applied = True
            rerank_model = getattr(reranker, "model_id", None)
        except Exception as exc:
            # Rerank is best-effort: on ANY failure (missing extra, model
            # download race, onnxruntime error) degrade to the base ranking
            # instead of failing the whole search.
            warnings.append(f"rerank unavailable, returning base ranking: {exc}")
            hits = hits[:k]
        finally:
            rerank_latency_ms = round((time.time() - t_rerank) * 1000, 3)
    else:
        hits = hits[:k]

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
        warnings.append(f"git metadata unavailable: {exc}")
    source_revision = gitmeta.source_revision_from_staleness(git_status)
    revision_warning = gitmeta.source_revision_warning(source_revision)
    if revision_warning:
        warnings.append(revision_warning)
    annotated, dirty, tail = _annotate_hits(
        root, qi.pdir, hits[:k], query, mode_used, min_relevance,
    )
    catalog_data, catalog_reason = _load_valid_catalog(qi)
    if catalog_data is None:
        if catalog_reason == "catalog sidecar unavailable":
            warnings.append("catalog sidecar unavailable; project_map/facets may require rebuild")
        else:
            warnings.append(
                f"catalog sidecar unavailable; project_map/facets may require rebuild ({catalog_reason})"
            )
    total_matches, facet_payload, count_warnings = _search_count_metadata(
        qi,
        query,
        where,
        mode_used,
        vector_candidates,
        requested_facets,
        catalog_data,
    )
    warnings.extend(count_warnings)
    if not return_meta:
        return annotated
    return {
        "query": query,
        "project_path": str(root),
        "requested_project_path": str(qi.requested_root or requested_root),
        "project_id": m.project_id,
        "logical_project_id": m.logical_project_id,
        "checkout_kind": m.checkout_kind,
        "indexed_ref": m.indexed_ref,
        "requested_ref": qi.requested_ref,
        "index_generation": m.generation,
        "embedder_id": m.embedder_id,
        "source_type": "static_indexed_source",
        "mode_requested": mode,
        "mode_used": mode_used,
        "warnings": warnings,
        "rerank_requested": rerank,
        "rerank_applied": rerank_applied,
        "rerank_skipped_reason": rerank_skipped_reason,
        "rerank_model": rerank_model,
        "rerank_latency_ms": rerank_latency_ms,
        "candidate_k": candidate_k,
        "facets_requested": requested_facets,
        "facets": facet_payload,
        "total_matches": total_matches,
        "dirty": dirty,
        "index_stale": dirty["stale"],
        "source_revision": source_revision,
        **tail,
        "hits": annotated,
    }


def _symbol_suggestions(store: LanceStore, name: str, limit: int = 8) -> list[dict]:
    needle = name.lower()
    seen: set[str] = set()
    suggestions: list[tuple[float, dict]] = []
    for row in store.symbol_inventory():
        sym = row.get("symbol") or ""
        if not sym or sym in seen:
            continue
        seen.add(sym)
        low = sym.lower()
        leaf = low.rsplit(".", 1)[-1]
        score = SequenceMatcher(None, needle, low).ratio()
        if leaf.startswith(needle) or low.startswith(needle):
            score += 0.45
        elif needle in low:
            score += 0.30
        elif needle in leaf:
            score += 0.20
        if score < 0.45:
            continue
        item = dict(row)
        item["score"] = round(min(score, 1.0), 3)
        suggestions.append((score, item))
    suggestions.sort(key=lambda x: (-x[0], x[1].get("symbol", "")))
    return [s for _, s in suggestions[:limit]]


def find_definition(
    root: str | Path,
    name: str,
    k: int = 20,
    include_suggestions: bool = False,
    ref: str | None = None,
) -> list[dict] | dict:
    """Exact symbol lookup (no embedding): definitions named `name` or `Parent.name`.

    Returns whole-symbol chunks (path + line range + content), preferring real
    definitions over module-level chunks. If ``ref`` is supplied and not
    indexed for the same logical git project, ``E_REF_NOT_INDEXED`` is raised.
    """
    if not name or not name.strip():
        raise ValueError("symbol must not be empty")
    qi = load_query_index(root, ref=ref)
    rows = qi.store.by_symbol(name, k=k)
    for row in rows:
        _ensure_chunk_role(row)
    defs = [r for r in rows if r.get("symbol_kind") not in ("module", "file")]
    results = defs or rows
    if not include_suggestions:
        return results
    return {
        "symbol": name,
        "project_path": str(qi.root),
        "requested_project_path": str(qi.requested_root or qi.root),
        "project_id": qi.manifest.project_id,
        "logical_project_id": qi.manifest.logical_project_id,
        "checkout_kind": qi.manifest.checkout_kind,
        "indexed_ref": qi.manifest.indexed_ref,
        "requested_ref": qi.requested_ref,
        "source_type": "static_indexed_source",
        "warnings": list(qi.resolution_warnings),
        "count": len(results),
        "results": results,
        "suggestions": [] if results else _symbol_suggestions(qi.store, name),
    }


def get_chunk(
    root: str | Path,
    chunk_id: str,
    *,
    include_neighbors: bool = False,
    neighbor_window: int = 1,
    include_parent: bool = False,
    ref: str | None = None,
) -> dict:
    """Fetch the full stored content for one chunk id.

    If ``ref`` is supplied, the ref must already have a matching index for the
    same logical git project; otherwise ``E_REF_NOT_INDEXED`` is raised rather
    than silently hydrating the chunk id against a different index.
    """

    if not chunk_id or not chunk_id.strip():
        raise ValueError("chunk_id must not be empty")
    qi = load_query_index(root, ref=ref)
    row = qi.store.by_chunk_id(chunk_id)
    if row is None:
        raise ValueError(f"unknown chunk_id: {chunk_id}")
    _ensure_chunk_role(row)
    stale, reason = freshness_for_hit(qi.root, manifest.load_files(qi.pdir), row)
    row["stale"] = stale
    row["freshness_reason"] = reason
    row["project_path"] = str(qi.root)
    row["requested_project_path"] = str(qi.requested_root or qi.root)
    row["project_id"] = qi.manifest.project_id
    row["index_generation"] = qi.manifest.generation
    row["indexed_ref"] = qi.manifest.indexed_ref
    row["requested_ref"] = qi.requested_ref
    row["source_type"] = "static_indexed_source"
    if include_neighbors or include_parent:
        data, reason = _load_valid_catalog(qi)
        if data is None:
            detail = f": {reason}" if reason else ""
            row["warnings"] = [f"catalog sidecar unavailable; cannot expand neighborhood{detail}"]
            return row
        lookup = catalog.chunk_lookup(data)
        current = lookup.get(chunk_id)
        if current is None:
            row["warnings"] = ["chunk not found in catalog sidecar; cannot expand neighborhood"]
            return row
        file_entry, chunk_entry, idx = current
        chunk_refs = file_entry.get("chunk_refs") or []
        neighbor_window = max(0, min(int(neighbor_window), 5))
        if include_neighbors and neighbor_window:
            start = max(0, idx - neighbor_window)
            end = min(len(chunk_refs), idx + neighbor_window + 1)
            neighbors = []
            for nidx in range(start, end):
                cref = chunk_refs[nidx]
                cid = cref.get("chunk_id")
                if not cid or cid == chunk_id:
                    continue
                body = qi.store.by_chunk_id(cid)
                if body is None:
                    continue
                _ensure_chunk_role(body)
                body["relative_position"] = nidx - idx
                neighbors.append(body)
            row["neighbors"] = neighbors
        if include_parent:
            symbol = chunk_entry.get("symbol") or ""
            parent_symbol = symbol.rsplit(".", 1)[0] if "." in symbol else ""
            parent = None
            if parent_symbol:
                for cref in chunk_refs:
                    if cref.get("symbol") == parent_symbol and cref.get("chunk_id") != chunk_id:
                        parent = qi.store.by_chunk_id(cref.get("chunk_id") or "")
                        if parent is not None:
                            _ensure_chunk_role(parent)
                        break
            row["parent"] = parent
    return row
