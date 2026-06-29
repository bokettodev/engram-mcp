"""Index pipeline: atomic full rebuild + incremental update + hybrid search.

Full rebuild writes a fresh generation table (``chunks_g<N>``) and atomically
swaps the active pointer in ``project.json`` (old table dropped only after the
swap). Incremental updates touch only changed/added/deleted files. Embedding is
deduped via the global content-hash cache. Each chunk is embedded with a
contextual header (path/symbol/language) prepended; the raw content is kept
separately for display. All writers hold a per-project file lock.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from engram_mcp import config, manifest, paths, retrieval
from engram_mcp.embeddings.base import EmbeddingProvider
from engram_mcp.embeddings.cache import EmbeddingCache
from engram_mcp.indexing.chunker import chunk_file
from engram_mcp.indexing.hash import embedding_input_hash, sha256_text
from engram_mcp.indexing.languages import detect_language, is_valid_language
from engram_mcp.indexing.walker import looks_generated, walk
from engram_mcp.store.lancedb_store import LanceStore


class ProjectNotIndexedError(RuntimeError):
    """Raised when searching a project that has no index on disk yet."""


@dataclass(slots=True)
class IndexStats:
    files: int
    chunks: int
    embedded_unique: int
    reused_unique: int
    seconds: float
    chunks_per_sec: float
    mode: str = "full"  # "full" | "incremental"
    added: int = 0
    changed: int = 0
    deleted: int = 0
    unchanged: int = 0


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None


def _search_text(c) -> str:
    """The text actually embedded + full-text indexed: a contextual header
    (path / symbol / language) followed by the raw chunk content.

    (A richer header with file-level imports was tried and measured slightly
    WORSE on the eval set — the extra tokens diluted the signal — so the lean
    header stands.)"""
    header = [f"path: {c.rel_path}"]
    if c.symbol:
        header.append(f"symbol: {c.symbol}")
    if c.language:
        header.append(f"language: {c.language}")
    return "\n".join(header) + "\n\n" + c.text


def _is_compatible(m: manifest.ProjectManifest | None, provider: EmbeddingProvider) -> bool:
    return bool(
        m is not None
        and m.active_table
        and m.embedder_id == provider.model_id
        and m.dim == provider.dim
        and m.chunker_version == config.CHUNKER_VERSION
    )


def _embed(texts, provider, cache, batch_size, progress):
    """Embed only texts whose hash is not cached. Returns (vec_by_hash, hashes, n_new, n_reused)."""
    hashes = [embedding_input_hash(provider.model_id, config.CHUNKER_VERSION, t) for t in texts]
    cached = cache.get_many(set(hashes))
    missing: dict[str, str] = {}
    for t, h in zip(texts, hashes):
        if h not in cached and h not in missing:
            missing[h] = t
    new_vecs: dict[str, list[float]] = {}
    items = list(missing.items())
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        vecs = provider.embed_passages([t for _, t in batch])
        for (h, _), v in zip(batch, vecs):
            new_vecs[h] = v
        if progress:
            progress(min(i + batch_size, len(items)), len(items))
    cache.put_many(new_vecs)
    return {**cached, **new_vecs}, hashes, len(new_vecs), len(cached)


def _rows(chunks, search_texts, hashes, vec_by_hash, file_hash_by_path) -> list[dict]:
    out = []
    for idx, (c, st, h) in enumerate(zip(chunks, search_texts, hashes)):
        chash = sha256_text(c.text)
        out.append(
            {
                "chunk_id": f"{sha256_text(c.rel_path)[:12]}:{idx}:{chash[:8]}",
                "rel_path": c.rel_path,
                "language": c.language or "",
                "symbol": c.symbol or "",
                "symbol_kind": c.symbol_kind or "",
                "start_line": c.start_line,
                "end_line": c.end_line,
                "content": c.text,
                "search_text": st,
                "file_hash": file_hash_by_path.get(c.rel_path, ""),
                "chunk_hash": chash,
                "vector": vec_by_hash[h],
            }
        )
    return out


def index_project(
    root: str | Path,
    provider: EmbeddingProvider,
    *,
    full_rebuild: bool = False,
    batch_size: int = config.EMBED_BATCH_SIZE,
    progress: Callable[[int, int], None] | None = None,
) -> IndexStats:
    root = Path(root).resolve()
    pdir = paths.project_dir(root)
    with paths.project_lock(root):
        m = manifest.load_project(pdir)
        compatible = _is_compatible(m, provider) and (pdir / "files.json").is_file()
        if full_rebuild or not compatible:
            return _full_rebuild(root, pdir, provider, m, batch_size, progress)
        return _incremental(root, pdir, provider, m, batch_size, progress)


def _full_rebuild(root, pdir, provider, m, batch_size, progress) -> IndexStats:
    t0 = time.time()
    files_meta: dict[str, dict] = {}
    chunks = []
    for rec in walk(root):
        text = _read_text(rec.abs_path)
        if text is None or looks_generated(text, rec.language):
            continue
        cs = chunk_file(rec.rel_path, rec.language, text)
        chunks.extend(cs)
        files_meta[rec.rel_path] = {
            "file_hash": sha256_text(text), "mtime_ns": rec.mtime_ns, "size": rec.size,
            "language": rec.language or "", "chunks": len(cs),
        }
    file_hash_by_path = {p: meta["file_hash"] for p, meta in files_meta.items()}
    search_texts = [_search_text(c) for c in chunks]

    with EmbeddingCache(paths.global_cache_dir() / "embeddings.sqlite") as cache:
        vec_by_hash, hashes, embedded, reused = _embed(search_texts, provider, cache, batch_size, progress)
    rows = _rows(chunks, search_texts, hashes, vec_by_hash, file_hash_by_path)

    gen = (m.generation + 1) if m else 1
    new_table = f"chunks_g{gen}"
    db_dir = pdir / "lancedb"
    # GC tables left over from prior interrupted runs, but keep the current
    # active table so concurrent readers on the old pointer don't break.
    keep = {m.active_table} if m and m.active_table else set()
    LanceStore(db_dir, provider.dim).drop_stale_generations(keep)
    LanceStore(db_dir, provider.dim, table=new_table).create(rows)

    # Commit the pointer FIRST (it references the fully-built new table), then
    # the file manifest. A crash in between leaves the new table active + a
    # stale files manifest, which the next incremental reconciles cheaply.
    manifest.save_project(
        pdir,
        manifest.ProjectManifest(
            project_id=paths.project_id_for(root), root_path=str(root),
            active_table=new_table, generation=gen,
            embedder_id=provider.model_id, dim=provider.dim,
            chunker_version=config.CHUNKER_VERSION,
            files=len(files_meta), chunks=len(chunks), indexed_at=time.time(),
        ),
    )
    manifest.save_files(pdir, files_meta)
    # The previous active table is intentionally retained for in-flight readers;
    # it is GC'd at the start of the next rebuild.

    elapsed = time.time() - t0
    return IndexStats(
        files=len(files_meta), chunks=len(chunks), embedded_unique=embedded,
        reused_unique=reused, seconds=elapsed,
        chunks_per_sec=(len(chunks) / elapsed if elapsed > 0 else 0.0),
        mode="full", added=len(files_meta),
    )


def _incremental(root, pdir, provider, m, batch_size, progress) -> IndexStats:
    t0 = time.time()
    old_files = manifest.load_files(pdir)
    new_files: dict[str, dict] = {}
    touched = []
    added = changed = unchanged = 0

    for rec in walk(root):
        old = old_files.get(rec.rel_path)
        if old and old.get("size") == rec.size and old.get("mtime_ns") == rec.mtime_ns:
            new_files[rec.rel_path] = old
            unchanged += 1
            continue
        text = _read_text(rec.abs_path)
        if text is None or looks_generated(text, rec.language):
            # Skipped (unreadable/generated): not added to new_files, so if it
            # was previously indexed it falls into deleted_paths below.
            continue
        fh = sha256_text(text)
        if old and old.get("file_hash") == fh:
            entry = dict(old)
            entry["mtime_ns"] = rec.mtime_ns
            new_files[rec.rel_path] = entry
            unchanged += 1
            continue
        touched.append((rec, text, fh))
        changed += 1 if old else 0
        added += 0 if old else 1

    chunks = []
    file_hash_by_path: dict[str, str] = {}
    for rec, text, fh in touched:
        cs = chunk_file(rec.rel_path, rec.language, text)
        chunks.extend(cs)
        file_hash_by_path[rec.rel_path] = fh
        new_files[rec.rel_path] = {
            "file_hash": fh, "mtime_ns": rec.mtime_ns, "size": rec.size,
            "language": rec.language or "", "chunks": len(cs),
        }

    # Anything previously indexed but no longer kept (deleted, generated, or
    # unreadable) gets its rows removed.
    deleted_paths = [p for p in old_files if p not in new_files]

    embedded = reused = 0
    rows = []
    if chunks:
        search_texts = [_search_text(c) for c in chunks]
        with EmbeddingCache(paths.global_cache_dir() / "embeddings.sqlite") as cache:
            vec_by_hash, hashes, embedded, reused = _embed(search_texts, provider, cache, batch_size, progress)
        rows = _rows(chunks, search_texts, hashes, vec_by_hash, file_hash_by_path)

    store = LanceStore(pdir / "lancedb", provider.dim, table=m.active_table)
    store.delete_paths([rec.rel_path for rec, _, _ in touched] + deleted_paths)
    store.add(rows)
    if touched or deleted_paths:
        store.refresh_fts()

    manifest.save_files(pdir, new_files)
    m.files = len(new_files)
    m.chunks = store.count()
    m.indexed_at = time.time()
    manifest.save_project(pdir, m)

    elapsed = time.time() - t0
    return IndexStats(
        files=len(new_files), chunks=m.chunks, embedded_unique=embedded,
        reused_unique=reused, seconds=elapsed,
        chunks_per_sec=(len(chunks) / elapsed if elapsed > 0 else 0.0),
        mode="incremental", added=added, changed=changed,
        deleted=len(deleted_paths), unchanged=unchanged,
    )


def reindex_file(root: str | Path, provider: EmbeddingProvider, rel_path: str) -> dict:
    """Re-index (or drop) a single file on the active table — no full walk."""
    root = Path(root).resolve()
    pdir = paths.project_dir(root)
    with paths.project_lock(root):
        m = manifest.load_project(pdir)
        if not _is_compatible(m, provider):
            raise ProjectNotIndexedError(f"project not indexed (or incompatible): {root}")
        abs_path = (root / rel_path).resolve()
        try:
            rel = abs_path.relative_to(root).as_posix()
        except ValueError:
            raise ValueError(f"path escapes project root: {rel_path}")

        old_files = manifest.load_files(pdir)
        store = LanceStore(pdir / "lancedb", provider.dim, table=m.active_table)
        store.delete_paths([rel])
        chunks = []
        # Apply the same admission rules as a normal walk (skip symlink, binary,
        # oversized, unreadable, generated); a rejected file is just dropped.
        text = None
        lang = detect_language(abs_path.suffix)
        if abs_path.is_file() and not abs_path.is_symlink():
            ext = abs_path.suffix.lower()
            try:
                size = abs_path.stat().st_size
            except OSError:
                size = 0
            if ext not in config.BINARY_EXTS and 0 < size <= config.MAX_FILE_BYTES:
                candidate = _read_text(abs_path)
                if candidate is not None and not looks_generated(candidate, lang):
                    text = candidate
        if text is not None:
            fh = sha256_text(text)
            chunks = chunk_file(rel, lang, text)
            if chunks:
                search_texts = [_search_text(c) for c in chunks]
                with EmbeddingCache(paths.global_cache_dir() / "embeddings.sqlite") as cache:
                    vec_by_hash, hashes, _, _ = _embed(search_texts, provider, cache, config.EMBED_BATCH_SIZE, None)
                store.add(_rows(chunks, search_texts, hashes, vec_by_hash, {rel: fh}))
            st = abs_path.stat()
            old_files[rel] = {
                "file_hash": fh, "mtime_ns": st.st_mtime_ns, "size": st.st_size,
                "language": lang or "", "chunks": len(chunks),
            }
            action = "reindexed"
        else:
            old_files.pop(rel, None)
            action = "deleted"

        store.refresh_fts()
        manifest.save_files(pdir, old_files)
        m.files = len(old_files)
        m.chunks = store.count()
        m.indexed_at = time.time()
        manifest.save_project(pdir, m)
        return {"rel_path": rel, "action": action, "chunks": len(chunks)}


def search_project(
    root: str | Path,
    provider: EmbeddingProvider,
    query: str,
    k: int = 8,
    language: str | None = None,
    mode: str = "auto",
    candidate_k: int = 50,
    rerank: bool = False,
) -> list[dict]:
    root = Path(root).resolve()
    if language is not None and not is_valid_language(language):
        raise ValueError(f"unknown language filter: {language!r}")
    if mode not in ("auto", "hybrid", "vector"):
        raise ValueError(f"unknown search mode: {mode!r}")
    resolved_mode = retrieval.classify_query(query) if mode == "auto" else mode
    pdir = paths.project_dir(root, create=False)
    if not pdir.exists():
        raise ProjectNotIndexedError(f"project not indexed: {root}")
    m = manifest.load_project(pdir)
    if m is None or not m.active_table:
        raise ProjectNotIndexedError(f"project not indexed: {root}")
    if m.embedder_id and m.embedder_id != provider.model_id:
        raise ValueError(
            f"index built with a different embedder ({m.embedder_id}); reindex with this profile"
        )
    store = LanceStore(pdir / "lancedb", m.dim or provider.dim, table=m.active_table)
    if store.count() == 0:
        raise ProjectNotIndexedError(f"project not indexed: {root}")

    where = f"language = '{language}'" if language else None  # language is whitelisted above
    qv = provider.embed_queries([query])[0]
    n = max(k, candidate_k) if rerank else k
    if resolved_mode == "hybrid":
        hits = retrieval.hybrid_search(store, query, qv, k=n, where=where, candidate_k=candidate_k)
    else:
        hits = store.search(qv, k=n, where=where)
        for h in hits:
            h["score"] = -float(h.get("_distance", 0.0))
    if rerank:
        try:
            from engram_mcp.rerankers import get_reranker

            hits = get_reranker().rerank(query, hits, top_k=k)
        except ImportError as exc:  # surface as the normal handled error shape
            raise ValueError(str(exc)) from exc
    return hits[:k]


def find_definition(root: str | Path, name: str, k: int = 20) -> list[dict]:
    """Exact symbol lookup (no embedding): definitions named `name` or `Parent.name`.

    Returns whole-symbol chunks (path + line range + content), preferring real
    definitions over module-level chunks.
    """
    root = Path(root).resolve()
    pdir = paths.project_dir(root, create=False)
    m = manifest.load_project(pdir) if pdir.exists() else None
    if m is None or not m.active_table:
        raise ProjectNotIndexedError(f"project not indexed: {root}")
    store = LanceStore(pdir / "lancedb", m.dim or 0, table=m.active_table)
    rows = store.by_symbol(name, k=k)
    defs = [r for r in rows if r.get("symbol_kind") not in ("module", "file")]
    return defs or rows


def remove_project(root: str | Path) -> bool:
    root = Path(root).resolve()
    pdir = paths.project_dir(root, create=False)
    if not pdir.exists():
        return False
    # The lock lives outside the project dir, so we can hold it through the
    # delete; surface rmtree failures instead of silently "succeeding".
    with paths.project_lock(root):
        shutil.rmtree(pdir)
    return True
