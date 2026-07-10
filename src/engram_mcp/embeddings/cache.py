"""Global content-hash embedding cache (SQLite).

Keyed by ``embedding_input_hash`` so identical chunks — across projects, fresh
clones, or re-indexes — are embedded once. Vectors are stored as float32 blobs.

The cache has no built-in size cap: unbounded growth is deliberate (deleting
an entry only costs a future re-embed, never correctness), but that means a
long-lived machine can accumulate a large file with nothing ever reclaiming
it. ``stats()``/``prune_to_budget()`` below give it a size, a report, and an
explicit, opt-in retention policy — see ``cache_max_bytes`` for the env knob
and why the default is unlimited.
"""

from __future__ import annotations

import os
import sqlite3
import time
from array import array
from pathlib import Path
from typing import Iterable, Mapping

from engram_mcp import paths

# No default budget: the cache is shared across every project/worktree on the
# machine, and evicting an entry is invisible until the *next* time that exact
# chunk needs embedding again (then it's just a cache miss, silently paid for
# again). Auto-deleting gigabytes of a user's accumulated cache the first time
# they upgrade would be a surprising, silent behavior change. So: unlimited
# unless an operator opts in via ENGRAM_CACHE_MAX_MB.
DEFAULT_CACHE_MAX_MB: int | None = None


def cache_max_mb() -> int | None:
    """Parse ``ENGRAM_CACHE_MAX_MB``: unset/blank/``0``/negative => unlimited (``None``)."""
    raw = os.environ.get("ENGRAM_CACHE_MAX_MB", "").strip()
    if not raw:
        return DEFAULT_CACHE_MAX_MB
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_CACHE_MAX_MB
    return value if value > 0 else None


def cache_max_bytes() -> int | None:
    mb = cache_max_mb()
    return None if mb is None else mb * 1024 * 1024


class EmbeddingCache:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        # busy_timeout + WAL so a concurrent CLI/server writer waits instead of
        # failing with "database is locked".
        self.conn = sqlite3.connect(str(db_path), timeout=30.0)
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (h TEXT PRIMARY KEY, dim INTEGER, vec BLOB, "
            "last_used_at REAL NOT NULL DEFAULT 0)"
        )
        self._migrate_last_used_at()
        self.conn.commit()

    def _migrate_last_used_at(self) -> None:
        """Add ``last_used_at`` to a pre-existing cache file (metadata-only ALTER)."""
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(embeddings)").fetchall()}
        if "last_used_at" not in cols:
            self.conn.execute(
                "ALTER TABLE embeddings ADD COLUMN last_used_at REAL NOT NULL DEFAULT 0"
            )

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def get_many(self, hashes: Iterable[str], dim: int | None = None) -> dict[str, list[float]]:
        hashes = list(hashes)
        out: dict[str, list[float]] = {}
        if not hashes:
            return out
        now = time.time()
        # SQLite caps variables per statement; chunk the IN-list.
        for i in range(0, len(hashes), 800):
            batch = hashes[i : i + 800]
            placeholders = ",".join("?" * len(batch))
            if dim is None:
                rows = self.conn.execute(
                    f"SELECT h, vec FROM embeddings WHERE h IN ({placeholders})", batch
                ).fetchall()
            else:
                rows = self.conn.execute(
                    f"SELECT h, vec FROM embeddings WHERE dim = ? AND h IN ({placeholders})",
                    [dim, *batch],
                ).fetchall()
            for h, blob in rows:
                a = array("f")
                a.frombytes(blob)
                out[h] = a.tolist()
            if rows:
                hit_hashes = [row[0] for row in rows]
                hit_placeholders = ",".join("?" * len(hit_hashes))
                self.conn.execute(
                    f"UPDATE embeddings SET last_used_at = ? WHERE h IN ({hit_placeholders})",
                    [now, *hit_hashes],
                )
        if out:
            self.conn.commit()
        return out

    def put_many(self, items: Mapping[str, list[float]]) -> None:
        if not items:
            return
        now = time.time()
        payload = [
            (h, len(vec), array("f", vec).tobytes(), now) for h, vec in items.items()
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO embeddings (h, dim, vec, last_used_at) VALUES (?, ?, ?, ?)",
            payload,
        )
        self.conn.commit()

    def _file_bytes(self) -> int:
        total = 0
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(self.db_path) + suffix)
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total

    def stats(self) -> dict:
        """Cheap size reporting: row count + on-disk bytes (main file + WAL/SHM)."""
        rows = self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        return {"rows": int(rows), "bytes": self._file_bytes()}

    def prune_to_budget(self, max_bytes: int | None) -> dict:
        """Evict least-recently-used rows until on-disk size is <= ``max_bytes``.

        ``max_bytes`` of ``None`` (or <= 0) means unlimited: a no-op that still
        reports current stats. Always VACUUMs (+ WAL-checkpoints) after an
        actual eviction so freed bytes are reflected in the file size, not just
        logically deleted rows. Never called from a search/index path — this
        is an explicit operator action (`engram gc`) or an opt-in startup task,
        gated on ``ENGRAM_CACHE_MAX_MB`` being set.
        """
        before = self.stats()
        result = {
            "before_rows": before["rows"],
            "before_bytes": before["bytes"],
            "after_rows": before["rows"],
            "after_bytes": before["bytes"],
            "pruned_rows": 0,
            "budget_bytes": max_bytes,
            "vacuumed": False,
        }
        if max_bytes is None or max_bytes <= 0:
            return result
        if before["rows"] == 0 or before["bytes"] <= max_bytes:
            return result
        avg_bytes_per_row = max(1, before["bytes"] // before["rows"])
        target_rows = max(0, max_bytes // avg_bytes_per_row)
        rows_to_delete = max(0, before["rows"] - target_rows)
        if rows_to_delete <= 0:
            return result
        cur = self.conn.execute(
            "DELETE FROM embeddings WHERE h IN "
            "(SELECT h FROM embeddings ORDER BY last_used_at ASC, h ASC LIMIT ?)",
            (rows_to_delete,),
        )
        self.conn.commit()
        self.conn.execute("VACUUM")
        self.conn.commit()
        # VACUUM is itself a large write transaction; in WAL mode that write
        # lands in -wal first. Checkpoint *after* it (not just before) so the
        # rewritten, compacted pages are folded back into the main file and
        # the -wal file is truncated -- otherwise `after` bytes can transiently
        # look larger than `before`, not smaller, defeating the whole point of
        # reporting freed bytes.
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.conn.commit()
        after = self.stats()
        result.update(
            {
                "after_rows": after["rows"],
                "after_bytes": after["bytes"],
                "pruned_rows": cur.rowcount if cur.rowcount and cur.rowcount > 0 else rows_to_delete,
                "vacuumed": True,
            }
        )
        return result

    def close(self) -> None:
        self.conn.close()


def _file_bytes_for(db_path: Path) -> int:
    total = 0
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(db_path) + suffix)
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def read_only_stats(db_path: str | Path) -> dict:
    """Stat the cache without ever writing to it — safe for a read tool.

    ``EmbeddingCache.__init__`` always runs a ``CREATE TABLE IF NOT EXISTS`` +
    column migration, which is a write even when it's a no-op; a read tool
    (``doctor_project``) must not do that. This opens the file (if any)
    strictly read-only at the OS level, so a failure here can never fall back
    to writing -- any surprise (missing table, WAL needing repair, concurrent
    writer) just degrades ``rows`` to ``None`` while the byte size, which
    needs no connection at all, is still reported.
    """
    db_path = Path(db_path)
    if not db_path.is_file():
        return {"path": str(db_path), "exists": False, "rows": None, "bytes": 0}
    bytes_on_disk = _file_bytes_for(db_path)
    rows: int | None = None
    try:
        uri = f"file:{db_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        try:
            row = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
            rows = int(row[0]) if row else 0
        finally:
            conn.close()
    except sqlite3.Error:
        rows = None
    return {"path": str(db_path), "exists": True, "rows": rows, "bytes": bytes_on_disk}


def global_cache_path() -> Path:
    return paths.global_cache_dir(create=False) / "embeddings.sqlite"


def global_cache_report(*, dry_run: bool = True, max_bytes: int | None = None) -> dict:
    """Report on (and, if not ``dry_run``, prune) the shared global embedding cache.

    Never prunes under ``ENGRAM_READONLY=1`` (degrades to a report). Never
    creates the cache file/dir merely to report on it -- a missing cache
    reports zeroed stats. ``max_bytes`` overrides ``cache_max_bytes()``
    (``ENGRAM_CACHE_MAX_MB``) for this call; ``None`` (the default) uses the
    configured budget, which is unlimited unless an operator opted in.
    """
    read_only = paths.read_only_enabled()
    effective_dry_run = dry_run or read_only
    db_path = global_cache_path()
    budget = cache_max_bytes() if max_bytes is None else (max_bytes if max_bytes > 0 else None)
    base = {
        "path": str(db_path),
        "budget_bytes": budget,
        "read_only": read_only,
    }
    if not db_path.is_file():
        return {**base, "exists": False, "dry_run": effective_dry_run, "rows": 0, "bytes": 0}
    if effective_dry_run:
        stats = read_only_stats(db_path)
        prunable = max(0, stats["bytes"] - budget) if budget is not None else 0
        return {
            **base,
            "exists": True,
            "dry_run": True,
            "rows": stats["rows"],
            "bytes": stats["bytes"],
            "prunable_bytes": prunable,
        }
    cache = EmbeddingCache(db_path)
    try:
        prune = cache.prune_to_budget(budget)
    finally:
        cache.close()
    return {**base, "exists": True, "dry_run": False, **prune}
