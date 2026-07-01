"""Global content-hash embedding cache (SQLite).

Keyed by ``embedding_input_hash`` so identical chunks — across projects, fresh
clones, or re-indexes — are embedded once. Vectors are stored as float32 blobs.
"""

from __future__ import annotations

import sqlite3
from array import array
from pathlib import Path
from typing import Iterable, Mapping


class EmbeddingCache:
    def __init__(self, db_path: str | Path):
        # busy_timeout + WAL so a concurrent CLI/server writer waits instead of
        # failing with "database is locked".
        self.conn = sqlite3.connect(str(db_path), timeout=30.0)
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (h TEXT PRIMARY KEY, dim INTEGER, vec BLOB)"
        )
        self.conn.commit()

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def get_many(self, hashes: Iterable[str], dim: int | None = None) -> dict[str, list[float]]:
        hashes = list(hashes)
        out: dict[str, list[float]] = {}
        if not hashes:
            return out
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
        return out

    def put_many(self, items: Mapping[str, list[float]]) -> None:
        if not items:
            return
        payload = [
            (h, len(vec), array("f", vec).tobytes()) for h, vec in items.items()
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO embeddings (h, dim, vec) VALUES (?, ?, ?)", payload
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
