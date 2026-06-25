"""LanceDB-backed vector store (embedded, on-disk, no server).

The table name is explicit so a full rebuild can write a fresh generation table
(`chunks_g<N>`) and the project manifest swaps the active pointer atomically.
Incremental updates delete a file's rows by ``rel_path`` and add the new ones.
"""

from __future__ import annotations

from pathlib import Path

import lancedb
import pyarrow as pa


def _sql_str(value: str) -> str:
    """Quote a string as a SQL literal (single quotes doubled)."""
    return "'" + value.replace("'", "''") + "'"


class LanceStore:
    def __init__(self, db_dir: str | Path, dim: int, table: str = "chunks"):
        self.db = lancedb.connect(str(db_dir))
        self.dim = dim
        self.table = table
        self._schema = pa.schema(
            [
                ("chunk_id", pa.string()),
                ("rel_path", pa.string()),
                ("language", pa.string()),
                ("symbol", pa.string()),
                ("symbol_kind", pa.string()),
                ("start_line", pa.int32()),
                ("end_line", pa.int32()),
                ("content", pa.string()),  # raw text for display
                ("search_text", pa.string()),  # contextual header + content, for embedding + FTS
                ("file_hash", pa.string()),
                ("chunk_hash", pa.string()),
                ("vector", pa.list_(pa.float32(), dim)),
            ]
        )

    def exists(self) -> bool:
        # list_tables() returns a ListTablesResponse; the names live in .tables
        return self.table in self.db.list_tables().tables

    def _open(self):
        if self.exists():
            return self.db.open_table(self.table)
        return self.db.create_table(self.table, schema=self._schema)

    def create(self, rows: list[dict]) -> None:
        """(Re)create this table from scratch with ``rows`` and an FTS index."""
        if self.exists():
            self.db.drop_table(self.table)
        tbl = self.db.create_table(self.table, schema=self._schema)
        if rows:
            tbl.add(rows)
            self.refresh_fts()

    def refresh_fts(self, field: str = "search_text") -> None:
        """(Re)build the full-text index used by hybrid search."""
        if self.exists() and self.count() > 0:
            try:
                self.db.open_table(self.table).create_fts_index(
                    field, replace=True, use_tantivy=False
                )
            except Exception:
                pass  # FTS is best-effort; vector search still works without it

    def search_text(self, query: str, k: int = 8, where: str | None = None) -> list[dict]:
        if not self.exists():
            return []
        try:
            q = self.db.open_table(self.table).search(query, query_type="fts").limit(k)
            if where:
                q = q.where(where)
            return q.to_list()
        except Exception:
            return []  # no FTS index yet -> caller falls back to vector-only

    def add(self, rows: list[dict]) -> None:
        if not rows:
            return
        self._open().add(rows)

    def delete_paths(self, rel_paths: list[str]) -> None:
        if not rel_paths or not self.exists():
            return
        clause = " OR ".join(f"rel_path = {_sql_str(p)}" for p in rel_paths)
        self.db.open_table(self.table).delete(clause)

    def drop(self) -> None:
        if self.exists():
            self.db.drop_table(self.table)

    def drop_stale_generations(self, keep: set[str]) -> None:
        """Drop leftover ``chunks*`` tables not in ``keep`` (post-swap cleanup)."""
        try:
            names = self.db.list_tables().tables
        except Exception:
            return
        for name in names:
            if name.startswith("chunks") and name not in keep:
                try:
                    self.db.drop_table(name)
                except Exception:
                    pass

    def count(self) -> int:
        return self.db.open_table(self.table).count_rows() if self.exists() else 0

    def by_symbol(self, name: str, k: int = 20) -> list[dict]:
        """Metadata lookup: rows whose symbol is exactly `name` or `Parent.name`."""
        if not self.exists():
            return []
        safe = name.replace("'", "''")
        # Escape LIKE wildcards (_ and %) so they match literally, not as patterns.
        like = (
            name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_").replace("'", "''")
        )
        pred = f"symbol = '{safe}' OR symbol LIKE '%.{like}' ESCAPE '\\'"
        try:
            rows = self.db.open_table(self.table).search().where(pred).limit(k).to_list()
        except Exception:
            return []
        for r in rows:  # drop the bulky embed-side columns
            r.pop("vector", None)
            r.pop("search_text", None)
        return rows

    def search(self, vector: list[float], k: int = 8, where: str | None = None) -> list[dict]:
        if not self.exists():
            return []
        q = self.db.open_table(self.table).search(vector).limit(k)
        if where:
            q = q.where(where)
        return q.to_list()
