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
                ("chunk_role", pa.string()),
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

    def create(self, rows: list[dict], *, refresh_fts: bool = True) -> None:
        """(Re)create this table from scratch with ``rows`` and an FTS index."""
        if self.exists():
            self.db.drop_table(self.table)
        tbl = self.db.create_table(self.table, schema=self._schema)
        if rows:
            tbl.add(rows)
            if refresh_fts:
                self.refresh_fts()

    def refresh_fts(self, field: str = "search_text") -> str | None:
        """(Re)build the full-text index used by hybrid search."""
        if self.exists() and self.count() > 0:
            try:
                self.db.open_table(self.table).create_fts_index(
                    field, replace=True, use_tantivy=False
                )
            except Exception:
                return "full-text index refresh failed; vector search is still available"
        return None

    def search_text(self, query: str, k: int = 8, where: str | None = None) -> list[dict]:
        rows, _warning = self.search_text_with_status(query, k=k, where=where)
        return rows

    def search_text_with_status(
        self, query: str, k: int = 8, where: str | None = None
    ) -> tuple[list[dict], str | None]:
        if not self.exists():
            return [], "active LanceDB table is missing"
        try:
            q = self.db.open_table(self.table).search(query, query_type="fts").limit(k)
            if where:
                q = q.where(where)
            return q.to_list(), None
        except Exception as exc:
            return [], f"full-text search unavailable; degraded to vector search ({exc})"

    def add(self, rows: list[dict]) -> None:
        if not rows:
            return
        tbl = self._open()
        names = set(tbl.schema.names)
        tbl.add([{k: v for k, v in row.items() if k in names} for row in rows])

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

    def by_chunk_id(self, chunk_id: str) -> dict | None:
        """Fetch one chunk by its stable chunk id."""
        if not self.exists():
            return None
        safe = chunk_id.replace("'", "''")
        try:
            rows = self.db.open_table(self.table).search().where(
                f"chunk_id = '{safe}'"
            ).limit(1).to_list()
        except Exception:
            return None
        if not rows:
            return None
        row = rows[0]
        row.pop("vector", None)
        row.pop("search_text", None)
        return row

    def symbol_inventory(self, k: int = 5000) -> list[dict]:
        """Return lightweight symbol rows for near-miss suggestions."""
        if not self.exists():
            return []
        try:
            rows = self.db.open_table(self.table).search().where(
                "symbol != ''"
            ).limit(k).to_list()
        except Exception:
            return []
        out = []
        for r in rows:
            out.append(
                {
                    "symbol": r.get("symbol") or "",
                    "symbol_kind": r.get("symbol_kind") or "",
                    "rel_path": r.get("rel_path") or "",
                    "start_line": r.get("start_line"),
                    "end_line": r.get("end_line"),
                    "chunk_role": r.get("chunk_role") or "",
                }
            )
        return out

    def search(self, vector: list[float], k: int = 8, where: str | None = None) -> list[dict]:
        if not self.exists():
            return []
        q = self.db.open_table(self.table).search(vector).limit(k)
        if where:
            q = q.where(where)
        return q.to_list()
