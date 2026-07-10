"""Small regex worker for grep_index timeout isolation.

The parent never passes the row corpus through ``multiprocessing.Process``'s
``args`` -- on the ``spawn`` start method (the only one available on
Windows), those args are serialized and copied into the child as part of
``Process.start()``, which runs synchronously *before* the parent's timeout
window begins. For a large corpus that copy could itself take longer than
the timeout is supposed to allow, defeating the whole point of isolating the
regex in a subprocess. Instead the parent streams the rows to this worker in
small batches over a pipe *after* the child is already running and the
timeout clock has started, so both the transfer and the matching happen
inside the same bounded window, and peak memory is one batch's content
rather than the whole corpus.

This module intentionally imports nothing beyond the standard library: it is
the multiprocessing ``spawn`` entry point, so whatever it imports gets
reloaded from scratch in the child interpreter. It must stay clear of torch,
lancedb, and the ``engram_mcp.pipeline``/``engram_mcp.server`` module graph
so a hung or crashed regex never drags a model or DB handle into a bare
worker process.
"""

from __future__ import annotations

import re


def grep_rows_worker(
    data_conn,
    result_conn,
    pattern: str,
    flags: int,
    include_lines: bool,
    max_matches: int,
) -> None:
    """Read row batches from ``data_conn`` until ``None`` (end of stream).

    ``data_conn`` carries one ``list[dict]`` batch per ``recv()``, terminated
    by a ``None`` sentinel. Once ``max_matches`` is hit, this worker stops
    reading and sends its result right away instead of draining the rest of
    the stream -- an early exit, same as the pre-streaming version. Closing
    ``data_conn`` here (see ``finally``) then makes the parent's still-in-
    flight batch sends fail fast (broken pipe) rather than block forever on a
    reader that is gone, so ``diagnostics.grep_rows_with_timeout``'s feeder
    thread and its own timeout/cleanup sequence are unaffected either way.
    """
    try:
        rx = re.compile(pattern, flags)
        by_path: dict[str, dict] = {}
        total_matches = 0
        stopped = False
        while True:
            try:
                batch = data_conn.recv()
            except EOFError:
                break
            if batch is None:
                break
            for row in batch:
                content = row.get("content") or ""
                rel = row.get("rel_path") or ""
                start = int(row.get("start_line", 1) or 1)
                line_cache = None
                for match in rx.finditer(content):
                    item = by_path.setdefault(
                        rel,
                        {"path": rel, "match_count": 0, "line_numbers": set(), "lines": []},
                    )
                    total_matches += 1
                    item["match_count"] += 1
                    line_no = start + content[: match.start()].count("\n")
                    item["line_numbers"].add(line_no)
                    if include_lines:
                        if line_cache is None:
                            line_cache = content.splitlines()
                        idx = max(0, min(line_no - start, len(line_cache) - 1))
                        line_text = line_cache[idx] if line_cache else ""
                        item["lines"].append({"line": line_no, "text": line_text[:300]})
                    if total_matches >= max_matches:
                        stopped = True
                        break
                if stopped:
                    break
            if stopped:
                break
        result_conn.send(("ok", {"by_path": by_path, "total_matches": total_matches, "stopped": stopped}))
    except Exception as exc:
        try:
            result_conn.send(("error", str(exc) or repr(exc)))
        except Exception:
            pass
    finally:
        try:
            data_conn.close()
        except Exception:
            pass
        try:
            result_conn.close()
        except Exception:
            pass
