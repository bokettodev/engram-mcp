from __future__ import annotations

import os
import subprocess
import sys


def test_server_and_pipeline_import_without_torch() -> None:
    env = {**os.environ, "ENGRAM_SKIP_MODEL": "1"}
    code = (
        "import sys; "
        "import engram_mcp.pipeline; "
        "import engram_mcp.server; "
        "assert 'torch' not in sys.modules, sorted(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_grepworker_stays_free_of_torch_lancedb_and_pipeline() -> None:
    """grep_index isolates the caller's regex in a bare subprocess whose
    entry point is engram_mcp.grepworker (see pipeline._grep_rows_with_timeout).
    That module must import nothing beyond the standard library: it must not
    drag torch, lancedb, or the engram_mcp.pipeline/engram_mcp.server module
    graph into what is otherwise a bare, lightweight worker process -- a hung
    or crashed regex must never leave a model/DB handle stranded in a child
    that only exists for timeout isolation.
    """
    code = (
        "import sys; "
        "import engram_mcp.grepworker; "
        "blocked = {'torch', 'lancedb', 'engram_mcp.pipeline', 'engram_mcp.server'}; "
        "leaked = blocked & set(sys.modules); "
        "assert not leaked, sorted(leaked)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_regex_child_stays_free_of_torch_lancedb_pipeline_and_server() -> None:
    code = (
        "import sys; "
        "import engram_mcp._regex_child; "
        "blocked = {'torch', 'lancedb', 'engram_mcp.pipeline', 'engram_mcp.server'}; "
        "leaked = blocked & set(sys.modules); "
        "assert not leaked, sorted(leaked)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
