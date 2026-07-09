from __future__ import annotations

import json

from engram_mcp import catalog


def _kind(name: str) -> dict:
    return {"kind": name, "source": "test", "reason": "fixture"}


def _symbol(name: str, kind: str) -> dict:
    return {
        "name": name,
        "symbol_kind": kind,
        "chunk_id": f"{name}-chunk",
        "start_line": 1,
        "end_line": 2,
    }


def _file(
    path: str,
    *,
    language: str,
    chunks: int,
    chunk_roles: dict[str, int],
    symbols: list[dict],
    kinds: list[dict] | None = None,
) -> dict:
    return {
        "path": path,
        "dir": catalog._dir_for(path),
        "language": language,
        "chunks": chunks,
        "chunk_roles": chunk_roles,
        "symbols": symbols,
        "chunk_refs": [],
        "kinds": kinds or [],
    }


def _catalog(files: list[dict]) -> dict:
    return {
        "schema_version": catalog.SCHEMA_VERSION,
        "project_id": "test-project",
        "root_path": "T:/repo",
        "generation": 3,
        "active_table": "chunks",
        "indexed_at": 1.0,
        "totals": {
            "files": len(files),
            "chunks": sum(int(file_row.get("chunks", 0) or 0) for file_row in files),
            "symbols": sum(len(file_row.get("symbols") or []) for file_row in files),
        },
        "files": files,
    }


def _sample_catalog() -> dict:
    return _catalog(
        [
            _file(
                "src/app.py",
                language="python",
                chunks=3,
                chunk_roles={"comment": 1, "executable": 2},
                symbols=[
                    _symbol("App", "class_definition"),
                    _symbol("run", "function_definition"),
                ],
            ),
            _file(
                "src/test_app.py",
                language="python",
                chunks=2,
                chunk_roles={"test": 2},
                symbols=[_symbol("test_run", "function_definition")],
                kinds=[_kind("test")],
            ),
            _file(
                "docs/readme.md",
                language="markdown",
                chunks=1,
                chunk_roles={"prose": 1},
                symbols=[_symbol("Intro", "section")],
                kinds=[_kind("doc")],
            ),
            _file(
                "config/settings.yaml",
                language="yaml",
                chunks=1,
                chunk_roles={"config": 1},
                symbols=[],
                kinds=[_kind("config")],
            ),
            _file(
                "src/empty.py",
                language="python",
                chunks=0,
                chunk_roles={},
                symbols=[],
            ),
        ]
    )


def _paths(data: dict, **kwargs) -> tuple[set[str], dict]:
    out = catalog.project_map(data, include_files=True, files_limit=50, **kwargs)
    return {file_row["path"] for file_row in out["files"]}, out


def test_project_map_default_dirs_only_and_compact_files() -> None:
    data = _sample_catalog()

    out = catalog.project_map(data, depth=1)
    assert out["files"] == []
    assert out["files_page"]["included"] is False
    assert out["files_page"]["count"] == 0
    assert out["files_page"]["total"] == 4
    assert out["filtered_totals"] == {"files": 4, "chunks": 7, "symbols": 4}

    compact = catalog.project_map(data, include_files=True, path_prefix="src/app.py")
    row = compact["files"][0]
    assert set(row) == {
        "path",
        "dir",
        "language",
        "chunks",
        "symbols_count",
        "chunk_roles",
        "kinds",
    }
    assert row["symbols_count"] == 2

    with_symbols = catalog.project_map(
        data,
        include_files=True,
        include_symbols=True,
        symbols_limit=1,
        path_prefix="src/app.py",
    )
    row = with_symbols["files"][0]
    assert len(row["symbols"]) == 1
    assert row["symbols_count"] == 2
    assert row["symbols_has_more"] is True


def test_project_map_independent_dir_and_file_pagination() -> None:
    data = _sample_catalog()

    out = catalog.project_map(
        data,
        depth=1,
        dirs_limit=1,
        dirs_offset=1,
        include_files=True,
        files_limit=2,
        files_offset=1,
        non_empty=False,
    )
    assert [row["dir"] for row in out["dirs"]] == ["docs"]
    assert out["dirs_page"] == {
        "offset": 1,
        "limit": 1,
        "count": 1,
        "total": 3,
        "has_more": True,
    }
    assert [row["path"] for row in out["files"]] == ["docs/readme.md", "src/app.py"]
    assert out["files_page"] == {
        "offset": 1,
        "limit": 2,
        "count": 2,
        "total": 5,
        "has_more": True,
        "included": True,
    }
    assert out["has_more"] is True

    legacy = catalog.project_map(data, depth=1, limit=1)
    assert legacy["dirs_page"]["limit"] == 1
    assert legacy["dirs_page"]["has_more"] is True


def test_project_map_filters_apply_to_files_and_dir_aggregates() -> None:
    data = _sample_catalog()

    paths, _out = _paths(data, code_only=True)
    assert paths == {"src/app.py", "src/test_app.py"}

    paths, _out = _paths(data, languages=["markdown"])
    assert paths == {"docs/readme.md"}

    paths, _out = _paths(data, chunk_roles=["config"])
    assert paths == {"config/settings.yaml"}

    paths, _out = _paths(data, kinds=["doc"])
    assert paths == {"docs/readme.md"}

    paths, _out = _paths(data, path_prefix="src\\")
    assert paths == {"src/app.py", "src/test_app.py"}

    paths, _out = _paths(data, path_glob="src/*test*.py")
    assert paths == {"src/test_app.py"}

    paths, _out = _paths(data, symbol_kinds=["section"])
    assert paths == {"docs/readme.md"}

    paths, _out = _paths(data, min_symbols=2)
    assert paths == {"src/app.py"}

    paths, _out = _paths(data, path_prefix="src", non_empty=False)
    assert paths == {"src/app.py", "src/empty.py", "src/test_app.py"}

    paths, out = _paths(
        data,
        code_only=True,
        languages=["python"],
        path_prefix="src",
        min_symbols=2,
    )
    assert paths == {"src/app.py"}
    assert out["filtered_totals"] == {"files": 1, "chunks": 3, "symbols": 2}
    assert out["dirs"] == [
        {
            "dir": "src",
            "files": 1,
            "chunks": 3,
            "symbols": 2,
            "languages": {"python": 1},
            "chunk_roles": {"comment": 1, "executable": 2},
            "kinds": {},
        }
    ]


def test_project_map_huge_symbol_lists_are_clipped() -> None:
    symbols = [_symbol(f"symbol_{idx}", "function_definition") for idx in range(500)]
    data = _catalog(
        [
            _file(
                "huge.py",
                language="python",
                chunks=1,
                chunk_roles={"executable": 1},
                symbols=symbols,
            )
        ]
    )

    compact = catalog.project_map(data, include_files=True)
    assert "symbols" not in compact["files"][0]
    assert "symbol_499" not in json.dumps(compact)

    clipped = catalog.project_map(
        data,
        include_files=True,
        include_symbols=True,
        symbols_limit=20,
    )
    row = clipped["files"][0]
    assert row["symbols_count"] == 500
    assert len(row["symbols"]) == 20
    assert row["symbols_has_more"] is True
