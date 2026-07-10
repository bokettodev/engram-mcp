"""Tests for the final-review fixes: atomic/GC, path-escape, stale-generated."""

from __future__ import annotations

import lancedb
import pytest

from engram_mcp import manifest, paths


def _tables(proj) -> set[str]:
    db = lancedb.connect(str(paths.project_dir(proj, create=False) / "lancedb"))
    return set(db.list_tables().tables)


def test_full_rebuild_keeps_previous_table_then_gcs(tmp_path, monkeypatch, provider):
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path / "home"))
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")

    from engram_mcp.pipeline import index_project

    index_project(proj, provider)
    m1 = manifest.load_project(paths.project_dir(proj, create=False))

    index_project(proj, provider, full_rebuild=True)
    m2 = manifest.load_project(paths.project_dir(proj, create=False))
    assert m2.active_table != m1.active_table
    assert m1.active_table in _tables(proj)  # previous kept for in-flight readers

    index_project(proj, provider, full_rebuild=True)
    assert m1.active_table not in _tables(proj)  # GC'd at the next rebuild


def test_reindex_rejects_path_escape(tmp_path, monkeypatch, provider):
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path / "home"))
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    evil = tmp_path / "proj_evil"
    evil.mkdir()
    (evil / "x.py").write_text("secret = 1\n", encoding="utf-8")

    from engram_mcp.pipeline import index_project, reindex_file

    index_project(proj, provider)
    with pytest.raises(ValueError):
        reindex_file(proj, provider, "../proj_evil/x.py")


def test_manifest_missing_schema_version_is_incompatible(tmp_path):
    pdir = tmp_path / "idx"
    pdir.mkdir()
    (pdir / "project.json").write_text(
        '{"project_id":"p","root_path":"T:/repo","active_table":"chunks"}',
        encoding="utf-8",
    )

    loaded = manifest.load_project(pdir)
    assert loaded is not None
    assert loaded.schema_version == 0
    with pytest.raises(Exception) as exc:
        manifest.load_project_strict(pdir)
    assert "schema_version" in str(exc.value)


def test_manifest_previous_schema_version_fails_loud_after_bump(tmp_path):
    """manifest.SCHEMA_VERSION was bumped 3 -> 4 alongside the new
    requested_git_fix_regex identity field; a manifest written by the
    previous (v3) code must not be silently accepted with the field defaulted
    — it must fail loud so the maintainer reindexes.
    """

    pdir = tmp_path / "idx"
    pdir.mkdir()
    import json

    (pdir / "project.json").write_text(
        json.dumps({"project_id": "p", "root_path": "T:/repo", "active_table": "chunks", "schema_version": 3}),
        encoding="utf-8",
    )

    assert manifest.SCHEMA_VERSION == 4
    with pytest.raises(Exception) as exc:
        manifest.load_project_strict(pdir)
    assert "schema_version" in str(exc.value)


def test_legacy_git_history_schema_lacking_identity_is_treated_as_stale(tmp_path, monkeypatch):
    """A history/SZZ sidecar written by the previous gitstore schema (v1, no
    requested-regex identity) must be invalidated (never assumed fresh), not
    silently loaded as if current.
    """

    import json

    from engram_mcp import gitstore, paths

    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path / "home"))
    paths._reset_data_home_for_tests()
    logical_project_id = "legacy-proj"

    assert gitstore.SCHEMA_VERSION == 2

    history_path = gitstore.history_path(logical_project_id, create=True)
    history_path.write_text(
        json.dumps(
            {
                "schema_version": 1,  # previous gitstore schema, predates requested_fix_regex
                "logical_project_id": logical_project_id,
                "status": "ready",
                "fix_regex": r"(?i)\bfix\b",
                # no requested_fix_regex: legacy payload
                "commits": [],
                "fingerprint": {"hash": "legacy-hash"},
            }
        ),
        encoding="utf-8",
    )
    szz_path = gitstore.szz_path(logical_project_id, create=True)
    szz_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "logical_project_id": logical_project_id,
                "status": "ready",
                "fix_regex": r"(?i)\bfix\b",
                "fingerprint": {"hash": "legacy-hash"},
            }
        ),
        encoding="utf-8",
    )

    # The version-gate rejects the legacy payload outright...
    assert gitstore.load_history(logical_project_id) is None
    assert gitstore.load_szz(logical_project_id) is None


def test_file_becoming_generated_is_removed(tmp_path, monkeypatch, provider):
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path / "home"))
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "a.py").write_text("def special_marker_fn():\n    return 1\n", encoding="utf-8")
    (proj / "b.py").write_text("def other_fn():\n    return 2\n", encoding="utf-8")

    from engram_mcp.pipeline import index_project, search_project

    index_project(proj, provider)
    # a.py now looks generated -> incremental must drop its rows, not keep stale.
    (proj / "a.py").write_text("// @generated\ndef special_marker_fn():\n    return 1\n", encoding="utf-8")
    stats = index_project(proj, provider)
    assert stats.deleted >= 1

    hits = search_project(proj, provider, "special marker fn", k=5)
    assert all(h["rel_path"] != "a.py" for h in hits)


def test_find_definition_exact_symbol(tmp_path, monkeypatch, provider):
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path / "home"))
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "cache.py").write_text(
        "class EmbeddingCache:\n    def get(self):\n        return 1\n", encoding="utf-8"
    )
    (proj / "other.py").write_text("def unrelated():\n    return 0\n", encoding="utf-8")

    from engram_mcp.pipeline import find_definition, index_project

    index_project(proj, provider)
    rows = find_definition(proj, "EmbeddingCache")
    assert rows
    assert rows[0]["rel_path"] == "cache.py"
    assert "EmbeddingCache" in (rows[0]["symbol"] or "")
