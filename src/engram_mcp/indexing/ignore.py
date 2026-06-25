"""Ignore matching: built-in excludes + per-directory ``.gitignore`` / ``.ignore``.

Each directory from the project root down to a file may carry its own ignore
file; a path is ignored if any ancestor directory's spec matches it (evaluated
relative to that directory, like git). Negations are honored within a single
ignore file; cross-file negation (a deeper file un-ignoring a shallower match)
is a known limitation. Specs are cached per directory.
"""

from __future__ import annotations

from pathlib import Path

from pathspec import GitIgnoreSpec

from engram_mcp import config


class IgnoreMatcher:
    def __init__(self, root: Path, extra_globs: tuple[str, ...] = ()):
        self.root = Path(root)
        self.exclude_dirs = config.DEFAULT_EXCLUDE_DIRS
        self._base = GitIgnoreSpec.from_lines(
            list(config.DEFAULT_EXCLUDE_GLOBS) + list(extra_globs)
        )
        self._dir_specs: dict[str, GitIgnoreSpec | None] = {}

    def is_excluded_dir(self, name: str) -> bool:
        return name in self.exclude_dirs

    def _spec_for_dir(self, dir_path: Path) -> GitIgnoreSpec | None:
        key = str(dir_path)
        if key not in self._dir_specs:
            patterns: list[str] = []
            for name in (".gitignore", ".ignore"):
                f = dir_path / name
                if f.is_file():
                    patterns += _read_patterns(f)
            self._dir_specs[key] = GitIgnoreSpec.from_lines(patterns) if patterns else None
        return self._dir_specs[key]

    def is_ignored(self, abs_path: Path) -> bool:
        try:
            rel_root = abs_path.relative_to(self.root).as_posix()
        except ValueError:
            return False
        if self._base.match_file(rel_root):
            return True
        d = abs_path.parent
        while True:
            spec = self._spec_for_dir(d)
            if spec is not None and spec.match_file(abs_path.relative_to(d).as_posix()):
                return True
            if d == self.root:
                break
            d = d.parent
        return False


def _read_patterns(path: Path) -> list[str]:
    out: list[str] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out
