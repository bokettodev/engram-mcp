"""Extension -> language mapping + a lazy, cached tree-sitter parser factory.

Grammars are loaded from the bundled ``tree-sitter-<lang>`` wheels (no runtime
download). Languages outside ``GRAMMAR_LANGS`` are still indexed, via the
chunker's line-window fallback.
"""

from __future__ import annotations

from functools import lru_cache

from tree_sitter import Language, Parser

EXT_TO_LANG: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    # indexed via line-window fallback (no bundled grammar yet):
    ".md": "markdown", ".markdown": "markdown", ".mdx": "markdown",
    ".rst": "rst",
    ".txt": "text",
    ".json": "json", ".jsonc": "json",
    ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".ini": "ini",
    ".html": "html", ".htm": "html", ".css": "css", ".scss": "css", ".less": "css",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell",
    ".sql": "sql",
    ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".rb": "ruby", ".php": "php", ".cs": "csharp", ".swift": "swift",
}

# Languages with a bundled tree-sitter grammar -> AST-aware chunking.
GRAMMAR_LANGS = frozenset(
    {
        "python", "javascript", "typescript", "tsx", "go", "rust",
        "java", "c", "cpp", "ruby", "csharp",
    }
)

# Every language id that can land in the store's `language` column. Used to
# validate search filters (the values are simple identifiers, so a whitelist
# check makes the SQL predicate injection-safe).
VALID_LANGUAGES = frozenset(EXT_TO_LANG.values())


def detect_language(ext: str) -> str | None:
    """Return the language name for a file extension, or None if unknown."""
    return EXT_TO_LANG.get(ext.lower())


def is_valid_language(lang: str) -> bool:
    return lang in VALID_LANGUAGES


@lru_cache(maxsize=None)
def get_parser(language: str) -> Parser | None:
    """Return a cached tree-sitter Parser for a grammar language, else None."""
    if language not in GRAMMAR_LANGS:
        return None
    if language == "python":
        import tree_sitter_python as ts

        lang = ts.language()
    elif language == "javascript":
        import tree_sitter_javascript as ts

        lang = ts.language()
    elif language == "typescript":
        import tree_sitter_typescript as ts

        lang = ts.language_typescript()
    elif language == "tsx":
        import tree_sitter_typescript as ts

        lang = ts.language_tsx()
    elif language == "go":
        import tree_sitter_go as ts

        lang = ts.language()
    elif language == "rust":
        import tree_sitter_rust as ts

        lang = ts.language()
    elif language == "java":
        import tree_sitter_java as ts

        lang = ts.language()
    elif language == "c":
        import tree_sitter_c as ts

        lang = ts.language()
    elif language == "cpp":
        import tree_sitter_cpp as ts

        lang = ts.language()
    elif language == "ruby":
        import tree_sitter_ruby as ts

        lang = ts.language()
    elif language == "csharp":
        import tree_sitter_c_sharp as ts

        lang = ts.language()
    else:  # pragma: no cover - guarded by GRAMMAR_LANGS
        return None
    return Parser(Language(lang))
