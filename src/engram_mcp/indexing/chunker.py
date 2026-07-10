"""Chunking: tree-sitter AST-aware for grammar languages, line-window fallback.

Strategy (AST path):
  * Walk top-level nodes. Consecutive non-definition nodes (imports, ...) are
    clustered into a ``module`` chunk.
  * Each definition (function/class/method/type/...) becomes a symbol chunk.
    JS/TS ``export`` wrappers and ``const x = () => ...`` are unwrapped so they
    are recognized as definitions, not anonymous module text.
  * At true module scope, a constant/variable definition (Python `NAME =
    value`, JS/TS top-level `const`, Go `const`/`var`, Rust `const`/`static`)
    also becomes its own symbol chunk instead of being folded into the
    ``module`` bundle -- see `const_defs` in `_ast_chunks` for exactly which
    languages/node shapes are covered. This is what makes `find_definition`
    and the search symbol-boost (`retrieval.hybrid_search`) work for a bare
    constant name like `MAX_FILE_BYTES`, since both key off a chunk's
    `symbol`/`symbol_kind` regardless of what kind of definition it is.
  * A definition over the token cap is split: a header chunk (signature +
    decorators) is emitted, then we recurse into the body's nested definitions;
    if there are none, the whole span is line-window split.

Line numbers are 1-based inclusive and always refer to original file lines.
Token counts are rough estimates (chars/4) used only for sizing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from engram_mcp import config
from engram_mcp.indexing.languages import GRAMMAR_LANGS, get_parser

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Chunk:
    rel_path: str
    language: str | None
    symbol: str | None
    symbol_kind: str | None
    start_line: int  # 1-based inclusive
    end_line: int  # 1-based inclusive
    text: str
    token_estimate: int


def _est_tokens(text: str) -> int:
    return max(1, len(text) // config.CHARS_PER_TOKEN)


# Node types emitted as their own symbol chunk, per grammar language.
DEFINITION_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "javascript": {
        "function_declaration", "generator_function_declaration",
        "class_declaration", "method_definition",
    },
    "typescript": {
        "function_declaration", "generator_function_declaration",
        "class_declaration", "method_definition",
        "interface_declaration", "type_alias_declaration",
        "enum_declaration", "abstract_class_declaration",
    },
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {
        "function_item", "struct_item", "enum_item", "trait_item",
        "impl_item", "mod_item", "macro_definition", "union_item",
    },
    "java": {
        "class_declaration", "interface_declaration", "enum_declaration",
        "record_declaration", "method_declaration", "constructor_declaration",
    },
    "c": {"function_definition", "struct_specifier", "enum_specifier", "union_specifier"},
    "cpp": {
        "function_definition", "class_specifier", "struct_specifier",
        "enum_specifier", "union_specifier", "namespace_definition", "template_declaration",
    },
    "ruby": {"method", "singleton_method", "class", "module"},
    "csharp": {
        "class_declaration", "interface_declaration", "struct_declaration",
        "enum_declaration", "record_declaration", "method_declaration",
        "constructor_declaration", "namespace_declaration",
    },
}
DEFINITION_TYPES["tsx"] = DEFINITION_TYPES["typescript"]

# JS/TS values that make a `const x = <value>` declaration a definition.
_FN_VALUE_TYPES = {
    "arrow_function", "function", "function_expression",
    "generator_function", "class", "class_expression",
}

_BODY_TYPES = {
    "block", "class_body", "declaration_list", "field_declaration_list",
    "statement_block", "enum_body", "interface_body",
    "compound_statement", "body_statement", "namespace_body",
}


def chunk_file(rel_path: str, language: str | None, text: str) -> list[Chunk]:
    """Chunk one file's text into a list of :class:`Chunk`."""
    if language == "markdown":
        try:
            chunks = _markdown_chunks(rel_path, language, text)
            if chunks:
                return chunks
        except Exception as exc:
            logger.debug("markdown chunking failed for %s: %r", rel_path, exc)
    elif language in GRAMMAR_LANGS:
        parser = get_parser(language)
        if parser is not None:
            try:
                chunks = _ast_chunks(rel_path, language, text, parser)
                if chunks:
                    return chunks
            except Exception as exc:
                logger.debug("AST chunking failed for %s (%s): %r", rel_path, language, exc)
    elif language == "text":
        # Plain text / reStructuredText: pack by paragraph, not raw lines.
        chunks = _prose_chunks(rel_path, language, text.splitlines(), 1, None, "prose")
        if chunks:
            return chunks
    lines = text.splitlines()
    return _line_window_chunks(rel_path, language, lines, 1, None, "file")


# --- Prose / Markdown chunking ------------------------------------------------
# ATX heading (`# ...` to `###### ...`), up to 3 leading spaces per CommonMark,
# with an optional trailing `#` run. Setext (underline) headings are not handled.
_ATX_HEADING = re.compile(r"^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
# Fenced code block open/close marker (``` or ~~~), possibly indented.
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


def _paragraphs(lines: list[str], base_line: int) -> list[tuple[int, int, str]]:
    """Split lines into blank-line-separated paragraphs.

    Returns ``(start_line, end_line, text)`` (1-based inclusive) for each block
    of consecutive non-blank lines; runs of blank lines are dropped.
    """
    paras: list[tuple[int, int, str]] = []
    n = len(lines)
    i = 0
    while i < n:
        while i < n and not lines[i].strip():
            i += 1
        if i >= n:
            break
        start = i
        while i < n and lines[i].strip():
            i += 1
        paras.append((base_line + start, base_line + i - 1, "\n".join(lines[start:i])))
    return paras


def _prose_chunks(
    rel_path: str,
    language: str | None,
    lines: list[str],
    base_line: int,
    symbol: str | None,
    symbol_kind: str | None,
) -> list[Chunk]:
    """Pack paragraphs into token-budgeted chunks without splitting a paragraph.

    Greedily fills each chunk up to the prose cap on paragraph boundaries, with
    a trailing-paragraph overlap carried into the next chunk. A single paragraph
    over the budget is line-window split (its own line range), so an unwrapped
    blob still chunks cleanly.
    """
    budget = config.PROSE_CHUNK_MAX_TOKENS * config.CHARS_PER_TOKEN
    overlap_budget = config.CHUNK_OVERLAP_TOKENS * config.CHARS_PER_TOKEN
    paras = _paragraphs(lines, base_line)
    if not paras:
        return []
    chunks: list[Chunk] = []
    cur: list[tuple[int, int, str]] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur, cur_len
        if not cur:
            return
        txt = "\n\n".join(p[2] for p in cur)
        if txt.strip():
            chunks.append(
                Chunk(rel_path, language, symbol, symbol_kind,
                      cur[0][0], cur[-1][1], txt, _est_tokens(txt))
            )
        cur = []
        cur_len = 0

    for para in paras:
        ps, _pe, ptext = para
        plen = len(ptext)
        if plen > budget:
            flush()
            chunks.extend(
                _line_window_chunks(rel_path, language, ptext.split("\n"), ps, symbol, symbol_kind)
            )
            continue
        if cur_len and cur_len + plen + 2 > budget:
            # Compute the trailing-paragraph overlap before flushing clears `cur`.
            tail: list[tuple[int, int, str]] = []
            tail_len = 0
            for p in reversed(cur):
                if tail and tail_len + len(p[2]) > overlap_budget:
                    break
                tail.insert(0, p)
                tail_len += len(p[2]) + 2
            flush()
            cur = list(tail)
            cur_len = tail_len
        cur.append(para)
        cur_len += plen + 2
    flush()
    return chunks


def _markdown_chunks(rel_path: str, language: str, text: str) -> list[Chunk]:
    """Split markdown into one chunk per heading section.

    The heading breadcrumb (e.g. ``Install > Requirements``) becomes the chunk
    ``symbol`` so it is embedded in the contextual header and surfaces in search
    — a cheap, static cousin of contextual retrieval. Over-cap sections are
    paragraph-packed. Headings inside fenced code blocks are ignored. Files with
    no headings fall through to plain prose packing.
    """
    lines = text.splitlines()
    headings: list[tuple[int, int, str]] = []  # (line_idx, level, title)
    in_fence = False
    fence_char = ""
    for i, line in enumerate(lines):
        fm = _FENCE.match(line)
        if fm:
            marker = fm.group(1)[0]
            if not in_fence:
                in_fence, fence_char = True, marker
            elif marker == fence_char:
                in_fence, fence_char = False, ""
            continue
        if in_fence:
            continue
        hm = _ATX_HEADING.match(line)
        if hm:
            headings.append((i, len(hm.group(1)), hm.group(2).strip()))

    if not headings:
        return _prose_chunks(rel_path, language, lines, 1, None, "prose")

    chunks: list[Chunk] = []
    first = headings[0][0]
    if first > 0 and "\n".join(lines[:first]).strip():
        chunks.extend(_prose_chunks(rel_path, language, lines[:first], 1, None, "prose"))

    stack: list[tuple[int, str]] = []  # (level, title)
    for hi, (idx, level, title) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        breadcrumb = " > ".join(t for _, t in stack)
        end = headings[hi + 1][0] if hi + 1 < len(headings) else len(lines)
        seg = lines[idx:end]
        seg_text = "\n".join(seg)
        if not seg_text.strip():
            continue
        if _est_tokens(seg_text) <= config.PROSE_CHUNK_MAX_TOKENS:
            chunks.append(
                Chunk(rel_path, language, breadcrumb, "section",
                      idx + 1, idx + len(seg), seg_text, _est_tokens(seg_text))
            )
        else:
            chunks.extend(_prose_chunks(rel_path, language, seg, idx + 1, breadcrumb, "section"))

    chunks.sort(key=lambda c: (c.start_line, c.end_line))
    return chunks


def _line_window_chunks(
    rel_path: str,
    language: str | None,
    lines: list[str],
    base_line: int,
    symbol: str | None,
    symbol_kind: str | None,
) -> list[Chunk]:
    """Split ``lines`` into overlapping char-budgeted windows.

    A single line longer than the budget (minified blob) is hard char-split so
    we never hand the embedder an oversized string that relies on truncation.
    """
    budget = config.CHUNK_MAX_TOKENS * config.CHARS_PER_TOKEN
    overlap = config.CHUNK_OVERLAP_TOKENS * config.CHARS_PER_TOKEN
    chunks: list[Chunk] = []
    n = len(lines)
    i = 0
    while i < n:
        if len(lines[i]) > budget:
            line = lines[i]
            step = max(1, budget - overlap)
            pos = 0
            while pos < len(line):
                seg = line[pos : pos + budget]
                if seg.strip():
                    chunks.append(
                        Chunk(rel_path, language, symbol, symbol_kind,
                              base_line + i, base_line + i, seg, _est_tokens(seg))
                    )
                pos += step
            i += 1
            continue
        cur = 0
        j = i
        while j < n and len(lines[j]) <= budget and (cur == 0 or cur + len(lines[j]) + 1 <= budget):
            cur += len(lines[j]) + 1
            j += 1
        text = "\n".join(lines[i:j])
        if text.strip():
            chunks.append(
                Chunk(rel_path, language, symbol, symbol_kind,
                      base_line + i, base_line + j - 1, text, _est_tokens(text))
            )
        if j >= n:
            break
        back = 0
        k = j
        while k > i + 1 and back < overlap:
            k -= 1
            back += len(lines[k]) + 1
        i = max(k, i + 1)
    return chunks


def _ast_chunks(rel_path: str, language: str, text: str, parser) -> list[Chunk]:
    data = text.encode("utf-8")
    tree = parser.parse(data)
    lines = text.splitlines()
    defs = DEFINITION_TYPES.get(language, set())
    chunks: list[Chunk] = []

    def text_of(node) -> str:
        return data[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")

    def span(node) -> tuple[int, int]:
        return node.start_point[0] + 1, node.end_point[0] + 1  # 1-based inclusive

    def name_of(node) -> str | None:
        if node is None:
            return None
        nm = node.child_by_field_name("name")
        if nm is not None:
            return text_of(nm)
        # e.g. python decorated_definition wraps the real def
        for c in node.named_children:
            if c.type in defs:
                inner = c.child_by_field_name("name")
                if inner is not None:
                    return text_of(inner)
        # C/C++ store the identifier under a (possibly nested) declarator field.
        decl = node.child_by_field_name("declarator")
        depth = 0
        while decl is not None and depth < 12:
            if decl.type in ("identifier", "field_identifier", "type_identifier", "scoped_identifier"):
                return text_of(decl)
            decl = decl.child_by_field_name("declarator")
            depth += 1
        return None

    def def_info(node):
        """Return (symbol, kind, body_node) if node is a definition, else None.

        Unwraps JS/TS ``export`` statements and ``const x = fn/arrow/class``.
        """
        t = node.type
        if t == "export_statement":
            for c in node.named_children:
                info = def_info(c)
                if info is not None:
                    nm, kind, body = info
                    return (nm, f"export_{kind}", body)
            return None
        if t in ("lexical_declaration", "variable_declaration"):
            for d in node.named_children:
                if d.type == "variable_declarator":
                    val = d.child_by_field_name("value")
                    if val is not None and val.type in _FN_VALUE_TYPES:
                        kind = "arrow_function" if val.type == "arrow_function" else val.type
                        return (name_of(d), kind, val.child_by_field_name("body"))
            return None
        if t in defs:
            return (name_of(node), t, node.child_by_field_name("body"))
        return None

    def const_defs(node) -> list[tuple[str, str, object]]:
        """Module-level constant/variable definitions understood for `language`.

        Returns ``[(symbol, kind, span_node), ...]`` -- a statement can name
        more than one constant (JS/TS ``const A = 1, B = 2``; Go's grouped
        ``const ( ... )`` block), so each gets its own (symbol, kind) using
        its own sub-node's span, not the whole statement's. Only consulted at
        true module scope (see `top_level` in `walk_children`) so a local
        `x = 1` inside a function/method body is never mistaken for a
        top-level definition.

        Covered: Python (any `NAME = value` / `NAME: T = value` statement),
        JavaScript/TypeScript/TSX (top-level `const` only -- `let`/`var` are
        ordinary mutable bindings, not "constants"), Go (`const`/`var`
        declarations, including the grouped `const ( ... )` form), Rust
        (`const`/`static` items). Skipped: Java/C#/Ruby, which have no true
        module scope -- every binding lives inside a class/module construct,
        so "module-level" doesn't apply the same way and folding class-level
        fields in here would blur this feature with class-member indexing,
        a separate concern. C/C++ top-level `const` declarations use the
        same value-bearing `declaration` node as ordinary (non-const)
        declarations, and `#define` is a distinct preprocessor node
        (`preproc_def`) needing its own name/value extraction -- both
        skipped for now as a follow-up rather than folded into this pass.
        """
        t = node.type
        if language == "python":
            if t != "expression_statement" or len(node.named_children) != 1:
                return []
            inner = node.named_children[0]
            if inner.type != "assignment":
                return []
            left = inner.child_by_field_name("left")
            if left is None or left.type != "identifier":
                return []  # skip tuple/attribute/subscript targets
            return [(text_of(left), "assignment", node)]
        if language in ("javascript", "typescript", "tsx"):
            if t == "export_statement":
                # `export const X = 1` -- unwrap like def_info() does, so an
                # exported module-level constant isn't missed just because
                # it's wrapped.
                inner = next(
                    (c for c in node.named_children if c.type == "lexical_declaration"), None
                )
                if inner is None:
                    return []
                node, t = inner, inner.type
            if t != "lexical_declaration" or not node.children or node.children[0].type != "const":
                return []
            out = []
            for decl in node.named_children:
                if decl.type != "variable_declarator":
                    continue
                nm = decl.child_by_field_name("name")
                if nm is None or nm.type != "identifier":
                    continue
                out.append((text_of(nm), "lexical_declaration", decl))
            return out
        if language == "go":
            if t not in ("const_declaration", "var_declaration"):
                return []
            spec_type = "const_spec" if t == "const_declaration" else "var_spec"
            out = []
            for spec in node.named_children:
                if spec.type != spec_type:
                    continue
                nm = spec.child_by_field_name("name")
                if nm is None:
                    continue
                out.append((text_of(nm), spec_type, spec))
            return out
        if language == "rust":
            if t not in ("const_item", "static_item"):
                return []
            nm = node.child_by_field_name("name")
            if nm is None:
                return []
            return [(text_of(nm), t, node)]
        return []

    def emit_const(name: str, kind: str, node) -> None:
        nt = text_of(node)
        s, e = span(node)
        if _est_tokens(nt) <= config.CHUNK_MAX_TOKENS:
            chunks.append(Chunk(rel_path, language, name, kind, s, e, nt, _est_tokens(nt)))
            return
        chunks.extend(_line_window_chunks(rel_path, language, lines[s - 1 : e], s, name, kind))

    def emit_def(node, parent: str | None, info) -> None:
        nm, kind, body = info
        full = f"{parent}.{nm}" if parent and nm else (nm or parent)
        nt = text_of(node)
        s, e = span(node)
        if _est_tokens(nt) <= config.CHUNK_MAX_TOKENS:
            chunks.append(Chunk(rel_path, language, full, kind, s, e, nt, _est_tokens(nt)))
            return
        if body is None or body.type not in _BODY_TYPES:
            body = next((c for c in node.named_children if c.type in _BODY_TYPES), None)
        if body is not None and any(def_info(c) is not None for c in body.named_children):
            bstart = body.start_point[0] + 1
            # Header = signature + decorators, by BYTES so it survives brace
            # styles where the body's `{` shares the def's first line (JS/Java/
            # C#/C++), not just Python-style bodies on the next line.
            htext = data[node.start_byte : body.start_byte].decode("utf-8", errors="ignore")
            if htext.strip():
                chunks.append(
                    Chunk(rel_path, language, full, kind, s, bstart, htext, _est_tokens(htext))
                )
            walk_children(body.named_children, full)
        else:
            chunks.extend(_line_window_chunks(rel_path, language, lines[s - 1 : e], s, full, kind))

    def walk_children(children, parent: str | None, top_level: bool = False) -> None:
        buf_start: int | None = None
        buf_end: int | None = None

        def flush() -> None:
            nonlocal buf_start, buf_end
            if buf_start is None:
                return
            seg = lines[buf_start - 1 : buf_end]
            txt = "\n".join(seg)
            if txt.strip():
                if _est_tokens(txt) <= config.CHUNK_MAX_TOKENS:
                    chunks.append(
                        Chunk(rel_path, language, parent, "module",
                              buf_start, buf_end, txt, _est_tokens(txt))
                    )
                else:
                    chunks.extend(
                        _line_window_chunks(rel_path, language, seg, buf_start, parent, "module")
                    )
            buf_start = buf_end = None

        for child in children:
            info = def_info(child)
            if info is not None:
                flush()
                emit_def(child, parent, info)
                continue
            if top_level:
                consts = const_defs(child)
                if consts:
                    flush()
                    for name, kind, cnode in consts:
                        emit_const(name, kind, cnode)
                    continue
            s, e = span(child)
            if buf_start is None:
                buf_start = s
            buf_end = e
        flush()

    walk_children(tree.root_node.named_children, None, top_level=True)
    chunks.sort(key=lambda c: (c.start_line, c.end_line))
    return chunks
