"""Golden-ish tests for AST chunking + line-window fallback."""

from __future__ import annotations

from engram_mcp import config
from engram_mcp.indexing.chunker import chunk_file


def _symbols(chunks):
    return {c.symbol for c in chunks}


def test_python_function_class_and_module():
    src = (
        "import os\n"
        "\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "class Calc:\n"
        "    def mul(self, a, b):\n"
        "        return a * b\n"
    )
    chunks = chunk_file("m.py", "python", src)
    syms = _symbols(chunks)
    assert "add" in syms
    assert "Calc" in syms
    add = next(c for c in chunks if c.symbol == "add")
    assert (add.start_line, add.end_line) == (3, 4)
    assert add.symbol_kind == "function_definition"
    # the leading `import os` becomes a module chunk
    assert "module" in {c.symbol_kind for c in chunks}


def test_javascript_symbols():
    src = "function foo(x){ return x }\nclass Bar { baz(){ return 1 } }\n"
    chunks = chunk_file("m.js", "javascript", src)
    syms = _symbols(chunks)
    assert "foo" in syms
    assert "Bar" in syms


def test_typescript_interface_and_function():
    src = (
        "interface User { id: number; name: string }\n"
        "function make(): User { return { id: 1, name: 'a' } }\n"
    )
    chunks = chunk_file("m.ts", "typescript", src)
    syms = _symbols(chunks)
    assert "User" in syms
    assert "make" in syms


def test_go_function():
    src = "package main\n\nfunc Add(a int, b int) int {\n\treturn a + b\n}\n"
    chunks = chunk_file("m.go", "go", src)
    assert "Add" in _symbols(chunks)


def test_rust_function():
    src = "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n"
    chunks = chunk_file("m.rs", "rust", src)
    assert "add" in _symbols(chunks)


def test_markdown_single_section_chunk():
    chunks = chunk_file("readme.md", "markdown", "# Title\n\nsome text\n")
    assert len(chunks) == 1
    assert chunks[0].symbol_kind == "section"
    assert chunks[0].symbol == "Title"
    assert chunks[0].start_line == 1


def test_large_function_splits_into_windows():
    body = "\n".join(f"    x{i} = {i}" for i in range(3000))
    src = f"def big():\n{body}\n"
    chunks = chunk_file("m.py", "python", src)
    big_chunks = [c for c in chunks if c.symbol == "big"]
    assert len(big_chunks) >= 2
    # windows stay within the token cap (with a little slack for a long line)
    assert all(c.token_estimate <= 600 for c in big_chunks)


def test_line_ranges_are_one_based_and_contiguous_coverage():
    src = "def f():\n    return 1\n"
    chunks = chunk_file("m.py", "python", src)
    f = next(c for c in chunks if c.symbol == "f")
    assert (f.start_line, f.end_line) == (1, 2)


def test_unparsable_grammar_file_falls_back():
    # Not valid python, but the chunker must still yield something, not crash.
    chunks = chunk_file("broken.py", "python", "def (((( :\n  ???\n")
    assert chunks
    assert all(c.start_line >= 1 for c in chunks)


def test_js_export_function_and_class():
    src = "export function foo(x){ return x }\nexport class Bar { baz(){ return 1 } }\n"
    chunks = chunk_file("m.js", "javascript", src)
    syms = _symbols(chunks)
    assert "foo" in syms
    assert "Bar" in syms


def test_js_const_arrow_is_a_symbol():
    src = "const Widget = (props) => {\n  return props.value\n}\n"
    chunks = chunk_file("m.jsx", "javascript", src)
    assert "Widget" in _symbols(chunks)


def test_ts_export_interface_and_type():
    src = "export interface User { id: number }\nexport type Id = number\n"
    chunks = chunk_file("m.ts", "typescript", src)
    syms = _symbols(chunks)
    assert "User" in syms
    assert "Id" in syms


def test_oversized_class_emits_signature_chunk():
    methods = "\n".join(
        f"    def method_{i}(self, a, b, c):\n"
        f"        return a + b + c + {i}  # padding padding padding padding padding\n"
        for i in range(40)
    )
    src = f"class Big(Base):\n    '''A big class.'''\n{methods}\n"
    chunks = chunk_file("big.py", "python", src)
    # the class header must survive the split
    assert any("class Big(Base):" in c.text for c in chunks)
    # methods become their own nested symbol chunks
    assert any((c.symbol or "").startswith("Big.method_") for c in chunks)


def test_long_single_line_is_hard_split():
    long_line = "x = [" + ",".join(str(i) for i in range(3000)) + "]"
    chunks = chunk_file("data.txt", "text", long_line + "\n")
    budget = 480 * 4
    assert len(chunks) >= 2
    assert all(len(c.text) <= budget for c in chunks)


def test_java_class_symbol():
    src = "public class Calc {\n    public int add(int a, int b) {\n        return a + b;\n    }\n}\n"
    assert "Calc" in _symbols(chunk_file("Calc.java", "java", src))


def test_c_function_chunked():
    chunks = chunk_file("m.c", "c", "int add(int a, int b) {\n    return a + b;\n}\n")
    assert any(c.symbol_kind == "function_definition" for c in chunks)


def test_ruby_method_symbol():
    assert "add" in _symbols(chunk_file("m.rb", "ruby", "def add(a, b)\n  a + b\nend\n"))


def test_csharp_class_symbol():
    src = "public class Calc {\n    public int Add(int a, int b) { return a + b; }\n}\n"
    assert "Calc" in _symbols(chunk_file("Calc.cs", "csharp", src))


def test_oversized_brace_class_keeps_signature():
    # JS/brace style: the body `{` shares the class's first line.
    methods = "\n".join(
        f"  method_{i}(a, b) {{ return a + b + {i}; }} // padding padding padding padding"
        for i in range(60)
    )
    src = f"class Big extends Base {{\n{methods}\n}}\n"
    chunks = chunk_file("big.js", "javascript", src)
    assert any("class Big extends Base" in c.text for c in chunks)
    assert any((c.symbol or "").startswith("Big.method_") for c in chunks)


def test_c_function_name_from_declarator():
    chunks = chunk_file("m.c", "c", "int add_two(int a, int b) {\n    return a + b;\n}\n")
    assert "add_two" in _symbols(chunks)


# --- prose / markdown ---------------------------------------------------------


def test_markdown_sections_split_by_heading():
    src = (
        "# Guide\n\nintro paragraph\n\n"
        "## Install\n\nrun the installer\n\n"
        "## Usage\n\ncall the tool\n"
    )
    chunks = chunk_file("doc.md", "markdown", src)
    syms = _symbols(chunks)
    assert "Guide" in syms
    assert "Guide > Install" in syms
    assert "Guide > Usage" in syms
    assert all(c.symbol_kind == "section" for c in chunks)
    install = next(c for c in chunks if c.symbol == "Guide > Install")
    assert "run the installer" in install.text


def test_markdown_breadcrumb_pops_deeper_levels():
    src = (
        "# A\n\nt\n\n"
        "## B\n\nt\n\n"
        "### C\n\nt\n\n"
        "## D\n\nt\n"
    )
    syms = _symbols(chunk_file("d.md", "markdown", src))
    assert "A > B > C" in syms
    assert "A > D" in syms  # level-2 D pops B and C off the stack


def test_markdown_preamble_before_first_heading():
    src = "lead-in text before any heading\n\n# Heading\n\nbody\n"
    chunks = chunk_file("d.md", "markdown", src)
    pre = next(c for c in chunks if c.symbol is None)
    assert pre.symbol_kind == "prose"
    assert pre.start_line == 1
    assert "lead-in" in pre.text


def test_markdown_ignores_headings_in_fenced_code():
    src = (
        "# Real\n\n"
        "```\n"
        "# not a heading\n"
        "```\n\n"
        "more text\n"
    )
    chunks = chunk_file("d.md", "markdown", src)
    syms = _symbols(chunks)
    assert "Real" in syms
    assert "not a heading" not in syms
    assert len(chunks) == 1  # the fenced `#` did not start a new section


def test_markdown_no_headings_is_prose():
    src = "just a paragraph\n\nand another one\n"
    chunks = chunk_file("d.md", "markdown", src)
    assert all(c.symbol_kind == "prose" for c in chunks)
    assert all(c.symbol is None for c in chunks)


def test_plain_text_packs_by_paragraph():
    chunks = chunk_file("notes.txt", "text", "first para\n\nsecond para\n\nthird para\n")
    # small paragraphs pack into a single chunk on paragraph boundaries
    assert len(chunks) == 1
    assert chunks[0].symbol_kind == "prose"
    assert chunks[0].start_line == 1
    assert "first para" in chunks[0].text and "third para" in chunks[0].text


def test_prose_oversized_section_splits_into_multiple_chunks():
    paras = "\n\n".join(f"Paragraph number {i} " + "word " * 60 for i in range(40))
    src = f"# Big\n\n{paras}\n"
    chunks = chunk_file("big.md", "markdown", src)
    big = [c for c in chunks if (c.symbol or "").startswith("Big")]
    assert len(big) >= 2
    assert all(c.token_estimate <= config.PROSE_CHUNK_MAX_TOKENS + 5 for c in big)


