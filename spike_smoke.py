"""Phase 0 Windows dependency spike — throwaway smoke test.

Proves the native stack works on Windows without a compiler:
  tree-sitter (AST parse) + fastembed (bge-small ONNX) + LanceDB (write/search).
Run: uv run python spike_smoke.py
First run downloads the bge-small ONNX model (network, ~tens of MB).
"""

import os
import tempfile
import time

import lancedb
import tree_sitter_python as tspython
from fastembed import TextEmbedding
from tree_sitter import Language, Parser


def main() -> None:
    # 1. tree-sitter: parse a snippet, find the function symbol.
    parser = Parser(Language(tspython.language()))
    tree = parser.parse(b"def hello(x):\n    return x + 1\n")
    root = tree.root_node
    func = next((n for n in root.children if n.type == "function_definition"), None)
    name = None
    if func is not None:
        ident = next((c for c in func.children if c.type == "identifier"), None)
        if ident is not None:
            name = ident.text.decode()
    print(f"[tree-sitter] root={root.type} function='{name}'")
    assert name == "hello", "tree-sitter did not extract the function name"

    # 2. fastembed: embed 100 code-ish chunks via bge-small.
    texts = [
        f"def func_{i}(a, b):\n    # compute thing {i}\n    return a + b + {i}"
        for i in range(100)
    ]
    t0 = time.time()
    model = TextEmbedding("BAAI/bge-small-en-v1.5")  # first call: load + maybe download
    load_s = time.time() - t0
    t1 = time.time()
    vecs = [v.tolist() for v in model.embed(texts)]  # passage embeddings
    embed_s = time.time() - t1
    dim = len(vecs[0])
    print(
        f"[fastembed] model load {load_s:.1f}s; embedded {len(vecs)} chunks "
        f"dim={dim} in {embed_s:.2f}s ({len(texts)/embed_s:.0f} chunks/s)"
    )
    assert dim == 384, f"unexpected embedding dim {dim}"

    # 3. LanceDB: write rows with vector + metadata.
    rows = [
        {"chunk_id": i, "path": f"f{i}.py", "content": texts[i], "vector": vecs[i]}
        for i in range(len(texts))
    ]
    dbdir = os.path.join(tempfile.gettempdir(), "cidx_spike_db")
    db = lancedb.connect(dbdir)
    if "chunks" in db.table_names():
        db.drop_table("chunks")
    tbl = db.create_table("chunks", data=rows)
    print(f"[lancedb] wrote {tbl.count_rows()} rows -> {dbdir}")

    # 4. vector search with a query embedding.
    qv = list(model.query_embed(["a function that adds two numbers"]))[0].tolist()
    hits = tbl.search(qv).limit(3).to_list()
    print("[search] top-3 for 'a function that adds two numbers':")
    for h in hits:
        print(f"   #{h['chunk_id']:>3}  dist={h['_distance']:.4f}  {h['path']}")
    assert len(hits) == 3

    print("SMOKE OK")


if __name__ == "__main__":
    main()
