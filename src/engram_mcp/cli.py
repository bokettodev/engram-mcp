"""Phase 1 CLI: walk a project, chunk it, and print stats (and sample chunks).

    uv run engram chunk <path> [--show N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from engram_mcp.indexing.chunker import chunk_file
from engram_mcp.indexing.ignore import IgnoreMatcher
from engram_mcp.indexing.walker import walk


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None
    except OSError:
        return None


class _JsonlWriter:
    def __init__(self) -> None:
        self.seq = 0

    def progress(self, event: dict) -> None:
        self.seq += 1
        payload = {
            "event": "progress",
            "version": 1,
            "seq": self.seq,
            "stage": event.get("stage", ""),
            "unit": event.get("unit"),
            "done": event.get("done"),
            "total": event.get("total"),
        }
        for key, value in event.items():
            if key not in payload:
                payload[key] = value
        print(json.dumps(payload), flush=True)

    def result(self, payload: dict) -> None:
        print(json.dumps({"event": "result", "version": 1, **payload}), flush=True)


def _json_result(payload: dict) -> None:
    print(json.dumps({"event": "result", "version": 1, **payload}), flush=True)


def cmd_chunk(args: argparse.Namespace) -> int:
    root = Path(args.path).resolve()
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2

    ignore = IgnoreMatcher(root)
    t0 = time.time()
    files = chunks_total = tokens_total = 0
    files_by_lang: Counter[str] = Counter()
    chunks_by_lang: Counter[str] = Counter()
    sample: list = []

    for rec in walk(root, ignore):
        text = _read_text(rec.abs_path)
        if text is None:
            continue
        files += 1
        key = rec.language or "(other)"
        files_by_lang[key] += 1
        cs = chunk_file(rec.rel_path, rec.language, text)
        chunks_total += len(cs)
        chunks_by_lang[key] += len(cs)
        for c in cs:
            tokens_total += c.token_estimate
        if args.show and len(sample) < args.show:
            sample.extend(cs[: args.show - len(sample)])

    dt = time.time() - t0
    print(f"root:           {root}")
    print(f"files indexed:  {files}")
    print(f"chunks:         {chunks_total}")
    print(f"est. tokens:    {tokens_total:,}")
    print(f"walk+chunk:     {dt:.2f}s")
    print("by language (files / chunks):")
    for lang in sorted(files_by_lang, key=lambda x: -chunks_by_lang[x]):
        print(f"  {lang:<12} {files_by_lang[lang]:>5} / {chunks_by_lang[lang]:>6}")

    if sample:
        print("\n--- sample chunks ---")
        for c in sample:
            sym = c.symbol or "-"
            print(f"\n[{c.rel_path}:{c.start_line}-{c.end_line}] "
                  f"{c.symbol_kind} {sym} (~{c.token_estimate} tok)")
            preview = c.text if len(c.text) <= 300 else c.text[:300] + " …"
            print(preview)
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    as_json = getattr(args, "json", False)
    writer = _JsonlWriter() if as_json else None
    root = Path(args.path).resolve()
    if not root.is_dir():
        if as_json:
            _json_result({
                "ok": False,
                "error": f"not a directory: {root}",
                "code": "E_BAD_REQUEST",
                "hint": None,
            })
        else:
            print(f"error: not a directory: {root}", file=sys.stderr)
        return 2
    # Imported lazily so `chunk` doesn't pay the embedder import cost.
    from engram_mcp import errors
    from engram_mcp.embeddings import factory
    from engram_mcp.pipeline import index_project

    provider = None
    try:
        index_device = factory.resolve_index_device(
            index_device=getattr(args, "index_device", None), gpu=args.gpu, cpu=args.cpu
        )
        if not as_json:
            print(f"loading embedder (index device {index_device}) ...", file=sys.stderr)

        def _progress(event: dict) -> None:
            if as_json and writer is not None:
                writer.progress(event)
                return
            stage = event.get("stage")
            if stage == "waiting_for_gpu":
                print("waiting for GPU index slot ...", file=sys.stderr, flush=True)
            elif stage == "embedding":
                done = event.get("done") or 0
                total = event.get("total")
                if total is None:
                    print(f"\rembedding {done} ...", end="", file=sys.stderr, flush=True)
                else:
                    print(f"\rembedding {done}/{total} ...", end="", file=sys.stderr, flush=True)

        provider = factory.make_index_provider(index_device, progress=_progress)

        stats = index_project(root, provider, full_rebuild=args.rebuild, progress=_progress)
        if as_json:
            # machine-readable result (used when the MCP server runs a GPU index
            # in this short-lived subprocess so its own CUDA context fully exits).
            assert writer is not None
            writer.result({
                "ok": True, "mode": stats.mode, "files": stats.files, "chunks": stats.chunks,
                "embedded_unique": stats.embedded_unique, "reused_unique": stats.reused_unique,
                "added": stats.added, "changed": stats.changed, "deleted": stats.deleted,
                "unchanged": stats.unchanged, "embedder_id": provider.model_id,
                "backend_id": provider.backend_id, "device": provider.device,
                "seconds": stats.seconds,
            })
            return 0
        print("\r" + " " * 40 + "\r", end="", file=sys.stderr)
        print(f"root:            {root}")
        print(f"embedder:        {provider.model_id} (dim {provider.dim})")
        print(f"index backend:   {provider.backend_id}")
        print(f"index device:    {provider.device}")
        print(f"mode:            {stats.mode}")
        if stats.mode == "incremental":
            print(f"changes:         +{stats.added} ~{stats.changed} -{stats.deleted}  (unchanged {stats.unchanged})")
        print(f"files:           {stats.files}")
        print(f"chunks:          {stats.chunks}")
        print(f"embedded (new):  {stats.embedded_unique}")
        print(f"reused (cache):  {stats.reused_unique}")
        print(f"index time:      {stats.seconds:.2f}s ({stats.chunks_per_sec:.0f} chunks/s)")
        return 0
    except errors.EngramError as exc:
        if as_json:
            assert writer is not None
            writer.result({"ok": False, "error": str(exc), "code": exc.code, "hint": exc.hint})
        else:
            print(f"error: {exc}", file=sys.stderr)
            if exc.hint:
                print(f"hint: {exc.hint}", file=sys.stderr)
        return 2
    except Exception as exc:
        if as_json:
            assert writer is not None
            writer.result({
                "ok": False,
                "error": str(exc) or repr(exc),
                "code": "E_MODEL_LOAD_FAILED",
                "hint": None,
            })
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if provider is not None:
            factory.release_index_provider(provider)


def cmd_remove(args: argparse.Namespace) -> int:
    from engram_mcp.pipeline import remove_project

    root = Path(args.path).resolve()
    removed = remove_project(root)
    print(f"{'removed' if removed else 'nothing to remove'}: {root}")
    return 0


def cmd_gc(args: argparse.Namespace) -> int:
    from engram_mcp.inventory import gc_orphans

    out = gc_orphans(prune=args.prune)
    print(json.dumps(out, indent=2))
    return 2 if out.get("errors") else 0


def cmd_find_def(args: argparse.Namespace) -> int:
    from engram_mcp.pipeline import ProjectNotIndexedError, find_definition

    root = Path(args.path).resolve()
    try:
        out = find_definition(root, args.symbol, include_suggestions=True)
    except ProjectNotIndexedError:
        print(f'project not indexed: {root}\nrun: engram index "{root}"', file=sys.stderr)
        return 2
    rows = out["results"]
    if not rows:
        print(f"no definition found for {args.symbol!r}")
        if out["suggestions"]:
            print("suggestions: " + ", ".join(s["symbol"] for s in out["suggestions"][:5]))
        return 0
    for r in rows:
        print(f"\n[{r['rel_path']}:{r['start_line']}-{r['end_line']}] "
              f"{r.get('symbol_kind', '')} {r.get('symbol', '')}")
        content = r.get("content", "")
        print(content if len(content) <= 600 else content[:600] + " …")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from engram_mcp import evaluate
    from engram_mcp.embeddings import factory
    from engram_mcp.pipeline import load_query_index

    root = Path(args.path).resolve()
    cases = evaluate.load_cases(args.evalfile)
    qi = load_query_index(root)
    provider = factory.provider_for_model_id(qi.manifest.embedder_id)
    report = evaluate.run_evaluation(root, provider, cases, k=args.k, mode=args.mode, rerank=args.rerank)
    o = report.overall
    print(f"eval: {o.n} queries  mode={args.mode}  (mean {report.mean_latency_ms:.0f} ms)")
    print(f"  overall        hit@1 {o.hit1:6.1%}  hit@5 {o.hit5:6.1%}  hit@10 {o.hit10:6.1%}  MRR {o.mrr:.3f}")
    print("  by category:")
    for cat in sorted(report.by_category):
        s = report.by_category[cat]
        print(f"    {cat:<14} ({s.n:>2})  hit@1 {s.hit1:6.1%}  hit@5 {s.hit5:6.1%}  MRR {s.mrr:.3f}")
    if args.verbose:
        for r in report.rows:
            tag = "ok  " if r["rank"] else "MISS"
            print(f"    [{tag}] {r['category']:<12} rank={r['rank']}  {r['query']!r} -> {r['top']}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    root = Path(args.path).resolve()
    from engram_mcp.embeddings import factory
    from engram_mcp.pipeline import ProjectNotIndexedError, load_query_index, search_project

    try:
        qi = load_query_index(root)
        provider = factory.provider_for_model_id(qi.manifest.embedder_id)
        hits = search_project(root, provider, args.query, k=args.k, language=args.lang,
                              mode=args.mode, rerank=args.rerank)
    except ProjectNotIndexedError:
        print(f'project not indexed: {root}\nrun: engram index "{root}"', file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not hits:
        print("no results.")
        return 0
    for h in hits:
        sym = h.get("symbol") or "-"
        score = h.get("score", 0.0)
        print(f"\n[{h['rel_path']}:{h['start_line']}-{h['end_line']}] "
              f"{h.get('symbol_kind', '')} {sym}  (score={score:.4f})")
        content = h.get("content", "")
        preview = content if len(content) <= 400 else content[:400] + " …"
        print(preview)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="engram", description="Semantic code index."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("chunk", help="walk a project and chunk it (no embedding)")
    pc.add_argument("path", help="project root directory")
    pc.add_argument("--show", type=int, default=0, metavar="N",
                    help="print the first N sample chunks")
    pc.set_defaults(func=cmd_chunk)

    pi = sub.add_parser("index", help="build/update the semantic index for a project")
    pi.add_argument("path", help="project root directory")
    pi.add_argument("--rebuild", action="store_true",
                    help="force a full rebuild (atomic table swap) instead of incremental")
    pi.add_argument("--gpu", action="store_true",
                    help="force CUDA indexing (needs `uv sync --extra gpu`); errors if no GPU. Default already prefers GPU.")
    pi.add_argument("--cpu", action="store_true",
                    help="force CPU indexing — a slow fallback; only when you can't use a GPU")
    pi.add_argument("--index-device", choices=["auto", "cpu", "cuda"], default=None,
                    help=argparse.SUPPRESS)  # explicit setting; used by the MCP server subprocess
    pi.add_argument("--json", action="store_true", help=argparse.SUPPRESS)  # machine-readable; used by the MCP server's GPU subprocess
    pi.set_defaults(func=cmd_index)

    prm = sub.add_parser("remove", help="delete a project's index from disk")
    prm.add_argument("path", help="project root directory")
    prm.set_defaults(func=cmd_remove)

    pgc = sub.add_parser("gc", help="find or prune index dirs whose project root is missing")
    gc_mode = pgc.add_mutually_exclusive_group()
    gc_mode.add_argument("--dry-run", dest="prune", action="store_false",
                         help="report orphaned index dirs without deleting them (default)")
    gc_mode.add_argument("--prune", action="store_true",
                         help="delete orphaned index dirs")
    pgc.set_defaults(func=cmd_gc, prune=False)

    pf = sub.add_parser("find-def", help="exact symbol definition lookup (no embedding)")
    pf.add_argument("path", help="project root directory")
    pf.add_argument("symbol", help="symbol name (e.g. EmbeddingCache or LanceStore.refresh_fts)")
    pf.set_defaults(func=cmd_find_def)

    pv = sub.add_parser("eval", help="measure retrieval quality on a query set")
    pv.add_argument("path", help="project root directory")
    pv.add_argument("evalfile", help="JSON list of {query, expected_path, expected_symbol?}")
    pv.add_argument("-k", type=int, default=10, help="results considered per query")
    pv.add_argument("--mode", default="auto", choices=["auto", "hybrid", "vector"])
    pv.add_argument("--rerank", action="store_true", help="cross-encoder rerank (needs --extra gpu)")
    pv.add_argument("-v", "--verbose", action="store_true", help="print per-query ranks")
    pv.set_defaults(func=cmd_evaluate)

    ps = sub.add_parser("search", help="semantic search over an indexed project")
    ps.add_argument("path", help="project root directory")
    ps.add_argument("query", help="natural-language query")
    ps.add_argument("-k", type=int, default=8, help="number of results")
    ps.add_argument("--lang", default=None, help="filter by language")
    ps.add_argument("--mode", default="auto", choices=["auto", "hybrid", "vector"],
                    help="retrieval mode (auto routes identifier queries to hybrid, NL to vector)")
    ps.add_argument("--rerank", action="store_true",
                    help="cross-encoder rerank the candidates (needs `uv sync --extra gpu`)")
    ps.set_defaults(func=cmd_search)

    args = p.parse_args(argv)
    # Set up TLS trust (OS store / CA bundle / insecure) before any embedder
    # import triggers a model download.
    from engram_mcp.net import configure_tls

    configure_tls()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
