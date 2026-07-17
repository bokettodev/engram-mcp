"""Command-line interface for chunking, indexing, search, eval, and index administration."""

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


def _configure_standard_streams() -> None:
    """Keep Unicode documentation output reliable on Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, OSError, ValueError):
            # Test captures and embedded hosts may expose immutable streams.
            pass


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
    from engram_mcp import errors, gitmeta
    from engram_mcp.embeddings import factory
    from engram_mcp.pipeline import index_project

    provider = None
    try:
        index_device = factory.resolve_index_device(
            index_device=getattr(args, "index_device", None), gpu=args.gpu, cpu=args.cpu
        )
        # Precedence: explicit --git-analytics/--no-git-analytics > ENGRAM_GIT_ANALYTICS > enabled.
        requested_git_analytics = getattr(args, "git_analytics", None)
        git_analytics = (
            gitmeta.git_analytics_default() if requested_git_analytics is None else bool(requested_git_analytics)
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

        stats = index_project(
            root,
            provider,
            full_rebuild=args.rebuild,
            progress=_progress,
            git_analytics=git_analytics,
            git_max_commits=getattr(args, "git_max_commits", None),
            git_fix_regex=getattr(args, "git_fix_regex", None),
        )
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
    from engram_mcp.index_repository import remove_project

    root = Path(args.path).resolve()
    removed = remove_project(root)
    print(f"{'removed' if removed else 'nothing to remove'}: {root}")
    return 0


def cmd_gc(args: argparse.Namespace) -> int:
    from engram_mcp.embeddings.cache import global_cache_report
    from engram_mcp.gcreclaim import reclaim_all
    from engram_mcp.inventory import gc_orphans

    prune = args.prune
    out = {
        "dry_run": not prune,
        "orphans": gc_orphans(prune=prune),
        "stale_generations": reclaim_all(dry_run=not prune),
        "embedding_cache": global_cache_report(dry_run=not prune),
    }
    print(json.dumps(out, indent=2))
    has_errors = bool(out["orphans"].get("errors") or out["stale_generations"].get("errors"))
    return 2 if has_errors else 0


def cmd_find_def(args: argparse.Namespace) -> int:
    from engram_mcp.index_repository import ProjectNotIndexedError
    from engram_mcp.query_service import find_definition

    root = Path(args.path).resolve()
    ref = getattr(args, "ref", None)
    try:
        if ref is None:
            out = find_definition(root, args.symbol, include_suggestions=True)
        else:
            out = find_definition(root, args.symbol, include_suggestions=True, ref=ref)
    except ProjectNotIndexedError:
        print(f'project not indexed: {root}\nrun: engram index "{root}"', file=sys.stderr)
        return 2
    for warning in out.get("warnings") or []:
        print(f"warning: {warning}", file=sys.stderr)
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
    from engram_mcp.index_repository import load_query_index
    from engram_mcp.query_service import rerank_enabled

    root = Path(args.path).resolve()
    cases = evaluate.load_cases(args.evalfile)
    qi = load_query_index(root)
    provider = factory.provider_for_model_id(qi.manifest.embedder_id)
    if args.rerank and not rerank_enabled():
        print(
            "warning: --rerank requested but ENGRAM_RERANK_ENABLED is off; "
            "measuring baseline (rerank_applied=false)",
            file=sys.stderr,
        )
    report = evaluate.run_evaluation(root, provider, cases, k=args.k, mode=args.mode, rerank=args.rerank)
    o = report.overall
    print(f"eval: {o.n} queries  mode={args.mode}  (mean {report.mean_latency_ms:.0f} ms)")
    if args.rerank:
        print(f"  rerank_applied {report.rerank_applied_count}/{o.n}")
        for reason, count in sorted(report.rerank_skipped_reasons.items()):
            print(f"  rerank_skipped_reason ({count}): {reason}")
    delta = "n/a" if o.delta_rank is None else f"{o.delta_rank:.2f}"
    print(
        f"  overall        hit@1 {o.hit1:6.1%}  hit@5 {o.hit5:6.1%}  "
        f"hit@10 {o.hit10:6.1%}  MRR {o.mrr:.3f}  "
        f"HNSR@5 {o.hnsr5:6.1%}  HNSR@10 {o.hnsr10:6.1%}  delta {delta}"
    )
    print("  by category:")
    for cat in sorted(report.by_category):
        s = report.by_category[cat]
        delta = "n/a" if s.delta_rank is None else f"{s.delta_rank:.2f}"
        print(
            f"    {cat:<14} ({s.n:>2})  hit@1 {s.hit1:6.1%}  "
            f"hit@5 {s.hit5:6.1%}  MRR {s.mrr:.3f}  HNSR@10 {s.hnsr10:6.1%}  delta {delta}"
        )
    print("  by lexical-overlap bucket:")
    for bucket in sorted(report.by_overlap_bucket):
        s = report.by_overlap_bucket[bucket]
        print(f"    {bucket:<8} ({s.n:>2})  hit@5 {s.hit5:6.1%}  MRR {s.mrr:.3f}  HNSR@10 {s.hnsr10:6.1%}")
    if args.verbose:
        for r in report.rows:
            tag = "ok  " if r["rank"] else "MISS"
            print(f"    [{tag}] {r['category']:<12} rank={r['rank']}  {r['query']!r} -> {r['top']}")

    if args.save_baseline:
        evaluate.save_baseline(
            args.save_baseline, report, evalfile=args.evalfile, mode=args.mode, rerank=args.rerank
        )
        print(f"\nbaseline saved: {args.save_baseline}")

    if args.baseline:
        margin = args.margin if args.margin is not None else evaluate.DEFAULT_NONINFERIORITY_MARGIN
        baseline = evaluate.load_baseline(args.baseline)
        cmp = evaluate.compare_to_baseline(report, baseline, margin=margin)
        print(f"\nbaseline check ({args.baseline}, margin={margin}):")
        for c in cmp["checks"]:
            status = "OK  " if c["ok"] else "FAIL"
            reason = f"  ({c['reason']})" if c.get("reason") else ""
            print(
                f"  [{status}] {c['metric']:<24} current={c['current']}  "
                f"baseline={c['baseline']}  delta={c['delta']}{reason}"
            )
        if not cmp["ok"]:
            print(
                f"\nNON-INFERIORITY CHECK FAILED: {len(cmp['failures'])} metric(s) dropped "
                f"more than {margin} below baseline {args.baseline}",
                file=sys.stderr,
            )
            return 1
        print("\nnon-inferiority check passed.")
    return 0


def cmd_grep(args: argparse.Namespace) -> int:
    """Bounded regex probe over indexed chunk text (CLI-only diagnostic).

    Not exposed as an MCP tool: it approximates grep over overlapping chunks
    (can double-count or miss cross-chunk matches), and agents already have
    better exact-search tools. The underlying implementation stays available
    here and via `engram_mcp.diagnostics.grep_index` for operators.
    """
    from engram_mcp import diagnostics
    from engram_mcp.index_repository import ProjectNotIndexedError

    root = Path(args.path).resolve()
    try:
        out = diagnostics.grep_index(
            root,
            args.pattern,
            ignore_case=args.ignore_case,
            limit=args.limit,
            offset=args.offset,
            max_matches=args.max_matches,
            max_scan_chunks=args.max_scan_chunks,
            include_lines=args.include_lines,
        )
    except ProjectNotIndexedError:
        print(f'project not indexed: {root}\nrun: engram index "{root}"', file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        _json_result(out)
        return 0

    for warning in out.get("warnings") or []:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"pattern:        {out['pattern']!r}  (ignore_case={out['ignore_case']})")
    print(f"status:         {out['status']}")
    print(
        f"matches:        {out['total_matches']} across {out['total_paths']} path(s) "
        f"(scanned {out['scanned_chunks']} chunks)"
    )
    results = out["results"]
    if not results:
        print("no matches.")
        return 0
    for item in results:
        print(f"\n[{item['path']}] {item['match_count']} match(es)  lines={item['line_numbers']}")
        if args.include_lines:
            for line in item.get("lines", []):
                print(f"  {line['line']}: {line['text']}")
    if out.get("has_more"):
        print(f"\n... more results available (offset={out['offset']}, limit={out['limit']})")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    root = Path(args.path).resolve()
    from engram_mcp.embeddings import factory
    from engram_mcp.pipeline import ProjectNotIndexedError, load_query_index, rerank_enabled, search_project

    ref = getattr(args, "ref", None)
    try:
        qi = load_query_index(root) if ref is None else load_query_index(root, ref=ref)
        provider = factory.provider_for_model_id(qi.manifest.embedder_id)
        if args.rerank and not rerank_enabled():
            print(
                "warning: --rerank requested but ENGRAM_RERANK_ENABLED is off; "
                "returning baseline ranking (rerank_applied=false)",
                file=sys.stderr,
            )
        outcome = search_project(
            root,
            provider,
            args.query,
            k=args.k,
            language=args.lang,
            mode=args.mode,
            rerank=args.rerank,
            return_meta=True,
            ref=ref,
            _query_index=qi,
        )
        hits = outcome["hits"]
        from engram_mcp import gitmeta

        source_revision = outcome.get("source_revision")
        revision_warning = gitmeta.source_revision_warning(
            source_revision,
            include_commit_mismatch=True,
        )
        if revision_warning:
            print(f"warning: {revision_warning}", file=sys.stderr)
    except ProjectNotIndexedError:
        print(f'project not indexed: {root}\nrun: engram index "{root}"', file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.rerank:
        if outcome.get("rerank_applied"):
            print(f"rerank_applied=true model={outcome.get('rerank_model') or 'unknown'}", file=sys.stderr)
        else:
            reason = outcome.get("rerank_skipped_reason") or "rerank unavailable"
            print(f"rerank_applied=false reason={reason}", file=sys.stderr)
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
    _configure_standard_streams()
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
    pi.add_argument("--json", action="store_true", help=argparse.SUPPRESS)  # machine-readable; used by the MCP server's index subprocess
    pi.add_argument(
        "--git-analytics",
        dest="git_analytics",
        action="store_true",
        default=None,
        help=(
            "capture git history/SZZ analytics in the catalog sidecar "
            "(default: ENGRAM_GIT_ANALYTICS env var, or on if unset); "
            "overrides ENGRAM_GIT_ANALYTICS for this run"
        ),
    )
    pi.add_argument(
        "--no-git-analytics",
        dest="git_analytics",
        action="store_false",
        help="do not capture git history/SZZ analytics in the catalog sidecar; overrides ENGRAM_GIT_ANALYTICS for this run",
    )
    pi.add_argument(
        "--git-max-commits",
        type=int,
        default=None,
        metavar="N",
        help="limit shared git-history analytics to the newest N commits (default: full history)",
    )
    pi.add_argument(
        "--git-fix-regex",
        default=None,
        metavar="REGEX",
        help=argparse.SUPPRESS,
    )
    pi.set_defaults(func=cmd_index)

    prm = sub.add_parser("remove", help="delete a project's index from disk")
    prm.add_argument("path", help="project root directory")
    prm.set_defaults(func=cmd_remove)

    pgc = sub.add_parser(
        "gc",
        help=(
            "report or reclaim disk usage: orphaned index dirs, superseded LanceDB "
            "generations, and the global embedding cache"
        ),
    )
    gc_mode = pgc.add_mutually_exclusive_group()
    gc_mode.add_argument(
        "--dry-run", dest="prune", action="store_false",
        help=(
            "report orphaned index dirs, stale generations, and embedding-cache size "
            "without deleting anything (default)"
        ),
    )
    gc_mode.add_argument(
        "--prune", action="store_true",
        help=(
            "delete orphaned index dirs, drop superseded (non-active) LanceDB "
            "generation tables + their catalog sidecars for every indexed project, "
            "and prune the global embedding cache to ENGRAM_CACHE_MAX_MB (a no-op "
            "on the cache if that env var is unset -- the cache has no default "
            "budget). Never active generations. Refused under ENGRAM_READONLY=1."
        ),
    )
    pgc.set_defaults(func=cmd_gc, prune=False)

    pf = sub.add_parser("find-def", help="exact symbol definition lookup (no embedding)")
    pf.add_argument("path", help="project root directory")
    pf.add_argument("symbol", help="symbol name (e.g. EmbeddingCache or LanceStore.refresh_fts)")
    pf.add_argument("--ref", default=None, help="search an indexed checkout recorded for this git ref")
    pf.set_defaults(func=cmd_find_def)

    pv = sub.add_parser("eval", help="measure retrieval quality on a query set")
    pv.add_argument("path", help="project root directory")
    pv.add_argument("evalfile", help="JSON list of {query, expected_path, expected_symbol?}")
    pv.add_argument("-k", type=int, default=10, help="results considered per query")
    pv.add_argument("--mode", default="auto", choices=["auto", "hybrid", "vector"])
    pv.add_argument(
        "--rerank",
        action="store_true",
        help=(
            "request rerank; runs only when ENGRAM_RERANK_ENABLED=1 and "
            "mode_used=vector"
        ),
    )
    pv.add_argument("-v", "--verbose", action="store_true", help="print per-query ranks")
    pv.add_argument(
        "--baseline", default=None, metavar="PATH",
        help=(
            "compare measured hit@1/5/10 + MRR (overall and per category) against a "
            "baseline JSON saved by --save-baseline; exits 1 on a non-inferiority failure "
            "(a metric more than --margin below its baseline value)"
        ),
    )
    pv.add_argument(
        "--save-baseline", default=None, metavar="PATH",
        help="write the measured metrics as a new baseline JSON at PATH",
    )
    pv.add_argument(
        "--margin", type=float, default=None,
        help=(
            "max allowed absolute drop below a --baseline metric before it counts as a "
            "regression (default: engram_mcp.evaluate.DEFAULT_NONINFERIORITY_MARGIN, "
            "currently 0.05)"
        ),
    )
    pv.set_defaults(func=cmd_evaluate)

    ps = sub.add_parser("search", help="semantic search over an indexed project")
    ps.add_argument("path", help="project root directory")
    ps.add_argument("query", help="natural-language query")
    ps.add_argument("-k", type=int, default=8, help="number of results")
    ps.add_argument("--lang", default=None, help="filter by language")
    ps.add_argument("--ref", default=None, help="search an indexed checkout recorded for this git ref")
    ps.add_argument("--mode", default="auto", choices=["auto", "hybrid", "vector"],
                    help="retrieval mode (auto routes identifier queries to hybrid, NL to vector)")
    ps.add_argument(
        "--rerank",
        action="store_true",
        help=(
            "request rerank; runs only when ENGRAM_RERANK_ENABLED=1 and "
            "mode_used=vector"
        ),
    )
    ps.set_defaults(func=cmd_search)

    pg = sub.add_parser(
        "grep",
        help="bounded regex probe over indexed chunk text (operator diagnostic, not an MCP tool)",
    )
    pg.add_argument("path", help="project root directory")
    pg.add_argument("pattern", help="Python regex pattern")
    pg.add_argument("--ignore-case", action="store_true", help="case-insensitive match")
    pg.add_argument("--limit", type=int, default=50, help="max paths returned per page")
    pg.add_argument("--offset", type=int, default=0, help="pagination offset over matched paths")
    pg.add_argument("--max-matches", type=int, default=500, help="stop scanning after this many matches")
    pg.add_argument(
        "--max-scan-chunks", type=int, default=10000,
        help="max indexed chunks scanned for this query",
    )
    pg.add_argument(
        "--include-lines", action="store_true",
        help="include matched line text (truncated) alongside line numbers",
    )
    pg.add_argument("--json", action="store_true", help="machine-readable JSON output")
    pg.set_defaults(func=cmd_grep)

    args = p.parse_args(argv)
    # Set up TLS trust (OS store / CA bundle / insecure) before any embedder
    # import triggers a model download.
    from engram_mcp.net import configure_tls

    configure_tls()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
