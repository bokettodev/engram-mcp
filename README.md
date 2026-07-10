# Engram

[![CI](https://github.com/bokettodev/engram-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/bokettodev/engram-mcp/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> Local, private semantic code and prose search for AI coding agents, exposed as
> a self-hosted MCP stdio server.

Engram is a local MCP server for static indexed code search. It chunks source,
embeds locally, stores vectors plus FTS in LanceDB, and exposes `search_code`,
`get_chunk`, and `find_definition` to agents and the CLI. No cloud embeddings or
API keys.

Use ripgrep when you know the exact token; use Engram when you know the behavior,
component, or concept and want an MCP tool to return likely source locations
with context.

> **Model license notice:** Engram's source code is MIT-licensed. The default
> embedder (`ibm-granite/granite-embedding-97m-multilingual-r2`) is Apache-2.0.
> The default reranker model (`jinaai/jina-reranker-v2-base-multilingual`) is
> **CC-BY-NC-4.0** — not licensed for commercial use — and running it inside a
> for-profit organization, even on private internal code and even without
> redistribution, may itself count as commercial use. It is **off by default**
> (set `ENGRAM_RERANK_ENABLED=1` to enable); Engram's own MIT license is
> unaffected, since the restriction attaches only to that model's weights. If
> you're unsure whether your use qualifies, consult your own counsel, or point
> `ENGRAM_RERANKER_MODEL` at a different reranker.

## Contents

- [Requirements](#requirements)
- [Install](#install)
- [Quickstart](#quickstart)
- [CLI](#cli)
- [Use it as an MCP server](#use-it-as-an-mcp-server)
- [How it works](#how-it-works)
- [Embedder](#embedder)
- [Retrieval quality](#retrieval-quality)
- [Performance](#performance)
- [Development](#development)
- [License](#license)

## Requirements

- [**uv**](https://docs.astral.sh/uv/) — manages Python 3.12 + dependencies.
- **Recommended:** an NVIDIA GPU + the `gpu` extra for faster indexing. CPU
  indexing works but is slower. Search always stays on CPU (no torch, ~0 VRAM).

## Install

Engram currently runs from a source checkout:

```bash
git clone https://github.com/bokettodev/engram-mcp.git
cd engram-mcp
uv sync
```

For GPU-accelerated indexing, install the optional GPU extra:

```bash
uv sync --extra gpu
```

CPU indexing works without the extra but is much slower. Search stays on
FastEmbed/ONNX CPU either way.

## Quickstart

```bash
uv run engram index  /path/to/your/repo   # build the index (first run downloads a small model)
uv run engram search /path/to/your/repo "where are http requests retried?"
```

PowerShell:

```powershell
uv run engram index "C:\path\to\your\repo"
uv run engram search "C:\path\to\your\repo" "where are http requests retried?"
```

Example output:

```
$ uv run engram search ./myrepo "retry an http request with backoff"

[src/http/client.py:88-121]   function request_with_retry   (score=0.78)
def request_with_retry(url, *, attempts=3, backoff=0.5):
    for i in range(attempts):
        ...

[src/http/session.py:40-58]   function _sleep_backoff       (score=0.71)
...
```

## CLI

```bash
uv run engram index    <path> [--rebuild] [--gpu|--cpu] [--git-max-commits N]  # build / update
uv run engram search   <path> "<query>" -k 8 [--mode auto|vector|hybrid] [--rerank] [--lang py] [--ref <git-ref>]
uv run engram find-def <path> <symbol> [--ref <git-ref>]  # exact symbol definition lookup
uv run engram eval     <path> evals/self.json [--mode M] [--rerank]  # measure retrieval quality
                        [--save-baseline evals/baseline.json]         # record metrics as the baseline
                        [--baseline evals/baseline.json] [--margin M] # gate a run against the baseline
uv run engram remove   <path>                             # delete a project's index
uv run engram gc       [--dry-run|--prune]                 # report/reclaim orphans, stale generations, embedding cache
uv run engram chunk    <path> [--show N]                  # walk + chunk only (no embedding)
```

`--rerank` requests rerank; it only runs when `ENGRAM_RERANK_ENABLED=1` and
`mode_used=vector`. See [Retrieval quality](#retrieval-quality) and
[License](#license) for the default reranker license.

**`engram gc`** reports (`--dry-run`, the default) or reclaims (`--prune`)
three kinds of disk usage in one JSON payload: orphaned index directories
(unchanged from before), superseded ("stale") LanceDB generation tables +
their catalog sidecars for every indexed project, and the global embedding
cache. A full rebuild writes a fresh `chunks_g<N>` table and atomically swaps
`project.json` to point at it; the *previous* generation is deliberately left
on disk (an in-flight reader may still hold a pointer to it) and, before this,
was only ever cleaned up at the start of the *next* rebuild — a project
rebuilt once and then just queried or incrementally updated afterward kept
that stale generation forever. `engram gc --prune` (and, separately, clean MCP
server startup — see below) now reclaim it explicitly. The current active
generation is never touched by either path. The embedding-cache section is a
no-op unless `ENGRAM_CACHE_MAX_MB` is set (see below); it never deletes a
user's cache by default. Both reclaim paths are refused under
`ENGRAM_READONLY=1`.

Reclaiming is an explicit operator action, not something that happens behind
your back. Engram runs one stdio server process per MCP client window, and they
all share `ENGRAM_HOME` — so a generation one process considers stale may be the
one another live process is mid-search on. Retaining exactly that generation
across a swap is what lets in-flight readers finish, so nothing reclaims it
automatically. Run `engram gc --prune` (preview with `--dry-run`) when no server
is serving. A host that genuinely runs a single server may opt in to the same
reclaim at startup with `ENGRAM_GC_ON_START=1`; it is skipped entirely under
`ENGRAM_READONLY=1` regardless.

Incremental indexing reprocesses changed/added/deleted files and reuses cached
embeddings for unchanged content. `--rebuild` forces a full atomic rebuild (a
crash mid-rebuild leaves the previous index searchable).
Git-history analytics are captured by default during indexing (see
`ENGRAM_GIT_ANALYTICS` below to change the default). Use `--no-git-analytics`
for a zero-git catalog, `--git-analytics` to force them on even when the env
var defaults them off, or `--git-max-commits N` to cap shared history
capture; omitted means full history.

For operators, `engram index --json` emits line-delimited JSON as work proceeds:
progress events use `{"event":"progress","version":1,"seq":N,"stage":"...",
"unit":"...","done":n,"total":n|null,...}` and the last line is
`{"event":"result","version":1,"ok":true,...}` or
`{"event":"result","version":1,"ok":false,"error":"...","code":"...","hint":"..."}`.
Stages include `waiting_for_gpu`, `waiting_for_lock`, `scanning`,
`embedding_plan`, `embedding`, `writing_table`, `writing_fts`, and `done`.

## Use it as an MCP server

Register Engram with your agent, then it can call the tools below itself. Use the
**absolute path to where you cloned the repo**.

**Claude Code:**

```bash
claude mcp add engram -- uv --directory /ABSOLUTE/PATH/TO/engram-mcp run engram-mcp
```

**Cursor** (`.cursor/mcp.json` or global `mcp.json`):

```json
{
  "mcpServers": {
    "engram": {
      "command": "uv",
      "args": ["--directory", "/ABSOLUTE/PATH/TO/engram-mcp", "run", "engram-mcp"]
    }
  }
}
```

**Claude Code on Windows PowerShell:**

```powershell
claude mcp add engram -- uv --directory "C:\Users\you\src\engram-mcp" run engram-mcp
```

**Cursor on Windows** (`.cursor/mcp.json` or global `mcp.json`):

Windows paths in JSON need escaped backslashes.

```json
{
  "mcpServers": {
    "engram": {
      "command": "uv",
      "args": ["--directory", "C:\\Users\\you\\src\\engram-mcp", "run", "engram-mcp"]
    }
  }
}
```

**Indexing device on the server.** Engram has one embedder:
`fastembed:ibm-granite/granite-embedding-97m-multilingual-r2` (384d). Search
always uses FastEmbed/ONNX on CPU. **Indexing defaults to `auto` — it prefers a
CUDA GPU and only falls back to CPU when none is available** (CPU indexing is a
much slower fallback). The MCP `index_project` tool takes `index_device`
(`"auto"` default, or `"cuda"` to require the GPU, or `"cpu"` to force the slow
fallback); `ENGRAM_INDEX_DEVICE` sets the default. Launch with the `gpu` extra so
the GPU path is available:

```bash
claude mcp add engram -- \
  uv --directory /ABSOLUTE/PATH/TO/engram-mcp run --no-sync --extra gpu engram-mcp
```

PowerShell:

```powershell
claude mcp add engram -- uv --directory "C:\Users\you\src\engram-mcp" run --no-sync --extra gpu engram-mcp
```

- Every index job — CPU, GPU, or `auto` — runs in a short-lived subprocess
  (`engram index --json`). The long-lived server process never constructs an
  embedding provider or embeds a passage for an index job, so a running index
  can never contend with the query path's cached-provider inference lock, and
  the server stays ~0 VRAM even after GPU jobs. The GPU path additionally
  needs the torch/sentence-transformers packages (`--extra gpu` /
  `uv sync --extra gpu`); without the extra (or without a GPU) the default
  `auto` transparently uses the CPU fallback, and `index_device="cuda"` fails
  explicitly instead of falling back.
- `index_project` returns a `job_id` immediately, before the project is even
  walked: the walk/chunk/hash/cache-plan step and `auto` device routing both
  run inside the background job, not the tool call. Poll `index_status` for
  `stage` (`planning` → `loading_model` → `embedding`/`writing_table`/… →
  `done`), the `plan` object once computed (delta-aware: counts missing
  unique chunk embeddings; if `missing_unique_chunks <= ENGRAM_DELTA_CPU_MAX`,
  default `1024`, an `auto` job routes to `cpu`), and `routing`
  (`"delta_cpu"` vs `"requested"`).
- A second `index_project` call for a project that already has a job
  queued/running returns that job's id with `coalesced: true` instead of
  starting a duplicate job.
- `cancel_index(job_id)` cancels a queued/running job, killing the index
  subprocess's whole process tree. The job ends in a terminal `cancelled`
  status; since the child is killed before it can reach the atomic
  manifest/generation swap, a previously published index is left intact and
  searchable.
- `ENGRAM_INPROCESS_CPU_MAX` (default `0`, disabled) opts into an in-process
  CPU fast path for deltas at or below this many missing unique chunk
  embeddings, trading the subprocess's fixed startup+model-load overhead for
  lower latency on tiny deltas. It always uses a separate, uncached FastEmbed
  provider instance (never the cached query provider), so it still can't
  contend with the query path's lock — but unlike the subprocess path it
  can't be interrupted mid-batch by `cancel_index`.
- Completed index jobs schedule a one-worker CPU query-model warmup separate
  from the single index worker. Set `ENGRAM_WARMUP_ON_START=1` to also warm the
  query provider at server startup.
- **Windows footgun:** if `uv run` reports `failed to remove … engram-mcp.exe …
  used by another process`, a previous server instance is holding the script
  while `uv` tries to re-sync. Launch with `run --no-sync` (use the already-set-up
  venv, skip the sync) to avoid the lock:
  `uv --directory … run --no-sync engram-mcp`.

**Read-only mode.** Set the env var `ENGRAM_READONLY=1` on the server and only the
read tools (`search_code`, `get_chunk`, `find_definition`, `project_map`,
`doctor_project`, `model_status`, `index_status`,
`list_indexed_projects`, `server_info`) are registered. The mutating tools
(`index_project`, `cancel_index`, `reindex_file`, `remove_project`) are withheld,
so the client physically cannot alter an index. Indexing is then driven
out-of-band via the `engram` CLI/operator. A missing index in read-only mode
does not load or download a model.

```json
{
  "mcpServers": {
    "engram": {
      "command": "engram-mcp",
      "env": { "ENGRAM_READONLY": "1" }
    }
  }
}
```

Tools exposed (indexing is async — `index_project` returns a `job_id` you poll, so a
tool call never blocks for minutes). **13 tools total: 9 read-only + 4 mutating.**
`grep_index` is deliberately not one of them — see "CLI-only diagnostics" below.

| Tool | Purpose |
|---|---|
| `index_project(project_path, full_rebuild=False, index_device=None, git_max_commits=None, git_analytics=None)` | start a background index and return its `job_id` immediately (before the project is walked); a second call for a project with an already-running job returns that job's id with `coalesced: true`. `git_analytics` omitted defers to `ENGRAM_GIT_ANALYTICS` (default: on); an explicit `True`/`False` overrides it |
| `cancel_index(job_id)` | cancel a queued/running index job, killing its subprocess process tree; ends in a terminal `cancelled` status and never disturbs a previously published index |
| `index_status(job_id)` | current-process progress snapshot (stage, counts, timestamps, update sequence, ETA, `plan`, `routing`) |
| `search_code(project_path, query, k=8, language=None, mode="auto", rerank=False, content="preview", max_chars_per_result=800, max_total_chars=None, candidate_k=None, facets=None, min_relevance=None, ref=None)` | compact ranked hits over static indexed source |
| `get_chunk(project_path, chunk_id, max_chars=None, include_neighbors=False, neighbor_window=1, include_parent=False, ref=None)` | fetch full content for one search hit, optionally adjacent/parent context; `ref` addresses the same indexed git ref a hit came from |
| `find_definition(project_path, symbol, ref=None)` | exact symbol definition lookup, with suggestions on miss (no embedding) |
| `project_map(project_path, depth=2, sort="path", dirs_limit=200, dirs_offset=0, include_files=False, files_limit=50, files_offset=0, include_symbols=False, symbols_limit=20, code_only=False, languages=None, chunk_roles=None, kinds=None, path_prefix=None, path_glob=None, symbol_kinds=None, min_symbols=0, non_empty=True, include_git=None, group_by="commit", ticket_regex=None, window_hours=2.0, git_max_commits=None, recent_days=90, max_files_per_change=50, cochange_limit=5, hotspots_limit=25)` | body-free dirs by default; compact file rows are opt-in and paginated; filters compose and report `filtered_totals`; VCS analytics default to `ENGRAM_GIT_ANALYTICS` (on) when `include_git` is omitted, can be disabled per call with `include_git=False`, and always report `status="disabled"` (never a live git walk) for a project indexed with analytics off |
| `doctor_project(project_path, check_git=True)` | use before debugging empty/odd results; returns `ok`, `summary`, `git`, `storage`, and `issues[]`. `storage` reports active-vs-stale generation bytes, catalog sidecar bytes, and the global embedding-cache size/entry count -- purely by stat-ing files already on disk, so it never opens LanceDB or creates a directory (a read tool must not, even for a project that was never indexed) |
| `model_status(project_path=None)` | reports whether the project's recorded query model is loaded/loading/not_loaded in this process |
| `reindex_file(project_path, rel_path)` | incrementally re-index/drop one file |
| `remove_project(project_path)` | delete a project's index |
| `list_indexed_projects(limit=50, cursor=None, verbose=False, prune_orphans=False)` | compact paginated on-disk index inventory, `data_home`, broken manifest/table `errors[]`, and orphan-GC summary |
| `server_info()` | data-home, read-only, embedder/index-device diagnostics, reranker enable state, ONNX probe, default model, candidate default, and CC-BY-NC note |

**CLI-only diagnostics.** `grep_index` (bounded Python regex probe over indexed
chunk text; counts/line numbers by default, snippets with `include_lines=true`)
is not part of the MCP tool surface: it approximates grep over overlapping
chunks (can double-count or miss cross-chunk matches) and agents already have
better exact-search tools available. The capability still exists as
`engram grep <path> <pattern> [--ignore-case] [--limit N] [--offset N]
[--max-matches N] [--max-scan-chunks N] [--include-lines] [--json]` and as
`engram_mcp.diagnostics.grep_index` for operators/scripts. It accepts
`ignore_case`, `limit`, `offset`, `max_matches`, `max_scan_chunks`, and
`include_lines`; it is capped by `max_matches`, `max_scan_chunks`, and
`ENGRAM_GREP_REGEX_TIMEOUT_SEC`.

`index_status` includes `created_at`, `started_at`, `updated_at`,
`finished_at`, `duration_sec`, `seconds_since_update`, and `update_seq`. The
`progress` object is `{ "unit": "...", "done": n, "total": n|null }`; unknown
totals are reported as `null`, not `0`.

`search_code(ref=...)`, `find_definition(ref=...)`, and `get_chunk(ref=...)`
select an already indexed checkout/ref for the same logical git project. A ref
miss returns `E_REF_NOT_INDEXED`; it does not fall back to the default index —
so a `chunk_id` returned by a ref-scoped `search_code` hit stays addressable
by passing the same `ref` back to `get_chunk`.

`search_code` is a decision tool first: by default each hit contains `chunk_id`,
`rel_path`, `span` (`{start_line, end_line}`), `symbol`, `symbol_kind`, `chunk_role`,
`preview`, `raw_score`, `score_normalized`, `relevance` (`high|medium|low|uncertain`),
`matched`, `matched_in` (which of `content`/`symbol`/`path` the query tokens hit),
`match_reason`, `stale`, `index_stale`, and `truncated`. Each hit carries exactly one
field per signal — no `path` alias of `rel_path`, no separate top-level `start_line`/
`end_line` alongside `span`, no `excerpt` alias of `preview`, and no `score` alias of
`raw_score` — measured 37% smaller for a real 8-hit query against this repo's own
index (16090 -> 10128 response characters). It also
returns `mode_requested`, `mode_used`, `warnings[]`, `hints[]`, `total_matches`, optional
`facets`, `rerank_applied`, `source_type: "static_indexed_source"`, and a
`dirty` freshness summary. There is no top-level `map[]` — it duplicated `results[]`
and has been removed; derive a compact index from `results[]` directly if needed.
It also includes a top-level `source_revision`
object with indexed/current `worktree_root`, `ref`, `commit`, and `dirty`
fields plus `stale`, mismatch flags, and `reasons[]`. Use `content="none"` to get metadata only,
`content="full"` for bounded inline text, or `get_chunk` for exact full content
by `chunk_id`. `k` and `candidate_k` above `50` are clamped down to `50` (not
rejected) with a warning in `warnings[]`; below `1`, or not an integer, is
still a request error. If omitted, `candidate_k` resolves from
`ENGRAM_RERANK_CANDIDATE_K` (default `20`, clamped to `1..50`) and is raised to
at least `k` internally. `max_total_chars` caps aggregate returned
body/preview text across results.
Valid `facets` are `dir`, `language`, `chunk_role`, `kind`; `facets.scope` says
whether counts are exact FTS, capped lower bound, or vector candidate estimate.
`min_relevance` filters to `uncertain|low|medium|high` and stricter. `dirty`,
`stale`, and `index_stale` are per-file mtime/size freshness. The top-level
`source_revision` object is the single git revision signal; search no longer
returns a top-level `git` object or per-hit `git_stale`.

`total_matches.fts_exact` is an exact LanceDB FTS/BM25 metadata scan when hybrid
FTS is available, `facets` was requested, and the body-free metadata scan
completes under `ENGRAM_FTS_COUNT_MAX_SCAN` (default `50000`). If the scan
hits that cap, `total_matches.fts_exact` is reported with `exact: false`,
`capped: true`, and the returned count is a lower bound. When `facets` is
*not* requested in hybrid mode, that second metadata scan does not run at
all -- `total_matches.fts_exact` is reported `available: false` with a
`reason` instead of paying for a count nothing reads; requesting `facets`
gets you the exact same numbers as before. `total_matches.vector_estimate` is
explicitly a candidate-pool estimate above a relative similarity threshold,
never an exact vector count.
If the query model is not loaded yet, search waits up to
`ENGRAM_SEARCH_WAIT_SEC` seconds (default `8`) for the FastEmbed/ONNX CPU load
future before returning `E_MODEL_LOADING` with `retry_after_sec`.

Search and rerank environment knobs:

| Env var | Default | Purpose |
|---|---:|---|
| `ENGRAM_SEARCH_WAIT_SEC` | `8` | how long search waits for the CPU query model warmup before returning `E_MODEL_LOADING` |
| `ENGRAM_RERANK_ENABLED` | `off` | master switch for reranking. Off by default: a per-call `rerank=true` is ignored and **no reranker model is ever loaded/downloaded** until an operator sets this (`1`/`true`/`yes`/`on`) |
| `ENGRAM_RERANK_CANDIDATE_K` | `20` | default rerank/search candidate pool when `candidate_k` is omitted; clamped to `1..50` |
| `ENGRAM_RERANKER_MODEL` | `jinaai/jina-reranker-v2-base-multilingual` | FastEmbed ONNX reranker model used only when `ENGRAM_RERANK_ENABLED=1`, per-call `rerank=true`, and `mode_used=="vector"`. MCP/CLI search cannot override the model per call |
| `ENGRAM_FTS_COUNT_MAX_SCAN` | `50000` | maximum body-free FTS metadata rows scanned for `total_matches.fts_exact`; hitting the cap reports a lower bound |
| `ENGRAM_GIT_STALENESS` | `on` | set to `0`/`false`/`no`/`off` to disable all git staleness and analytics probes |
| `ENGRAM_GIT_ANALYTICS` | `on` | default for whether indexing/`project_map` compute git-history/SZZ analytics at all; set to `0`/`false`/`no`/`off` to default it off. An explicit `--git-analytics`/`--no-git-analytics` CLI flag or MCP `index_project` `git_analytics` argument always overrides this env var for that call. A project indexed with analytics disabled reports `project_map` `git_analytics.status == "disabled"` instead of doing a live git walk on the request path |
| `ENGRAM_GIT_INDEX_TIMEOUT` | `120` | timeout in seconds for heavy index-time shared git history walks and SZZ diff/blame subprocesses; search-time staleness checks keep their short 3 second cap |
| `ENGRAM_GREP_REGEX_TIMEOUT_SEC` | `2` | regex execution timeout for `grep_index`; clamped to `0.05..30` seconds |
| `ENGRAM_USER_REGEX_TIMEOUT_SEC` | `1` | validation timeout for caller-supplied `ticket_regex`/`git_fix_regex` patterns before they are allowed to run in-process |
| `ENGRAM_CACHE_MAX_MB` | unset (unlimited) | opt-in retention budget for the global embedding cache (`ENGRAM_HOME/global-cache/embeddings.sqlite`), in MB. Unset/`0`/negative means unlimited -- the cache never auto-deletes anything by default, since an evicted entry is invisible until the next time that exact chunk needs embedding again (then it's just a cache miss, silently re-paid). Only takes effect via `engram gc --prune` or the startup task (`ENGRAM_GC_ON_START`); never applied from a search or index path |
| `ENGRAM_GC_ON_START` | `off` | set to `1`/`true`/`yes`/`on` to run the stale-generation reclaim (and, if `ENGRAM_CACHE_MAX_MB` is set, the embedding-cache prune) once in the background at server startup. Off by default: one server process runs per MCP client window and they share `ENGRAM_HOME`, so a generation this process calls stale may be the one another live process is mid-search on. Prefer the explicit `engram gc --prune`. Always skipped under `ENGRAM_READONLY=1` regardless of this setting |

A body-free `catalog_g<N>.json` sidecar is generated during indexing and tied to
the active Lance generation. It powers `project_map`, structural `kind` facets
(`test`, `config`, `migration`, `doc`, marked inferred), neighborhood lookup,
and doctor checks without loading a model or copying source bodies. Every
search validates the catalog against a commit token (generation, active
table, row count, and a chunk-id digest) written once, at build time, into
both the catalog sidecar and the project manifest -- an O(1) string
comparison instead of re-scanning the active table on every query. The full
O(total chunks) id-set comparison the token replaced still runs, but only in
`doctor_project`, a diagnostic tool where that cost is expected.
`project_map` returns `totals`, `dirs`, and `files`; `depth` is clamped to
`0..20`, `dirs_limit`/`files_limit`/`symbols_limit` to `0..1000`, and `sort`
to `path|files|chunks|symbols`.
`include_git` defaults to `ENGRAM_GIT_ANALYTICS` when omitted (built-in
default: on). With it effectively true, `project_map` attaches
`git_analytics` plus per-file `git` rows when files are included.
`git_analytics.status` is `ready`, `freshened`, `uncached`, `unavailable`, or
`disabled`; it reports `cache_head`, `current_head`, and `freshened_commits`.
`disabled` means the project was indexed with git analytics off
(`git_analytics_enabled=false` in its manifest) — `project_map` never falls
back to a live request-path git walk in that case, even with an explicit
`include_git=True`; re-index with analytics enabled instead. `git_max_commits`
similarly defaults to the cap recorded on the project's manifest at index
time when omitted. Shared history and SZZ live under
`git-analytics/<logical_project_id>` so worktrees of the same repository reuse
one repo-wide cache. Indexing writes the shared history keyed by a full
all-refs fingerprint; reads compare that cached fingerprint to the current
fingerprint, freshen by rebuilding the in-memory response when refs differ, and
do not write the shared sidecar.
Per-file git rows include churn/change counts, `fix_density`, co-change rules
ranked by lift, indentation complexity, and hotspot quadrant. Once the
generation's SZZ sidecar is ready, rows also include SZZ defect signals
(`defect_introducing_commits`, `defect_introducing_lines`,
`defect_hotspot_score`); before that, `git_analytics.szz.status` reports
`computing`, `partial`, or `unavailable` and defect fields are omitted.
Indentation complexity is the sum of leading
indentation depth over non-blank, non-comment-ish lines, with one depth unit per
4 columns and tabs advancing to the next tab stop. SZZ attribution detects fix
commits with the default regex
`(?i)\b(fix(e[sd])?|bug|hotfix|patch|close[sd]?\s+#\d+)\b`, diffs each fix
against its parent, blames removed/changed parent lines with `git blame -w`,
and caches per-fix-commit attributions in the shared `szz.json` sidecar written
atomically by a background index-time task after the core generation is already
active. If a prior run left SZZ at `computing` or `partial`, the next index
resumes it from the per-fix-commit cache; only `ready` is terminal.
Blames run through an adaptive bounded thread pool:
`min(8, max(2, (os.cpu_count() or 2) - 1))`, overrideable with
`ENGRAM_SZZ_WORKERS`.

### Server-side limits

Every externally supplied limit is clamped to a documented server-side
maximum rather than rejected: a request above the maximum still runs, capped
at that maximum, and the response's `warnings[]` (or `git_analytics.warnings`
for the analytics limits below) says the value was clamped and to what.
Values *below* the valid range (or the wrong type) are still request errors —
clamping only ever protects the server from an excessively *large* ask.

| Limit | Server maximum | Where reported |
|---|---:|---|
| `search_code` `k` | `50` | `warnings[]` |
| `search_code`/`get_chunk` `candidate_k` | `50` | `warnings[]` |
| `search_code` `max_chars_per_result` | `20000` | `warnings[]` |
| `search_code` `max_total_chars` | `200000` | `warnings[]` |
| `get_chunk` `max_chars` | `20000` | `warnings[]` |
| `grep_index` `limit` | `200` | `warnings[]` |
| `grep_index` `max_matches` | `5000` | `warnings[]` |
| `grep_index` `max_scan_chunks` | `100000` | `warnings[]` |
| `project_map` `depth` | `20` | `warnings[]` |
| `project_map` `dirs_limit`/`files_limit`/`symbols_limit` | `1000` | `warnings[]` |
| `project_map` `path_glob` pattern | `512` chars / `64` segments | pattern silently matches nothing past the cap (no error, no scan) |
| `project_map` `max_files_per_change` | `500` | `git_analytics.warnings` |
| `project_map` `cochange_limit` | `100` | `git_analytics.warnings` |
| `project_map` `hotspots_limit` | `500` | `git_analytics.warnings` |
| `project_map`/`index_project` `git_max_commits` | `1000000` | `git_analytics.warnings` |

`max_files_per_change` doubles as a guard on `gitanalytics.cochange`'s
association-rule generation, which is quadratic in a change set's file count;
capping the caller-supplied value (not just the top-N outputs from
`cochange_limit`/`hotspots_limit`) keeps a single oversized changeset from
producing quadratic work. `grep_index`'s pre-scan corpus is streamed to its
regex-timeout subprocess in small batches (see "How it works") rather than
copied whole, so a large `max_scan_chunks` no longer means a large up-front
copy outside the timeout window.

`list_indexed_projects` is compact by default and reads only `project.json`
manifests, so it does not open LanceDB tables or count rows unless
`verbose=true`. Each compact project has `project_id`, `root_path`,
`root_exists`, `files`, `chunks`, `indexed_at`, `embedder_id`, `generation`, and
git metadata when present in the manifest (current indexes record git
worktree/ref/commit/dirty state). Pass the returned `cursor` to fetch the next
page. `prune_orphans` defaults to `false`: a listing call must never delete an
index merely because its recorded root looks momentarily missing
(disconnected drive, unmounted share, renamed workspace). Deleting orphaned
index directories is an explicit operator action via the CLI:
`engram gc --dry-run` to preview or `engram gc --prune` to delete. Passing
`prune_orphans=true` to `list_indexed_projects` still works for callers that
explicitly opt in, and in `ENGRAM_READONLY=1` pruning is forced off
regardless of what a caller passes.

All read-tool errors use a structural shape: `{ "error": "...", "code": "...",
"hint": "..." }`. Codes include `E_PROJECT_NOT_INDEXED`, `E_REF_NOT_INDEXED`,
`E_INDEX_INVALID`, `E_UNKNOWN_PROFILE`, `E_EXTRA_MISSING`, `E_MODEL_LOAD_FAILED`,
`E_MODEL_LOADING`, and `E_BAD_REQUEST`. TLS certificate download guidance is
returned as the `hint` on `E_MODEL_LOAD_FAILED`.

## How it works

```
walk (gitignore-aware, skips binary/generated)
  → chunk by symbol (tree-sitter) · markdown heading section · prose paragraph (+ line-window fallback)
  → embed each chunk locally, with a "path / symbol / language" header
  → store vectors + metadata in LanceDB (+ a full-text index for hybrid search)

search:  embed the query → nearest vectors  (+ optional BM25 hybrid + reranker)
```

Embeddings are cached by content hash, so re-indexing unchanged code costs nothing
and a full rebuild swaps the index in atomically.

**Where data lives** (one index per project, outside the repo):
`%LOCALAPPDATA%\engram` on Windows, `$XDG_DATA_HOME/engram` on Linux, else `~/.engram`.
Override with `ENGRAM_HOME`. It stores your code — treat it as private.

The global content-hash embedding cache lives at
`ENGRAM_HOME/global-cache/embeddings.sqlite`, shared (deliberately) across
every project and worktree so identical chunks are only ever embedded once.
It has no size cap by default -- see `ENGRAM_CACHE_MAX_MB` above, and
`engram gc` / `doctor_project`'s `storage` section for its current size.

**Model weights** are a separate, shared cache: downloaded once into the
Hugging Face cache (`~/.cache/huggingface`, or wherever `HF_HOME` points) and
reused across projects - not under `ENGRAM_HOME`. The Granite 97m embedder is
small (~0.2 GB); the optional `gpu` extra also installs torch wheels.

## Embedder

Engram supports one embedder:

| Model | Canonical id | Dim | Search backend | Index backend |
|---|---|---:|---|---|
| granite-embedding-97m-multilingual-r2 | `fastembed:ibm-granite/granite-embedding-97m-multilingual-r2` | 384 | FastEmbed/ONNX CPU | sentence-transformers CUDA by default (auto); FastEmbed/ONNX CPU fallback |

[Granite R2](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2)
(IBM, Apache-2.0) is multilingual (100+ languages incl. Russian) + code. Engram
pins it to an exact upstream commit revision (see `config.py`) rather than a
mutable branch name, so an upstream weight/tokenizer swap under the same repo
name can't silently start mixing old cached vectors with new ones — the
canonical id folds in the revision, dimension, pooling, and backend, and a
mismatch forces a rebuild instead of serving. The canonical `fastembed:` id
(now `fastembed:<repo>@<revision>#dim=384;pool=cls;norm=1`) is recorded in the
index manifest and used in the embedding cache even when indexing was
produced by the CUDA backend (the ONNX and CUDA vectors match to cosine
0.99997, checked by `tests/test_model_fingerprint.py`'s parity test whenever
the `gpu` extra is installed). That keeps search torch-free: a CUDA-built
index is still queried with FastEmbed/ONNX on CPU.

```bash
uv run --extra gpu engram index <path>        # default: GPU if available, else CPU
uv run --extra gpu engram index <path> --gpu  # require GPU (error if none)
uv run engram index <path> --cpu              # force the slow CPU fallback (no torch)
```

The default `auto` prefers the GPU and quietly uses the CPU fallback when no
usable CUDA GPU is present. Explicit `--gpu` / `index_device="cuda"` never falls
back — if `sentence-transformers`/torch or CUDA is missing it returns a
structured error, so a misconfigured "must be GPU" job fails loudly instead of
silently crawling on CPU.

### Behind a corporate TLS-inspecting proxy

Model downloads go over HTTPS and verify certificates. On a corporate network
whose proxy injects a self-signed root CA, Python's bundled `certifi` doesn't
trust it (even though `curl`/the OS do) and the first download fails with
`CERTIFICATE_VERIFY_FAILED`. Engram handles this automatically: by default it
verifies against the **OS trust store** (`truststore`), where the corporate CA
already lives. If you still hit issues:

- point `SSL_CERT_FILE` at your corporate CA bundle (`.pem`) — Engram mirrors it
  to `REQUESTS_CA_BUNDLE` / `CURL_CA_BUNDLE` for every HTTP client;
- disable the OS-trust path with `ENGRAM_SYSTEM_TRUST=0`;
- last resort, on a trusted network: `ENGRAM_INSECURE_DOWNLOADS=1` skips
  verification for downloads entirely.

### GPU memory

Search never loads torch and uses **~0 VRAM**. The long-lived MCP server never
initializes CUDA either: a GPU index job (the default `auto` when a GPU is
present, or explicit `cuda`) runs in a **short-lived subprocess** that loads
Granite on CUDA, indexes, and exits — so its entire CUDA context (not just the
model) is reclaimed when the job ends, and the server process itself stays
torch-free at all times. From the CLI, `engram index` is likewise a single
process that frees everything on exit.

- **`ENGRAM_ST_BATCH_SIZE`** (default 16) caps the encode batch — the real lever
  on activation VRAM during indexing. Lower it (e.g. 8) on a tight/shared GPU;
  raise it for throughput on a dedicated one.
- GPU index jobs acquire a machine-wide lock at
  `ENGRAM_HOME/locks/gpu-index.lock` before loading the CUDA model and release it
  when the provider is cleaned up. While a job is blocked there,
  `index_status` reports `stage="waiting_for_gpu"` and
  `seconds_since_update` shows the wait age.
- VRAM is held only for the duration of a GPU index job, by the child process,
  never by the always-on search server.

## Retrieval quality

Search modes (`--mode`, default `auto`):

- **auto** — routes identifier/literal queries (`EmbeddingCache`, `MAX_FILE_BYTES`,
  `"file not found"`) to hybrid, and plain natural-language queries to vector.
- **vector** — dense embedding similarity.
- **hybrid** — vector + full-text (BM25), fused with reciprocal-rank fusion + symbol/path boosts.
- **`--rerank`** — opt-in local cross-encoder reranking for the top candidates.
  Reranking is **off unless the operator sets `ENGRAM_RERANK_ENABLED=1`** — a hard
  guarantee that the ~1.1 GB ONNX cross-encoder never loads on the always-on
  server from a stray `rerank=true`. With the switch on, rerank is still per-call
  (`rerank=true`) and **gated to vector mode**: identifier/literal/error-string queries
  route to `hybrid` and are left un-reranked (a semantic cross-encoder demotes
  exact symbol definitions behind tests/helpers). Pass `mode="vector"` to force
  rerank on a query that would otherwise route to hybrid. When skipped, the
  response reports `rerank_skipped_reason`.

Run `uv run engram eval <path> evals/self.json [--rerank]` to compare baseline
and gated rerank.

Reranking metadata includes `candidate_k`, `rerank_model`, and
`rerank_latency_ms`. The only reranker backend is torch-free FastEmbed ONNX on
CPU: `jinaai/jina-reranker-v2-base-multilingual`, selected through
`fastembed.rerank.cross_encoder.TextCrossEncoder`. It keeps the always-on MCP
server at ~0 VRAM and requires no runtime LLM call. The model is a ~278M
cross-encoder, so CPU reranking has a real latency cost: keep the default
candidate pool conservative (`ENGRAM_RERANK_CANDIDATE_K=20`) and raise it only
when quality needs it. On CPU, `candidate_k=50` can take seconds per query.
`ENGRAM_RERANKER_MODEL` selects the FastEmbed ONNX model; MCP/CLI search cannot
override the model or backend per call.

**Reranker license:** The default ONNX reranker,
`jinaai/jina-reranker-v2-base-multilingual`, is licensed **CC-BY-NC-4.0** and is
not licensed for commercial use; running it inside a for-profit organization —
even on private internal code, even without redistribution — may itself count
as commercial use. The MIT license in this repository covers Engram's code, not
model weights downloaded from Hugging Face/FastEmbed. A normal install/index/search
does not load this model because reranking is default-off; it is downloaded only
after an operator sets `ENGRAM_RERANK_ENABLED=1` and requests reranking. If
you're unsure whether your use qualifies as commercial, consult your own
counsel, or choose a different reranker model via `ENGRAM_RERANKER_MODEL` and
verify both its license and retrieval quality before distribution.

A built-in `eval` harness reports hit@1/5/10, MRR, HNSR@5/10, delta-rank, and
lexical-overlap buckets **per query category**. Fixture generation is offline:
`evals/generate_fixture.py` writes a hand-seeded paraphrase/distractor starter
set and marks the build-time LLM integration point; runtime eval stays LLM-free.
HNSR means the target is inside cutoff while the hard negative is outside;
delta-rank is hard-negative rank minus target rank; overlap buckets come from
a query/target token Jaccard computed against the **expected chunk(s)** for a
case (the actual retrieval target(s): the file narrowed to `expected_symbol`
when the case sets one), not the whole file -- a single-file-wide comparison
mostly measures file length, not query/target similarity.

### Baseline / regression gate

`evals/baseline.json` is a checked-in snapshot of `engram eval`'s metrics
(overall + per category hit@1/5/10 and MRR) for `evals/self.json` at
`--mode auto`, regenerated with `--save-baseline`:

```bash
uv run engram eval <path> evals/self.json --mode auto --save-baseline evals/baseline.json
```

`--baseline evals/baseline.json` gates a run against it: a metric only fails
if it drops more than `--margin` (default `0.05`, absolute) below its
recorded baseline value -- **non-inferiority**, not an exact match, because
embedding is not bit-reproducible across platforms/onnxruntime versions.
Exits 1 on a real regression:

```bash
uv run engram eval <path> evals/self.json --mode auto --baseline evals/baseline.json
```

The real-model CI lane (see below) runs this after every real index build, so
a retrieval regression fails the build instead of landing silently on a green
run that never exercised the real embedder/reranker.

## Performance

GPU indexing is the intended fast path; CPU indexing works but is slower.
Re-indexing reuses content-addressed embeddings. Search uses CPU query embedding
plus LanceDB retrieval.

## Stack

| Concern | Choice |
|---|---|
| Runtime | Python 3.12 via [`uv`](https://docs.astral.sh/uv/) |
| Embedder | Granite R2 97m via [FastEmbed](https://github.com/qdrant/fastembed) for CPU search/index; optional [sentence-transformers](https://www.sbert.net/) CUDA for index-only acceleration |
| Vector store | [LanceDB](https://lancedb.com/) (embedded, on-disk, vector + full-text) |
| Chunker | tree-sitter — 11 languages (py/js/ts/tsx/go/rust/java/c/cpp/ruby/c#) · markdown by heading section · plain text by paragraph · line-window fallback |
| Server | MCP Python SDK (FastMCP, stdio) |

## Development

```bash
uv run --no-sync pytest -q   # model- and GPU-gated tests skip without the model / `--extra gpu`
```

## License

[MIT](LICENSE) for Engram's source code.

Third-party model weights keep their own licenses. The default Granite embedder
is Apache-2.0. The default Jina FastEmbed ONNX reranker is CC-BY-NC-4.0 and is
not licensed for commercial use — including private internal use inside a
for-profit organization, which may itself qualify as commercial use even
without redistribution — unless your use is compatible with that license or
you have separate commercial rights. It is off by default
(`ENGRAM_RERANK_ENABLED`); swap in a different model via `ENGRAM_RERANKER_MODEL`
if needed.
