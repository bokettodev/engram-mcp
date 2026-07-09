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
> The default reranker model
> (`jinaai/jina-reranker-v2-base-multilingual`) is **CC-BY-NC-4.0**. Reranking
> is disabled unless the operator sets `ENGRAM_RERANK_ENABLED=1`; when enabled,
> FastEmbed may download that non-commercial model. Do not use the default
> reranker for commercial use or redistribution unless your use is compatible
> with CC-BY-NC-4.0 or you have separate commercial rights. Commercial users
> should configure and verify a commercially usable reranker instead.

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
uv run engram index    <path> [--rebuild] [--gpu|--cpu]   # build / update (defaults to GPU if available, else CPU)
uv run engram search   <path> "<query>" -k 8 [--mode auto|vector|hybrid] [--rerank] [--lang py]
uv run engram find-def <path> <symbol>                    # exact symbol definition lookup
uv run engram eval     <path> evals/self.json [--mode M] [--rerank]  # measure retrieval quality
uv run engram remove   <path>                             # delete a project's index
uv run engram gc       [--dry-run|--prune]                 # find/prune indexes whose root path is gone
uv run engram chunk    <path> [--show N]                  # walk + chunk only (no embedding)
```

`--rerank` requests rerank; it only runs when `ENGRAM_RERANK_ENABLED=1` and
`mode_used=vector`. See [Retrieval quality](#retrieval-quality) and
[License](#license) for the default reranker license.

Incremental indexing reprocesses changed/added/deleted files and reuses cached
embeddings for unchanged content. `--rebuild` forces a full atomic rebuild (a
crash mid-rebuild leaves the previous index searchable).

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

- The GPU path needs the torch/sentence-transformers packages (`--extra gpu` /
  `uv sync --extra gpu`). A GPU index job runs in a short-lived subprocess, so
  the long-lived server never loads torch and stays ~0 VRAM even after GPU jobs.
  Without the extra (or without a GPU) the default `auto` transparently uses the
  CPU fallback; `index_device="cuda"` fails explicitly instead of falling back.
- Delta routing: before an `auto` job, the server runs a torch-free plan that
  counts missing unique chunk embeddings. If
  `missing_unique_chunks <= ENGRAM_DELTA_CPU_MAX` (default `1024`), it routes the
  job to in-process FastEmbed CPU and skips the CUDA subprocess. Explicit
  `cpu`/`cuda` still force their path. The job response reports requested vs
  routed device and the plan used for the route.
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
`doctor_project`, `grep_index`, `model_status`, `index_status`,
`list_indexed_projects`, `server_info`) are registered. The mutating tools
(`index_project`, `reindex_file`, `remove_project`) are withheld, so the client
physically cannot alter an index. Indexing is then driven out-of-band via the
`engram` CLI/operator. A missing index in read-only mode does not load or
download a model.

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
tool call never blocks for minutes):

| Tool | Purpose |
|---|---|
| `index_project(project_path, full_rebuild=False, index_device=None)` | start a background index; returns `job_id`, requested/routed device, routing, and plan |
| `index_status(job_id)` | current-process progress snapshot (stage, counts, timestamps, update sequence, ETA) |
| `search_code(project_path, query, k=8, language=None, mode="auto", rerank=False, content="preview", max_chars_per_result=800, max_total_chars=None, candidate_k=None, facets=None, min_relevance=None)` | compact ranked hits over static indexed source |
| `get_chunk(project_path, chunk_id, max_chars=None, include_neighbors=False, neighbor_window=1, include_parent=False)` | fetch full content for one search hit, optionally adjacent/parent context |
| `find_definition(project_path, symbol)` | exact symbol definition lookup, with suggestions on miss (no embedding) |
| `project_map(project_path, depth=2, sort="path", limit=200)` | body-free `totals`, `dirs`, and `files`; `depth=0..20`, `limit=1..1000`, `sort=path|files|chunks|symbols` |
| `doctor_project(project_path, check_git=True)` | use before debugging empty/odd results; returns `ok`, `summary`, `git`, and `issues[]` |
| `grep_index(project_path, pattern, ...)` | bounded Python regex probe; counts/line numbers by default, snippets with `include_lines=true` |
| `model_status(project_path=None)` | reports whether the project's recorded query model is loaded/loading/not_loaded in this process |
| `reindex_file(project_path, rel_path)` | incrementally re-index/drop one file |
| `remove_project(project_path)` | delete a project's index |
| `list_indexed_projects(limit=50, cursor=None, verbose=False, prune_orphans=True)` | compact paginated on-disk index inventory, `data_home`, broken manifest/table `errors[]`, and orphan-GC summary |
| `server_info()` | data-home, read-only, embedder/index-device diagnostics, reranker enable state, ONNX probe, default model, candidate default, and CC-BY-NC note |

`index_status` includes `created_at`, `started_at`, `updated_at`,
`finished_at`, `duration_sec`, `seconds_since_update`, and `update_seq`. The
`progress` object is `{ "unit": "...", "done": n, "total": n|null }`; unknown
totals are reported as `null`, not `0`.

`search_code` is a decision tool first: by default each hit contains `chunk_id`,
`rel_path`/`span`, `symbol`, `symbol_kind`, `chunk_role`, `preview`, `raw_score`,
`score_normalized`, `relevance` (`high|medium|low|uncertain`), `matched`,
`match_reason`, `stale`, and `truncated`. It also returns `mode_requested`,
`mode_used`, `warnings[]`, `hints[]`, `map[]`, `total_matches`, optional
`facets`, `rerank_applied`, `source_type: "static_indexed_source"`, and a
`dirty` freshness summary. Use `content="none"` to get metadata only,
`content="full"` for bounded inline text, or `get_chunk` for exact full content
by `chunk_id`. `k` and `candidate_k` are bounded to `1..50`. If omitted,
`candidate_k` resolves from `ENGRAM_RERANK_CANDIDATE_K` (default `20`, clamped
to `1..50`) and is raised to at least `k` internally. `max_total_chars` caps
aggregate returned body/excerpt text across results.
Valid `facets` are `dir`, `language`, `chunk_role`, `kind`; `facets.scope` says
whether counts are exact FTS, capped lower bound, or vector candidate estimate.
`min_relevance` filters to `uncertain|low|medium|high` and stricter. `dirty` is
per-file index freshness; `git.git_stale` is manifest-vs-current repo state and
does not affect ranking.

`total_matches.fts_exact` is an exact LanceDB FTS/BM25 metadata scan when hybrid
FTS is available and the body-free metadata scan completes under
`ENGRAM_FTS_COUNT_MAX_SCAN` (default `50000`). If the scan hits that cap,
`total_matches.fts_exact` is reported with `exact: false`, `capped: true`, and
the returned count is a lower bound. `total_matches.vector_estimate` is
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
| `ENGRAM_GREP_REGEX_TIMEOUT_SEC` | `2` | regex execution timeout for `grep_index`; clamped to `0.05..30` seconds |

A body-free `catalog_g<N>.json` sidecar is generated during indexing and tied to
the active Lance generation. It powers `project_map`, structural `kind` facets
(`test`, `config`, `migration`, `doc`, marked inferred), neighborhood lookup,
and doctor checks without loading a model or copying source bodies.
`project_map` returns `totals`, `dirs`, and `files`; `depth` is clamped to
`0..20`, `limit` to `1..1000`, and `sort` to `path|files|chunks|symbols`.
`grep_index` accepts `ignore_case`, `limit`, `offset`, `max_matches`,
`max_scan_chunks`, and `include_lines`; it is capped by `max_matches`,
`max_scan_chunks`, and `ENGRAM_GREP_REGEX_TIMEOUT_SEC`.

`list_indexed_projects` is compact by default and reads only `project.json`
manifests, so it does not open LanceDB tables or count rows unless
`verbose=true`. Each compact project has `project_id`, `root_path`,
`root_exists`, `files`, `chunks`, `indexed_at`, `embedder_id`, `generation`, and
git metadata when present in the manifest (current indexes record git
worktree/ref/commit/dirty state). Pass the returned `cursor` to fetch the next
page. With `prune_orphans=true`, list deletes index directories whose
manifest `root_path` no longer exists and reports them under `gc.pruned`; use
`engram gc --dry-run` or `engram gc --prune` for the same orphan rule from the
CLI.

All read-tool errors use a structural shape: `{ "error": "...", "code": "...",
"hint": "..." }`. Codes include `E_PROJECT_NOT_INDEXED`, `E_INDEX_INVALID`,
`E_UNKNOWN_PROFILE`, `E_EXTRA_MISSING`, `E_MODEL_LOAD_FAILED`,
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
(IBM, Apache-2.0) is multilingual (100+ languages incl. Russian) + code. The
canonical `fastembed:` id is recorded in the index manifest and used in the
embedding cache even when indexing was produced by the CUDA backend (the ONNX
and CUDA vectors match to cosine 0.99997). That keeps search torch-free: a
CUDA-built index is still queried with FastEmbed/ONNX on CPU.

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
`rerank_latency_ms`. The default reranker is torch-free FastEmbed ONNX on CPU:
`jinaai/jina-reranker-v2-base-multilingual`, selected through
`fastembed.rerank.cross_encoder.TextCrossEncoder`. It keeps the always-on MCP
server at ~0 VRAM and requires no runtime LLM call. The model is a ~278M
cross-encoder, so CPU reranking has a real latency cost: keep the default
candidate pool conservative (`ENGRAM_RERANK_CANDIDATE_K=20`) and raise it only
when quality needs it. On CPU, `candidate_k=50` can take seconds per query.
`ENGRAM_RERANKER_MODEL` selects the FastEmbed ONNX model; MCP/CLI search cannot
override the model or backend per call.

**Reranker license:** The default ONNX reranker,
`jinaai/jina-reranker-v2-base-multilingual`, is licensed **CC-BY-NC-4.0**. The
MIT license in this repository covers Engram's code, not model weights downloaded
from Hugging Face/FastEmbed. A normal install/index/search does not load this
model because reranking is default-off; it is downloaded only after an operator
sets `ENGRAM_RERANK_ENABLED=1` and requests reranking. For commercial use, do not
ship or operate the default reranker unless your use is compatible with
CC-BY-NC-4.0 or you have separate commercial rights. Choose a commercially usable
reranker model/backend and verify both its license and retrieval quality before
distribution.

The `sentence_transformers` backend remains available only to explicit
Python/offline callers via `get_reranker(backend="sentence_transformers")`; it is
not part of MCP/CLI search.

A built-in `eval` harness reports hit@1/5/10, MRR, HNSR@5/10, delta-rank, and
lexical-overlap buckets **per query category**. Fixture generation is offline:
`evals/generate_fixture.py` writes a hand-seeded paraphrase/distractor starter
set and marks the build-time LLM integration point; runtime eval stays LLM-free.
HNSR means the target is inside cutoff while the hard negative is outside;
delta-rank is hard-negative rank minus target rank; overlap buckets come from
query/target token Jaccard.

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
not suitable for commercial use unless your use is compatible with that license
or you have separate commercial rights.
