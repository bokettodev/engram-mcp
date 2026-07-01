# Engram

> Local, private **semantic code search** for AI coding agents — a self-hosted MCP server.

Point Engram at a codebase. It chunks the code by symbol (tree-sitter), embeds the
chunks **locally**, and stores them in an embedded vector database. Then an AI agent
(Claude Code, Cursor, Codex, …) — or you, from the CLI — can ask
*"where is X implemented?"* in plain language and get back the most relevant code,
ranked, with file + line ranges.

**Everything runs on your machine. No code ever leaves it** — no cloud embeddings,
no API keys.

Engram speaks the [Model Context Protocol](https://modelcontextprotocol.io) (MCP) — the
open standard AI agents use to call tools — so an agent calls a fast `search_code`
tool instead of grepping blindly.

*(The name: an **engram** is a memory trace — Engram "remembers" your codebase as
embeddings and recalls it by meaning.)*

## Requirements

- [**uv**](https://docs.astral.sh/uv/) — manages Python 3.12 + dependencies (the only thing you install globally).
- *Optional:* an NVIDIA GPU, to use the stronger code-specialized embedders. The default embedder runs fine on CPU.

## Quickstart

```bash
git clone <repo-url> engram-mcp
cd engram-mcp
uv sync                                   # installs Python 3.12 + dependencies

uv run engram index  /path/to/your/repo   # build the index (first run downloads a small model)
uv run engram search /path/to/your/repo "where are http requests retried?"
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
uv run engram index    <path> [--rebuild] [--profile P]   # build / update the index
uv run engram search   <path> "<query>" -k 8 [--mode auto|vector|hybrid] [--rerank] [--lang py]
uv run engram find-def <path> <symbol>                    # exact symbol definition lookup
uv run engram eval     <path> evals/self.json [--mode M]  # measure retrieval quality
uv run engram remove   <path>                             # delete a project's index
uv run engram chunk    <path> [--show N]                  # walk + chunk only (no embedding)
```

Indexing is **incremental by default** — only changed/added/deleted files are
re-processed (detected by content hash + mtime), and unchanged chunks are served
from a global embedding cache, so re-indexing is near-free. `--rebuild` forces a
full atomic rebuild (a crash mid-rebuild leaves the previous index searchable).

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

**Picking the model for the server.** The server defaults to `local_fast`
(bge-small). To make it use another profile for *new* indexing, set
`ENGRAM_DEFAULT_INDEX_PROFILE` in the server's env. `ENGRAM_PROFILE` still works
as a backwards-compatible alias. Neither affects search over an existing index:
search always uses the model recorded in the project's manifest.

```bash
claude mcp add engram -e ENGRAM_DEFAULT_INDEX_PROFILE=local_qwen -- \
  uv --directory /ABSOLUTE/PATH/TO/engram-mcp run --extra gpu engram-mcp
```

- A `local_qwen*` profile needs the torch models present, so launch the server
  with `run --extra gpu` (or pre-run `uv sync --extra gpu` once). Without it,
  `uv run` syncs only the base deps and the Qwen profiles can't load.
- **Windows footgun:** if `uv run` reports `failed to remove … engram-mcp.exe …
  used by another process`, a previous server instance is holding the script
  while `uv` tries to re-sync. Launch with `run --no-sync` (use the already-set-up
  venv, skip the sync) to avoid the lock:
  `uv --directory … run --no-sync engram-mcp`.

**Read-only mode.** Set the env var `ENGRAM_READONLY=1` on the server and only the
read tools (`search_code`, `get_chunk`, `find_definition`, `model_status`,
`index_status`, `list_indexed_projects`, `server_info`) are registered. The
mutating tools (`index_project`, `reindex_file`, `remove_project`) are withheld,
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
tool call never blocks for minutes):

| Tool | Purpose |
|---|---|
| `index_project(project_path, full_rebuild=False, profile=None)` | start a background index; returns `job_id` |
| `index_status(job_id)` | current-process progress snapshot (stage, counts, ETA) |
| `search_code(project_path, query, k=8, language=None, mode="auto", rerank=False, content="preview", max_chars_per_result=800, min_relevance=None)` | compact ranked hits over static indexed source |
| `get_chunk(project_path, chunk_id, max_chars=None)` | fetch full content for one search hit |
| `find_definition(project_path, symbol)` | exact symbol definition lookup, with suggestions on miss (no embedding) |
| `model_status(project_path=None)` | reports whether the project's recorded query model is loaded/loading/not_loaded in this process |
| `reindex_file(project_path, rel_path)` | incrementally re-index/drop one file |
| `remove_project(project_path)` | delete a project's index |
| `list_indexed_projects()` | on-disk index inventory, `data_home`, and broken manifest/table `errors[]` |
| `server_info()` | data-home, read-only, and default index-profile diagnostics |

`search_code` is a decision tool first: by default each hit contains `chunk_id`,
`rel_path`/`span`, `symbol`, `symbol_kind`, `chunk_role`, `preview`, `raw_score`,
`score_normalized`, `relevance` (`high|medium|low|uncertain`), `matched`,
`match_reason`, `stale`, and `truncated`. It also returns `mode_requested`,
`mode_used`, `warnings[]`, `rerank_applied`, `source_type:
"static_indexed_source"`, and a `dirty` freshness summary. Use
`content="none"` to get metadata only, `content="full"` for bounded inline text,
or `get_chunk` for exact full content by `chunk_id`. `k` is bounded to `1..50`.

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
reused across projects — not under `ENGRAM_HOME`. They are sizeable (bge-small
~130 MB; Qwen3-4B ~8 GB), so point `HF_HOME` at a roomy disk if your home
partition is small.

## Embedder profiles

FastEmbed (ONNX, light, **no torch, no GPU needed**) — the default:

| Profile | Model | Dim | Lang | Device |
|---|---|---|---|---|
| `local_fast` (default) | bge-small-en-v1.5 | 384 | English | CPU |
| `local_quality` | bge-large-en-v1.5 | 1024 | English | CUDA if available |
| `local_granite` | granite-embedding-97m-multilingual-r2 | 384 | **multilingual + code** | CPU |
| `local_granite_quality` | granite-embedding-311m-multilingual-r2 | 768 | **multilingual + code** | CPU |

[Granite R2](https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2)
(IBM, Apache-2.0) covers 100+ languages **including Russian** plus code, runs on
CPU/ONNX with **~0 VRAM**, and has a 32K context — the light everyday pick when
several editor/agent windows each spawn their own server (see *GPU memory* below).
The model is registered with FastEmbed on first use and downloaded once.

Quality models (**GPU recommended**) — behind the optional `gpu` extra (pulls
torch, ~2 GB). [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
tops MTEB on **both text and code**, so one family covers prose and source:

```bash
uv sync --extra gpu
uv run engram index <path> --profile local_qwen
```

On Windows/Linux this installs a **CUDA build of torch automatically** (the
models then run on your NVIDIA GPU — only an NVIDIA driver is needed, no separate
CUDA toolkit). On CPU they work but are far slower, so a GPU is strongly
recommended. The default `uv sync` (no extra) is unaffected — it stays
torch-free and CPU-only.

| Profile | Model | Dim | Model license |
|---|---|---|---|
| `local_qwen_small` | Qwen3-Embedding-0.6B | 1024 | Apache-2.0 |
| `local_qwen` | Qwen3-Embedding-4B | 1024 (MRL) | Apache-2.0 |

The 4B's native 2560 dims are truncated to 1024 via Matryoshka (MRL) for index
parity at <few % quality loss. The embedder id (incl. dim) is recorded in the
index manifest, so search auto-selects the matching model and switching model
re-indexes cleanly. Set the index-time default with `ENGRAM_DEFAULT_INDEX_PROFILE`
(or legacy `ENGRAM_PROFILE`), e.g. `ENGRAM_DEFAULT_INDEX_PROFILE=local_qwen`.
`bge-small` stays the default because it needs no GPU and no torch.

**Every model here is Apache-2.0 / MIT** — no non-commercial restrictions.

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

### GPU memory (torch profiles)

The `local_qwen*` profiles load the model into VRAM; the FastEmbed `local_*`
profiles use ONNX and effectively no VRAM. A few things worth knowing for a
GPU-shared, always-on server:

- **One model copy per process.** Each MCP client (editor/agent window) spawns
  its own stdio server process, and each loads its own copy of the model into
  VRAM. N open clients on a Qwen profile ≈ N model copies. The no-VRAM
  `local_fast` (bge, default) sidesteps this — keep it as the always-on default
  and request a `local_qwen*` profile per index when you want the quality.
- **A Qwen index pulls Qwen into search too:** search uses the model recorded in
  the project's manifest, so searching a Qwen-indexed project loads Qwen even if
  your default profile is bge.
- **`ENGRAM_ST_BATCH_SIZE`** (default 16) caps the encode batch — the real lever
  on activation VRAM during indexing. Lower it (e.g. 8) on a tight/shared GPU;
  raise it for throughput on a dedicated one.
- After a bulk index Engram returns the activation high-water to the GPU, but the
  **model stays resident** for warm search latency — it's freed when the server
  process exits (restart the client/server to reclaim it).

## Retrieval quality

Search modes (`--mode`, default `auto`):

- **auto** — routes identifier/literal queries (`EmbeddingCache`, `MAX_FILE_BYTES`,
  `"file not found"`) to hybrid, and plain natural-language queries to vector.
- **vector** — dense embedding similarity.
- **hybrid** — vector + full-text (BM25), fused with reciprocal-rank fusion + symbol/path boosts.
- **`--rerank`** — a local cross-encoder reranks the top candidates (needs `--extra gpu`).

A built-in `eval` harness reports hit@1/5/10 + MRR **per query category** — that's how
the defaults were chosen. On the bundled 50-query set, hybrid beats vector overall
(hit@1 80% vs 72%), and swapping the CPU `bge-small` for a GPU `local_qwen*`
(Qwen3-Embedding) model lifts hit@1 further. Run `eval` on your own repo to tune
for your codebase.

## Performance

Cold indexing is bound by embedding throughput: ~21 chunks/s on CPU with the default
model (a few minutes for a mid-size repo; ~15–35 min for 1–2M LOC). Re-indexing is
near-free thanks to the content-hash cache. The Qwen3 quality models are far
stronger but much slower on CPU — use a GPU for them.

## Stack

| Concern | Choice |
|---|---|
| Runtime | Python 3.12 via [`uv`](https://docs.astral.sh/uv/) |
| Embedder | [FastEmbed](https://github.com/qdrant/fastembed) (`bge`, ONNX) · optional [sentence-transformers](https://www.sbert.net/) (Qwen3-Embedding) |
| Vector store | [LanceDB](https://lancedb.com/) (embedded, on-disk, vector + full-text) |
| Chunker | tree-sitter — 11 languages (py/js/ts/tsx/go/rust/java/c/cpp/ruby/c#) · markdown by heading section · plain text by paragraph · line-window fallback |
| Server | MCP Python SDK (FastMCP, stdio) |

## Status

- ✅ Walk + nested-`.gitignore` ignore + tree-sitter chunking (11 langs, contextual headers)
- ✅ Structure-aware prose: markdown split by heading section (breadcrumb → chunk symbol) + plain text packed by paragraph
- ✅ Local embedding + LanceDB store + content-hash cache
- ✅ Atomic full rebuild + incremental re-index + per-project lock
- ✅ Embedder profiles: no-torch FastEmbed (bge) + Qwen3-Embedding quality models via `--extra gpu`
- ✅ TLS downloads work behind corporate MITM proxies (OS trust store by default)
- ✅ Bounded + reclaimed GPU memory (`ENGRAM_ST_BATCH_SIZE`, cache release after bulk index)
- ✅ Search modes auto / vector / hybrid + optional cross-encoder rerank
- ✅ Per-category eval harness · `find_definition` (with miss suggestions)
- ✅ Async MCP server: index / status / search / get_chunk / find_definition / model_status / reindex_file / remove_project / list / server_info
- ✅ Agent-grade tool contract: hermetic reads (model from manifest), structured `E_*` errors, compact-first results + `get_chunk`, normalized score + relevance buckets, search-time freshness (`stale`), honest `mode_used`/`warnings`, `chunk_role`
- ⏳ GPU-tuned serving · idle model unload · Merkle incremental · file-watching · durable job registry

## Development

```bash
uv run --no-sync pytest -q          # +2 tests skip unless `--extra gpu` is installed
```

## License

[MIT](LICENSE). Engram itself and every embedding/reranker model it can download
are MIT or Apache-2.0 — no non-commercial restrictions.
