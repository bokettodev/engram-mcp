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
- **Recommended:** an NVIDIA GPU + the `gpu` extra — indexing prefers CUDA by
  default and is ~150× faster there. CPU indexing works but is a **slow
  fallback**. Search always stays on CPU (no torch, ~0 VRAM).

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
uv run engram index    <path> [--rebuild] [--gpu|--cpu]   # build / update (defaults to GPU if available, else CPU)
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

- The GPU path needs the torch/sentence-transformers packages (`--extra gpu` /
  `uv sync --extra gpu`). A GPU index job runs in a short-lived subprocess, so
  the long-lived server never loads torch and stays ~0 VRAM even after GPU jobs.
  Without the extra (or without a GPU) the default `auto` transparently uses the
  CPU fallback; `index_device="cuda"` fails explicitly instead of falling back.
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
| `index_project(project_path, full_rebuild=False, index_device=None)` | start a background index; returns `job_id`. `index_device`: `auto` (default, prefers GPU) / `cuda` (require GPU) / `cpu` (slow fallback) |
| `index_status(job_id)` | current-process progress snapshot (stage, counts, ETA) |
| `search_code(project_path, query, k=8, language=None, mode="auto", rerank=False, content="preview", max_chars_per_result=800, min_relevance=None)` | compact ranked hits over static indexed source |
| `get_chunk(project_path, chunk_id, max_chars=None)` | fetch full content for one search hit |
| `find_definition(project_path, symbol)` | exact symbol definition lookup, with suggestions on miss (no embedding) |
| `model_status(project_path=None)` | reports whether the project's recorded query model is loaded/loading/not_loaded in this process |
| `reindex_file(project_path, rel_path)` | incrementally re-index/drop one file |
| `remove_project(project_path)` | delete a project's index |
| `list_indexed_projects()` | on-disk index inventory, `data_home`, and broken manifest/table `errors[]` |
| `server_info()` | data-home, read-only, canonical embedder, and index-device diagnostics |

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
- VRAM is held only for the duration of a GPU index job, by the child process,
  never by the always-on search server.

## Retrieval quality

Search modes (`--mode`, default `auto`):

- **auto** — routes identifier/literal queries (`EmbeddingCache`, `MAX_FILE_BYTES`,
  `"file not found"`) to hybrid, and plain natural-language queries to vector.
- **vector** — dense embedding similarity.
- **hybrid** — vector + full-text (BM25), fused with reciprocal-rank fusion + symbol/path boosts.
- **`--rerank`** — a local cross-encoder reranks the top candidates (needs `--extra gpu`).

A built-in `eval` harness reports hit@1/5/10 + MRR **per query category**. Run
`eval` on your own repo to tune search mode and reranking for your codebase.

## Performance

Cold indexing is bound by embedding throughput. By default it runs on the GPU
(sentence-transformers/CUDA) at ~hundreds–1000 chunks/s; the CPU fallback is
torch-free but far slower (single-digit to low-tens chunks/s), so it's a
last-resort for machines without a GPU. Re-indexing is near-free thanks to the
content-hash cache — only changed chunks are re-embedded. Search embeds one
query on CPU and is always cheap.

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

[MIT](LICENSE). Engram itself and every embedding/reranker model it can download
are MIT or Apache-2.0 — no non-commercial restrictions.
