"""Static defaults for walking, ignoring, and chunking.

Tunables live here so the walker/chunker stay declarative. Token counts are
rough estimates (chars/4) used only for chunk *sizing*; the real embedder
tokenizer is applied later in the embed phase.
"""

from __future__ import annotations

# Directory names pruned during the walk (matched by name at any depth).
DEFAULT_EXCLUDE_DIRS = frozenset(
    {
        ".git", ".hg", ".svn", ".bzr",
        "node_modules", "bower_components", "jspm_packages",
        "dist", "build", "out", "target", "bin", "obj", "_build",
        ".next", ".nuxt", ".svelte-kit", ".angular", ".parcel-cache",
        ".venv", "venv", "env", "virtualenv",
        "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        ".tox", ".gradle", ".m2",
        ".idea", ".vscode", ".vs",
        ".cache", "coverage", ".nyc_output", "htmlcov",
        "vendor", "third_party", "Pods", ".terraform", ".serverless",
    }
)

# Filename globs always skipped (gitwildmatch semantics via pathspec).
DEFAULT_EXCLUDE_GLOBS = (
    "*.min.js", "*.min.css", "*.min.mjs",
    "*.map", "*.js.map", "*.css.map",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "npm-shrinkwrap.json",
    "poetry.lock", "Pipfile.lock", "Cargo.lock", "composer.lock",
    "go.sum", "Gemfile.lock",
    "*.snap",
)

# Extensions treated as binary / non-source and skipped outright.
BINARY_EXTS = frozenset(
    {
        # images
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tiff", ".svg",
        # fonts
        ".woff", ".woff2", ".ttf", ".otf", ".eot",
        # media
        ".mp3", ".mp4", ".wav", ".ogg", ".webm", ".avi", ".mov", ".flac",
        # archives
        ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
        # compiled / binary
        ".pyc", ".pyo", ".so", ".dll", ".dylib", ".exe", ".bin", ".o", ".a",
        ".class", ".jar", ".wasm",
        # docs / data blobs
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        ".db", ".sqlite", ".sqlite3",
        # vcs / pack
        ".pack", ".idx", ".lock",
    }
)

MAX_FILE_BYTES = 2_000_000  # includes large API reference pages; still rejects data blobs
CHARS_PER_TOKEN = 4  # rough token estimate for chunk sizing only
CHUNK_MAX_TOKENS = 480  # hard cap per chunk (small enough for any embedder's context)
CHUNK_OVERLAP_TOKENS = 60  # overlap for the line-window fallback
# Prose (markdown/plain-text) chunk cap. Kept at the model context for now;
# split as a named knob so prose can be tuned independently of code chunks.
PROSE_CHUNK_MAX_TOKENS = 480

# Bumped whenever chunking OR embedded-text format changes, so the cache
# invalidates and incompatible indexes rebuild. v2: contextual chunk headers.
# v3: brace-style header preservation + C/C++ declarator symbol names.
# v3 retained lean path/symbol/language headers.
# v4: structure-aware prose chunking — markdown split by heading sections (the
# heading breadcrumb becomes the chunk symbol) + plain text packed by paragraph.
# chunk_role is stored for new rows and derived for old rows. It does not alter
# chunk boundaries or embedded text, so it intentionally does not bump the
# embedding-cache key.
# v5: module-level constant/variable definitions (Python assignment; JS/TS
# top-level const; Go const/var; Rust const/static) become their own symbol
# chunk instead of being folded into the generic "module" bundle chunk --
# changes chunk boundaries for files with such definitions, so old chunks
# must not be served against v5-produced ones.
# v6: reStructuredText split by adornment-based heading sections with hierarchy;
# the source-file cap also rises to 2 MB for large generated API reference pages.
CHUNKER_VERSION = "6"

# Bumped whenever the chunk_id derivation algorithm changes, so a strict
# manifest load rejects an index built with a different scheme instead of
# silently serving ids that won't round-trip through get_chunk. v1: id is a
# pure function of (rel_path, per-file chunk ordinal, chunk content hash) -
# stable whether the chunk was produced by a full rebuild or an incremental
# reindex of just its file, unlike the earlier global-batch ordinal.
CHUNK_ID_SCHEME = "per_file_ordinal_v1"

# Default local embedder (FastEmbed / ONNX, Granite R2 97m multilingual, 384-dim).
DEFAULT_EMBED_MODEL = "ibm-granite/granite-embedding-97m-multilingual-r2"
DEFAULT_EMBED_DIM = 384
EMBED_BATCH_SIZE = 256

# --- Pinned upstream model revisions ----------------------------------------
# Model *identity* is mutable at Hugging Face: an org can push new weights, a
# new tokenizer, or a re-exported ONNX artifact to the SAME repo name at any
# time. A bare repo name is therefore not a safe cache/manifest key -- without
# a pin, engram would silently start mixing old cached passage vectors and old
# indexes with vectors produced by different weights the moment upstream
# force-pushes a change, with no error and no signal to the user.
#
# Each value below is the exact Hugging Face commit SHA (not a mutable branch
# name like "main") that engram loads. It is threaded into
# `embeddings/factory.py::CANONICAL_EMBEDDER_ID` (-> the manifest's
# `embedder_id` field and the global embedding-cache key, see
# `indexing/hash.py::embedding_input_hash`) and passed to the loader
# (FastEmbed via `embeddings/hf_pin.py`'s pinned-snapshot resolution +
# `specific_model_path`; sentence-transformers natively via `revision=` for
# GPU indexing, see `embeddings/sentence_transformers_provider.py`).
#
# Changing any of these values is a breaking change for every existing index
# and every row in the global embedding cache -- there is no migration path.
# A changed revision simply stops matching `CANONICAL_EMBEDDER_ID`, so an old
# index/cache entry fails loud (forces a rebuild) instead of silently mixing
# vectors from two different model revisions. Obtained via
# `huggingface_hub.HfApi().model_info(repo_id).sha` on 2026-07-10.
EMBED_MODEL_REVISION = "835ad14087e140460703cf0fae09f97d469d65c2"
# ^ ibm-granite/granite-embedding-97m-multilingual-r2

RERANKER_ONNX_MODEL_REVISION = "9cfeff2df7d40d1b78e75e5e9cebec92a99813c9"
# ^ jinaai/jina-reranker-v2-base-multilingual (the only reranker backend)
