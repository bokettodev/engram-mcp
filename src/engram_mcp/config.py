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

MAX_FILE_BYTES = 1_000_000  # skip files larger than ~1 MB by default
CHARS_PER_TOKEN = 4  # rough token estimate for chunk sizing only
CHUNK_MAX_TOKENS = 480  # hard cap per chunk (bge-small context is 512)
CHUNK_OVERLAP_TOKENS = 60  # overlap for the line-window fallback
# Prose (markdown/plain-text) chunk cap. Kept at the model context for now;
# split as a named knob so prose can be tuned independently of code chunks.
PROSE_CHUNK_MAX_TOKENS = 480

# Bumped whenever chunking OR embedded-text format changes, so the cache
# invalidates and incompatible indexes rebuild. v2: contextual chunk headers.
# v3: brace-style header preservation + C/C++ declarator symbol names.
# (An earlier v4 added file-level imports to the header but measured slightly
# worse on the eval set, so it was reverted; the lean path/symbol/language
# header stands.)
# v4: structure-aware prose chunking — markdown split by heading sections (the
# heading breadcrumb becomes the chunk symbol) + plain text packed by paragraph.
CHUNKER_VERSION = "4"

# Default local embedder (FastEmbed / ONNX, bge-small-en-v1.5, 384-dim).
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBED_DIM = 384
EMBED_BATCH_SIZE = 256
