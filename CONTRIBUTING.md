# Contributing

Engram is currently a single-maintainer project. Focused bug reports, small fixes,
tests, and documentation improvements are welcome.

## Setup

```bash
git clone https://github.com/bokettodev/engram-mcp.git
cd engram-mcp
uv sync
```

For GPU indexing work:

```bash
uv sync --extra gpu
```

## Checks

Run the model-free test path before opening a PR.

PowerShell:

```powershell
$env:ENGRAM_SKIP_MODEL = "1"
uv run --no-sync pytest -q
uv run ruff check .
```

Bash:

```bash
ENGRAM_SKIP_MODEL=1 uv run --no-sync pytest -q
uv run ruff check .
```

Tests that require real model downloads or CUDA should stay opt-in and must not
run by default in CI.

## Pull Requests

- Keep PRs small and focused.
- Add or update tests for behavior changes.
- Do not commit local indexes, model caches, virtual environments, or generated
  cache files.
- Document user-visible changes in `README.md`.
- Call out any model-license implications, especially reranker changes.
