"""Maintenance run once at server startup. Opt-in, and off by default.

Two non-search-path reclaim actions live here:

- superseded ("stale") LanceDB generations for every indexed project (see
  ``gcreclaim``), guarded by the same non-blocking per-project lock the
  explicit ``engram gc`` path uses, so it never races a concurrent index
  writer.
- the global embedding cache, pruned to ``ENGRAM_CACHE_MAX_MB`` -- but only if
  an operator actually configured a budget. The default is unlimited, so by
  default this step changes nothing (see ``embeddings.cache.cache_max_bytes``
  for why unbounded is the safe default).

Why this is **opt-in** (``ENGRAM_GC_ON_START=1``) rather than automatic: a
"clean startup" is clean only for *this* process. Engram is deployed as one
stdio server process per MCP client window, so several servers run at once over
the same ``ENGRAM_HOME``. A generation this process considers stale may be the
one another live process is mid-search on -- and retaining exactly that
generation across a swap is what lets in-flight readers finish (see the atomic
generation swap in ``index_repository._full_rebuild``). Reclaiming it from under them
would defeat that guarantee. The per-project lock does not help: readers do not
take it.

The sanctioned reclaim path is therefore the explicit operator command,
``engram gc --prune`` (preview with ``--dry-run``), run when the operator knows
no server is serving. Hosts that genuinely run a single server may opt in with
``ENGRAM_GC_ON_START=1``. Both actions are skipped entirely under
``ENGRAM_READONLY=1``: a read-only server must never delete anything.
"""

from __future__ import annotations

import os

from engram_mcp import gcreclaim, paths
from engram_mcp.embeddings.cache import cache_max_bytes, global_cache_report


def startup_gc_enabled() -> bool:
    """Opt-in only. Unset means disabled -- see this module's docstring for why."""
    raw = os.environ.get("ENGRAM_GC_ON_START", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def run_startup_maintenance() -> dict | None:
    """Run once at process start. Returns the combined report, or ``None`` if skipped."""
    if paths.read_only_enabled() or not startup_gc_enabled():
        return None
    generations = gcreclaim.reclaim_all(dry_run=False)
    budget = cache_max_bytes()
    # `dry_run=True` here (budget unset) makes this a pure report -- no prune,
    # no VACUUM -- consistent with "unlimited" meaning "never auto-delete".
    cache = global_cache_report(dry_run=budget is None, max_bytes=budget)
    return {"stale_generations": generations, "embedding_cache": cache}
