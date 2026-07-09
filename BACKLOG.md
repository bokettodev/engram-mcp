# Backlog

Deferred work with an explicit rationale. Nothing here is deferred by "priority" or "difficulty" alone —
each entry has a concrete reason it is not being built yet. Items are pulled from here when the maintainer
says so.

## GPU queue observability (ETA / position)
When several engram processes (one per host) plus CLI indexing compete for the single GPU admission lock,
the wait is opaque (`waiting_for_gpu` with no ETA). Showing "you are Nth in line, ~Ns" needs:
- a shared ledger file (`{pid, job_id, requested_at, heartbeat}`) with stale-entry cleanup, and
- ETA from the active holder's start time plus rolling historical job durations.
**Reason deferred:** `filelock` exposes neither queue position nor holder timing; this is best built together
with the fair multi-slot GPU semaphore below (both need the same queue abstraction).

## Fair multi-slot GPU semaphore
Today the GPU lock is a single slot (1 job at a time, machine-wide). A fair N-slot semaphore would let a
capable GPU run a few small index jobs concurrently and would give the queue structure the ETA item needs.
**Reason deferred:** needs a real cross-process queue/semaphore abstraction; pairs with the ETA item.

## Opt-in keep-alive GPU worker (ENGRAM_GPU_WORKER)
A long-lived GPU worker process that keeps the CUDA model warm for back-to-back index jobs, instead of a
fresh short-lived subprocess each time.
**Reason deferred:** conflicts with the ~0-VRAM-at-rest default; only worth it for heavy continuous
indexing, must stay strictly opt-in.

## CLI `doctor` / `map` subcommands
`doctor_project` and `project_map` are MCP-only; an operator debugging from a terminal cannot run them.
`engram doctor <path>` (health of a CLI-built index) is the highest-value one.
**Reason deferred:** thin argparse wrappers over existing pipeline logic; low risk, not yet requested to build.

## CLI `reindex-all` / file-watch freshness
A scheduled/triggered "reindex the stale projects" command (or a watch mode) so an operator can keep many
indexes fresh without hand-running `engram index` per project.
**Reason deferred:** freshness belongs to an external scheduler, not the always-on server; not yet requested.
