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

## Enforced project capability boundary (read tools accept any project_path)
Every read tool (`search_code`, `get_chunk`, `find_definition`, `project_map`, `doctor_project`,
`model_status`, `index_status`) accepts an arbitrary `project_path`, and
`list_indexed_projects` discloses every indexed project root on the machine. The server runs with the
operator's full OS privileges, so a prompt-injected agent — even one whose host confined it to a single
workspace — can call these tools to enumerate the user's other indexed repositories, search them, and
hydrate chunks from them. In effect, a local MCP server can silently defeat the host agent's filesystem
sandbox, turning prompt injection into cross-repo private-code disclosure. `ENGRAM_READONLY=1` only
removes the mutating tools (`index_project`/`reindex_file`/`remove_project`); it is not a read boundary
and does nothing to stop this.
**Reason deferred:** today engram is single-user on one machine, indexing the operator's own repos — the
blast radius is the operator's own filesystem, which they already have full access to outside the server.
Worth closing before recommending engram for multi-agent setups where a host is relying on its own
sandbox to confine an untrusted or prompt-injectable agent to one project.
**Candidate fix:** an enforced project capability boundary — either bind one server instance to a single
configured root, or add an `ENGRAM_ALLOWED_ROOTS` allowlist. Canonicalize paths *after* ref/worktree
resolution (a worktree/ref can otherwise resolve outside an allowed root), filter
`list_indexed_projects`'s inventory to allowed roots, and default to deny cross-root access, at least
whenever `ENGRAM_READONLY=1` is set.
