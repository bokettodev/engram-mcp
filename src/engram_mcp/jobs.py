"""In-memory async index-job registry + progress snapshots.

v1 keeps jobs in memory (the index itself is durable on disk via LanceDB);
persisting job rows across server restarts is a later phase.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass


@dataclass
class JobState:
    job_id: str
    project_path: str
    status: str = "queued"  # queued | running | done | error
    stage: str = ""
    files: int = 0
    chunks: int = 0
    embedded: int = 0
    reused: int = 0
    done_units: int = 0
    total_units: int = 0
    error: str | None = None
    created_at: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0


_MAX_HISTORY = 200


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()

    def create(self, project_path: str) -> JobState:
        with self._lock:
            self._prune_locked()
            job = JobState(
                job_id=uuid.uuid4().hex[:12],
                project_path=project_path,
                created_at=time.time(),
            )
            self._jobs[job.job_id] = job
            return job

    def _prune_locked(self) -> None:
        """Drop the oldest finished jobs so the registry stays bounded."""
        if len(self._jobs) < _MAX_HISTORY:
            return
        finished = sorted(
            (j for j in self._jobs.values() if j.status in ("done", "error")),
            key=lambda j: j.finished_at,
        )
        for j in finished[: len(self._jobs) - _MAX_HISTORY + 1]:
            self._jobs.pop(j.job_id, None)

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for k, v in fields.items():
                setattr(job, k, v)

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[JobState]:
        with self._lock:
            return list(self._jobs.values())


def snapshot(job: JobState) -> dict:
    now = time.time()
    elapsed = ((job.finished_at or now) - job.started_at) if job.started_at else 0.0
    eta = None
    if job.status == "running" and job.done_units and job.total_units:
        per_unit = elapsed / job.done_units
        eta = max(0.0, per_unit * (job.total_units - job.done_units))
    return {
        "job_id": job.job_id,
        "project_path": job.project_path,
        "status": job.status,
        "stage": job.stage,
        "files": job.files,
        "chunks": job.chunks,
        "embedded": job.embedded,
        "reused": job.reused,
        "progress": {"done": job.done_units, "total": job.total_units},
        "elapsed_sec": round(elapsed, 2),
        "eta_sec": round(eta, 1) if eta is not None else None,
        "error": job.error,
    }
