from __future__ import annotations

import re
from pathlib import Path

import yaml

_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def _workflow_text() -> str:
    return _WORKFLOW.read_text(encoding="utf-8")


def _workflow_yaml() -> dict:
    return yaml.safe_load(_workflow_text())


def _steps(job: dict) -> list[dict]:
    return job.get("steps") or []


def test_setup_uv_action_is_sha_pinned() -> None:
    text = _workflow_text()

    assert "astral-sh/setup-uv@11f9893b081a58869d3b5fccaea48c9e9e46f990 # v8.3.2" in text
    assert "astral-sh/setup-uv@v8.3.2" not in text


def test_fast_lane_still_asserts_torch_free() -> None:
    """The pre-existing model-free `lint-test` lane must keep its torch-free
    guarantee -- the new real-model lane below is additive, not a replacement."""
    data = _workflow_yaml()
    job = data["jobs"]["lint-test"]
    assert job["env"]["ENGRAM_SKIP_MODEL"] == "1"

    step_runs = " ".join(s.get("run", "") for s in _steps(job))
    assert "'torch' not in sys.modules" in step_runs
    assert "import engram_mcp.pipeline" in step_runs
    assert "import engram_mcp.server" in step_runs


def test_real_model_lane_exists_and_does_not_skip_the_model() -> None:
    data = _workflow_yaml()
    jobs = data["jobs"]
    assert "real-model" in jobs, "expected a real-model CI lane alongside lint-test"
    job = jobs["real-model"]

    # The whole point of this lane: it must NOT set ENGRAM_SKIP_MODEL anywhere
    # a job/step could set it, so the real embedder/reranker actually load.
    assert "ENGRAM_SKIP_MODEL" not in (job.get("env") or {})
    for step in _steps(job):
        assert "ENGRAM_SKIP_MODEL" not in (step.get("env") or {})


def test_real_model_lane_caches_hugging_face_models_keyed_on_config() -> None:
    data = _workflow_yaml()
    steps = _steps(data["jobs"]["real-model"])
    cache_steps = [s for s in steps if str(s.get("uses", "")).startswith("actions/cache@")]
    assert cache_steps, "expected an actions/cache step for the Hugging Face model cache"
    cache_step = cache_steps[0]

    assert "huggingface" in cache_step["with"]["path"]
    # Keyed on config.py (which carries EMBED_MODEL_REVISION /
    # RERANKER_ONNX_MODEL_REVISION) so a pinned-revision bump busts the cache.
    assert "config.py" in cache_step["with"]["key"]


def test_real_model_lane_runs_tests_onnx_smoke_and_baseline_gate() -> None:
    data = _workflow_yaml()
    steps = _steps(data["jobs"]["real-model"])
    step_runs = " ".join(s.get("run", "") for s in steps)

    assert "pytest" in step_runs  # runs the model-dependent test suite for real
    assert "onnx_embedder_smoke" in step_runs
    assert "onnx_reranker_smoke" in step_runs
    assert re.search(r"engram\s+eval\b.*--baseline\s+evals/baseline\.json", step_runs), (
        "expected the real-model lane to gate on the checked-in eval baseline"
    )


def test_new_actions_in_real_model_lane_are_sha_pinned() -> None:
    """Matches the existing style: every `uses:` in ci.yml is pinned to a full
    commit SHA with a human-readable tag as a trailing comment, never a
    floating tag/branch."""
    data = _workflow_yaml()
    steps = _steps(data["jobs"]["real-model"])
    uses = [s["uses"] for s in steps if "uses" in s]
    assert uses, "expected at least one `uses:` step in the real-model lane"

    text = _workflow_text()
    for u in uses:
        owner_repo = u.split("@", 1)[0]
        assert re.search(
            rf"uses:\s*{re.escape(owner_repo)}@[0-9a-f]{{40}}\s+#\s*\S+", text
        ), f"{owner_repo} is not SHA-pinned with a trailing version comment: {u!r}"
