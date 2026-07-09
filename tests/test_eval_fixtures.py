from __future__ import annotations

from pathlib import Path

from engram_mcp import evaluate


ROOT = Path(__file__).resolve().parents[1]


def _case_paths(case: dict) -> list[str]:
    paths = []
    if case.get("expected_path"):
        paths.append(case["expected_path"])
    paths.extend(case.get("expected_paths") or [])
    paths.extend(case.get("distractor_paths") or [])
    return paths


def test_self_fixture_has_real_paraphrase_distractor_split():
    cases = evaluate.load_cases(ROOT / "evals" / "self.json")
    paraphrases = [c for c in cases if c.get("category") == "paraphrase"]

    assert 25 <= len(paraphrases) <= 40
    for case in cases:
        for rel in _case_paths(case):
            assert (ROOT / rel).is_file(), rel

    for case in paraphrases:
        expected = case.get("expected_path") or case.get("expected_paths", [""])[0]
        distractors = case.get("distractor_paths") or []
        assert 1 <= len(distractors) <= 3
        assert expected not in distractors
        overlap = evaluate._jaccard(
            case["query"],
            evaluate._read_expected_text(ROOT, {expected}),
        )
        assert overlap < 0.03, (case["query"], expected, overlap)


def test_starter_paraphrase_fixture_is_runnable_schema():
    cases = evaluate.load_cases(ROOT / "evals" / "starter_paraphrase.json")
    assert len(cases) == 10
    assert all(c.get("category") == "paraphrase" for c in cases)
    assert all(c.get("distractor_paths") for c in cases)
