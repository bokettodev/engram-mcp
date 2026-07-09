"""Offline eval fixture generator scaffold.

This script intentionally performs no network or LLM call by default. It emits a
small hand-seeded starter fixture with the schema used by ``engram eval``:

  category, query, expected_path(s), optional expected_symbol, distractor_paths.

TODO: wire an operator-provided build-time LLM in ``generate_with_llm`` to create
paraphrases and hard negatives. The generated JSON is committed or reviewed as a
static fixture; runtime evaluation must remain LLM-free.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def starter_cases() -> list[dict]:
    return json.loads(
        (Path(__file__).with_name("starter_paraphrase.json")).read_text(encoding="utf-8")
    )


def generate_with_llm(_root: Path) -> list[dict]:
    raise RuntimeError(
        "No build-time LLM provider is configured. Add one here to generate "
        "paraphrases and hard negatives, then review and save the static JSON."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".", help="project root")
    parser.add_argument("-o", "--output", default="evals/starter_paraphrase.json")
    parser.add_argument("--llm", action="store_true", help="use operator-wired build-time LLM")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    cases = generate_with_llm(root) if args.llm else starter_cases()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
