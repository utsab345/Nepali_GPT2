"""CLI: semi-automatically build a Nepali cloze benchmark from text."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from eval.cloze import build_cloze_examples, write_jsonl  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a cloze benchmark from a plain-text corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--corpus", default="data/nepali_corpus.txt", help="raw Nepali text")
    p.add_argument(
        "--out",
        default="eval/data/ne_cloze_auto.jsonl",
        help="output JSON Lines benchmark (not committed)",
    )
    p.add_argument("--max-examples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    corpus = Path(args.corpus)
    if not corpus.exists():
        print(
            f"Corpus not found: {corpus} (run `python -m nepali_gpt2 data-prep` first)."
        )
        return 1

    examples = build_cloze_examples(
        corpus.read_text(encoding="utf-8"),
        rng=random.Random(args.seed),
        max_examples=args.max_examples,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(examples, out)
    print(f"Wrote {len(examples)} cloze examples → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
