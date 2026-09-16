"""Create provenance-preserving instruction candidates from a Nepali corpus.

Candidates are deliberately marked ``reviewed: false``. A human must verify
the answer, task, language direction, safety and license before SFT ingestion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

TASKS = ("qa", "summarization", "rewriting", "generation")


def build(corpus: Path, output: Path, limit: int, seed: int):
    lines = [
        line.strip()
        for line in corpus.read_text(encoding="utf-8").splitlines()
        if len(line.split()) >= 8
    ]
    random.Random(seed).shuffle(lines)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for index, text in enumerate(lines[:limit]):
            task = TASKS[index % len(TASKS)]
            if task == "qa":
                instruction, answer = (
                    "यस अनुच्छेदको मुख्य विषय के हो?",
                    text.split("।")[0] + "।",
                )
            elif task == "summarization":
                instruction, answer = "यस पाठको संक्षिप्त सारांश लेख्नुहोस्।", text
            elif task == "rewriting":
                instruction, answer = "यस वाक्यलाई सरल नेपालीमा पुनर्लेखन गर्नुहोस्।", text
            else:
                instruction, answer = "यस विषयमा एउटा अनुच्छेद लेख्नुहोस्।", text
            record = dict(
                instruction=instruction,
                input=text,
                output=answer,
                task=task,
                source=str(corpus),
                source_hash=hashlib.sha256(text.encode()).hexdigest(),
                license="VERIFY_SOURCE_LICENSE",
                reviewer="",
                reviewed=False,
            )
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {min(len(lines), limit)} unreviewed candidates to {output}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True)
    p.add_argument("--output", default="data/instructions/candidates.jsonl")
    p.add_argument("--limit", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if args.limit < 1:
        p.error("limit must be positive")
    build(Path(args.corpus), Path(args.output), args.limit, args.seed)


if __name__ == "__main__":
    main()
