"""Validate, deduplicate and split reviewed instruction JSONL with source provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

TASKS = {
    "qa",
    "summarization",
    "rewriting",
    "translation_ne_en",
    "translation_en_ne",
    "generation",
}


def prepare(sources, outdir, minimum=10000, maximum=50000, seed=42):
    records, seen = [], set()
    for source in sources:
        for number, line in enumerate(
            Path(source).read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("instruction", "output", "source", "license", "reviewer"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"{source}:{number}: missing {key}")
            if row.get("task") not in TASKS or row.get("reviewed") is not True:
                raise ValueError(f"{source}:{number}: invalid task or not reviewed")
            if not isinstance(row.get("input", ""), str):
                raise ValueError(f"{source}:{number}: input must be text")
            # Group identical prompts to prevent train/validation leakage, even with different answers.
            identity = row["instruction"].strip() + "\n" + row.get("input", "").strip()
            digest = hashlib.sha256(identity.encode()).hexdigest()
            if digest not in seen:
                seen.add(digest)
                records.append(dict(row, id=digest))
    random.Random(seed).shuffle(records)
    records = records[:maximum]
    if not 2 <= minimum <= len(records) <= maximum:
        raise ValueError(
            f"Need {minimum}–{maximum} unique reviewed records, got {len(records)}"
        )
    missing = TASKS - {r["task"] for r in records}
    if missing:
        raise ValueError(f"Missing tasks: {sorted(missing)}")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    split = max(1, int(len(records) * 0.05))
    for name, rows in (("val", records[:split]), ("train", records[split:])):
        (outdir / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
        )
    manifest = dict(
        total=len(records),
        validation=split,
        seed=seed,
        tasks=dict(Counter(r["task"] for r in records)),
        sources={
            str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources
        },
    )
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sources", nargs="+")
    p.add_argument("--outdir", default="data/instructions")
    p.add_argument("--minimum", type=int, default=10000)
    p.add_argument("--maximum", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    print(prepare(args.sources, args.outdir, args.minimum, args.maximum, args.seed))


if __name__ == "__main__":
    main()
