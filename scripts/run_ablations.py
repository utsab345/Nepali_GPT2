"""Run a controlled NepaliGPT ablation matrix and preserve run provenance.

The command validates that each run has an isolated tokenizer/cache/checkpoint
directory, then invokes the existing training and evaluation CLIs. Use
``--dry-run`` to inspect the complete matrix without starting training.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def matrix():
    return itertools.product((8_000, 16_000, 32_000), (128, 256, 512), ("learned", "rope"), ("small", "base"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="ablation_runs")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-steps", type=int, default=15_000)
    p.add_argument("--tokenizer-dir", default="tokenizer")
    p.add_argument("--token-cache", default="data/tokens.npy")
    args = p.parse_args(argv)
    if args.max_steps < 1:
        p.error("max-steps must be positive")
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for vocab, context, position, size in matrix():
        name = f"vocab{vocab}_ctx{context}_{position}_{size}"
        run_dir = root / name
        command = [sys.executable, "-m", "nepali_gpt2", "train", "--model-size", size,
                   "--vocab-size", str(vocab), "--context-length", str(context),
                   "--position-encoding", position, "--max-steps", str(args.max_steps),
                   "--token-cache", args.token_cache, "--ckpt-dir", str(run_dir / "ckpt")]
        entry = dict(name=name, vocab_size=vocab, context_length=context, position_encoding=position,
                     model_size=size, command=command, status="planned")
        manifest.append(entry)
        print(" ".join(command))
        if not args.dry_run:
            entry["started_at"] = datetime.now(timezone.utc).isoformat()
            subprocess.run(command, check=True)
            entry["status"] = "trained"
            entry["finished_at"] = datetime.now(timezone.utc).isoformat()
    output = root / "manifest.json"
    output.write_text(json.dumps(dict(created_at=datetime.now(timezone.utc).isoformat(), runs=manifest), indent=2) + "\n")
    print(f"Manifest saved -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
