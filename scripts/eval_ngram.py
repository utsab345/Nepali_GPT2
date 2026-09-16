"""Evaluate a transparent bigram baseline on the same token split as NepaliGPT."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def evaluate(tokens: np.ndarray, vocab_size: int, alpha: float) -> dict[str, float]:
    split = int(len(tokens) * 0.95)
    train, valid = tokens[:split], tokens[split:]
    transitions: dict[int, Counter[int]] = defaultdict(Counter)
    for prev, nxt in zip(train[:-1], train[1:]):
        transitions[int(prev)][int(nxt)] += 1
    total_nll = 0.0
    correct1 = correct5 = 0
    count = 0
    for prev, nxt in zip(valid[:-1], valid[1:]):
        row = transitions.get(int(prev), Counter())
        denominator = sum(row.values()) + alpha * vocab_size
        probability = (row[int(nxt)] + alpha) / denominator
        total_nll -= math.log(probability)
        top = sorted(row, key=lambda k: row[k], reverse=True)[:5]
        if int(nxt) in top[:1]:
            correct1 += 1
        if int(nxt) in top:
            correct5 += 1
        count += 1
    if not count:
        raise ValueError("token cache is too short for a validation split")
    return {
        "perplexity": math.exp(total_nll / count),
        "top1_accuracy": correct1 / count,
        "top5_accuracy": correct5 / count,
        "tokens": count,
        "unique_contexts": len(transitions),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-cache", default="data/tokens.npy")
    parser.add_argument("--vocab-size", type=int, default=16_000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--results-dir", default="eval/results")
    args = parser.parse_args(argv)
    if args.vocab_size < 2 or args.alpha <= 0:
        parser.error("vocab-size must be >= 2 and alpha must be positive")
    tokens = np.load(args.token_cache, mmap_mode="r")
    started = time.perf_counter()
    metrics = evaluate(tokens, args.vocab_size, args.alpha)
    metrics["seconds"] = time.perf_counter() - started
    report = {
        "task": "baseline_ngram",
        "model": "N-gram (bigram)",
        "params": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "token_cache": str(args.token_cache),
        "vocab_size": args.vocab_size,
        "smoothing_alpha": args.alpha,
        **metrics,
        "PPL": metrics["perplexity"],
    }
    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    output = output_dir / f"ngram_{stamp}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Bigram baseline PPL: {metrics['perplexity']:.2f}; "
        f"top-1: {metrics['top1_accuracy']:.3f}"
    )
    print(f"Report saved -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
