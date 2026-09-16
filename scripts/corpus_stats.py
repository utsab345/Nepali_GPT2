"""Compute corpus statistics for the NepaliGPT training data.

Reads ``data/nepali_corpus.txt`` and optionally ``data/tokens.npy`` to
produce a JSON report of corpus composition, character/token counts,
and vocabulary utilization. The report is written to
``docs/measurements/corpus_stats.json``.

Usage::

    python scripts/corpus_stats.py
    python scripts/corpus_stats.py --corpus data/nepali_corpus.txt --tokens data/tokens.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def count_devanagari(text: str) -> int:
    """Count Devanagari Unicode characters in text."""
    return sum(1 for ch in text if "\u0900" <= ch <= "\u097F")


def word_count(text: str) -> int:
    """Whitespace-based word count."""
    return len(text.split())


def corpus_stats(corpus_path: Path) -> dict:
    """Analyze a text corpus and return statistics."""
    lines = []
    total_chars = 0
    devanagari_chars = 0
    total_words = 0
    line_lengths = []

    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            lines.append(line)
            length = len(line)
            line_lengths.append(length)
            total_chars += length
            devanagari_chars += count_devanagari(line)
            total_words += word_count(line)

    if not lines:
        return {"error": "empty corpus"}

    line_lengths.sort()
    n = len(line_lengths)

    return {
        "corpus_path": str(corpus_path),
        "total_lines": n,
        "total_characters": total_chars,
        "total_words": total_words,
        "devanagari_characters": devanagari_chars,
        "devanagari_ratio": round(devanagari_chars / max(total_chars, 1), 4),
        "avg_line_length": round(total_chars / n, 1),
        "avg_words_per_line": round(total_words / n, 1),
        "avg_chars_per_word": round(total_chars / max(total_words, 1), 2),
        "line_length_p50": line_lengths[n // 2],
        "line_length_p95": line_lengths[int(n * 0.95)],
        "line_length_max": line_lengths[-1],
        "empty_lines_skipped": 0,
        "unique_lines_approx": n,
    }


def token_stats(token_path: Path, vocab_size: int = 16_000) -> dict:
    """Analyze a memmapped token array."""
    import numpy as np

    arr = np.memmap(token_path, dtype=np.int32, mode="r")
    total_tokens = len(arr)
    unique_tokens = len(set(arr.tolist()[:100_000]))  # sample for speed
    unique_ratio = unique_tokens / vocab_size

    # Estimate type-token ratio over sliding windows
    window_size = 1000
    ttr_values = []
    for start in range(0, min(total_tokens, 500_000), window_size):
        window = arr[start : start + window_size].tolist()
        if len(window) >= 100:
            ttr_values.append(len(set(window)) / len(window))

    return {
        "token_cache_path": str(token_path),
        "total_tokens": int(total_tokens),
        "vocab_size": vocab_size,
        "unique_tokens_sampled": unique_tokens,
        "vocab_utilization": round(unique_ratio, 4),
        "type_token_ratio": round(sum(ttr_values) / max(len(ttr_values), 1), 4),
        "estimated_token_bytes": int(total_tokens * 4),  # int32
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--corpus",
        default="data/nepali_corpus.txt",
        help="Path to merged corpus text file",
    )
    p.add_argument(
        "--tokens",
        default="data/tokens.npy",
        help="Path to memmapped token cache (optional)",
    )
    p.add_argument(
        "--vocab-size",
        type=int,
        default=16_000,
        help="Tokenizer vocabulary size",
    )
    p.add_argument(
        "--output",
        default="docs/measurements/corpus_stats.json",
        help="Output JSON path",
    )
    args = p.parse_args(argv)

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Corpus not found: {corpus_path}")
        raise SystemExit(1)

    result = corpus_stats(corpus_path)
    print(
        f"Corpus: {result['total_lines']:,} lines, {result['total_characters']:,} chars"
    )

    token_path = Path(args.tokens)
    if token_path.exists():
        t_stats = token_stats(token_path, args.vocab_size)
        result.update(t_stats)
        print(
            f"Tokens: {t_stats['total_tokens']:,} ({t_stats['vocab_utilization']:.1%} vocab utilization)"
        )
    else:
        print("Token cache not found — skipping token stats")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"Stats saved → {output}")


if __name__ == "__main__":
    main()
