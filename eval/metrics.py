"""Pure generation-quality metrics for Nepali text (no torch required).

These operate on token sequences (e.g. SentencePiece ids turned into
pieces, or whitespace-tokenized words) and are shared by the generation
eval script and the planned baseline comparisons.

Definitions follow the standard text-generation eval practice:

* distinct-n — fraction of *unique* n-grams among all n-grams. A higher
  value means more vocabulary diversity; ``distinct-1`` and ``distinct-2``
  are the usual pair reported in LLM papers.
* repetition_rate(n) — 1 - distinct-n, i.e. the fraction of n-grams that
  are repeats. High repetition is the classic failure mode of small
  decoder-only models.
* mean_sentence_length — average number of tokens per sentence.
* perplexity_from_loss — exp(loss), converting a mean cross-entropy loss
  to the standard per-token perplexity number.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

Token = str


def ngrams(tokens: Sequence[Token], n: int) -> Iterator[tuple[Token, ...]]:
    """Yield all length-``n`` sliding windows over ``tokens``."""
    if n <= 0 or len(tokens) < n:
        return
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def distinct_n(tokens: Sequence[Token], n: int = 1) -> float:
    """Fraction of unique n-grams over all n-grams (0.0 with too few)."""
    all_grams = list(ngrams(tokens, n))
    if not all_grams:
        return 0.0
    return len(set(all_grams)) / len(all_grams)


def repetition_rate(tokens: Sequence[Token], n: int = 4) -> float:
    """Fraction of repeated n-grams (1 - distinct-n; 0.0 with too few)."""
    all_grams = list(ngrams(tokens, n))
    if not all_grams:
        return 0.0
    return 1.0 - len(set(all_grams)) / len(all_grams)


def mean_sentence_length(sentences: Sequence[Sequence[Token]]) -> float:
    """Average token count per non-empty sentence (0.0 when empty)."""
    lengths = [len(s) for s in sentences if s]
    return sum(lengths) / len(lengths) if lengths else 0.0


def perplexity_from_loss(mean_loss: float) -> float:
    """Convert a mean cross-entropy loss into per-token perplexity."""
    return math.exp(mean_loss)


def summarize_generation(tokens: Sequence[Token]) -> dict[str, float]:
    """Compact metric summary (distinct-1/2, repetition) for a completion."""
    return {
        "distinct-1": distinct_n(tokens, 1),
        "distinct-2": distinct_n(tokens, 2),
        "repetition": repetition_rate(tokens, 4),
        "num_tokens": float(len(tokens)),
    }
