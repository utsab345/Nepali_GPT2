"""Unit tests for evaluation metrics (pure, no torch required)."""

import math

import pytest
from eval.metrics import (
    distinct_n,
    mean_sentence_length,
    perplexity_from_loss,
    repetition_rate,
    summarize_generation,
)


def test_distinct_n_all_unique() -> None:
    assert distinct_n(["a", "b", "c"], 1) == 1.0


def test_distinct_n_with_repeats() -> None:
    assert distinct_n(["a", "a", "b"], 1) == 2 / 3


def test_distinct_n_bigrams() -> None:
    tokens = ["a", "a", "a"]  # bigrams (a,a) x2 -> 1 unique / 2
    assert distinct_n(tokens, 2) == 0.5


def test_distinct_n_too_few_tokens() -> None:
    assert distinct_n(["only"], 2) == 0.0
    assert distinct_n([], 1) == 0.0


def test_repetition_rate_values() -> None:
    assert repetition_rate(["a", "a", "b"], 1) == pytest.approx(1 / 3)
    assert repetition_rate(["a", "b", "c"], 1) == 0.0
    assert repetition_rate(["a", "a", "a"], 4) == 0.0  # no 4-gram windows


def test_mean_sentence_length() -> None:
    assert mean_sentence_length([["a", "b", "c"], ["x", "y"]]) == 2.5
    assert mean_sentence_length([]) == 0.0
    assert mean_sentence_length([[], ["x"]]) == 1.0


def test_perplexity_from_loss() -> None:
    assert perplexity_from_loss(0.0) == 1.0
    assert math.isclose(perplexity_from_loss(math.log(2)), 2.0)


def test_summarize_generation_keys_and_counts() -> None:
    out = summarize_generation(["k", "o", "k", "o"])
    assert set(out) == {"distinct-1", "distinct-2", "repetition", "num_tokens"}
    assert out["num_tokens"] == 4.0
