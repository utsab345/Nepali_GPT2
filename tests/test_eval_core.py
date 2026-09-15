"""Unit tests for the pure eval helper modules (cloze + results table)."""

import json
import random
from pathlib import Path

from eval.cloze import build_cloze_examples, load_examples, split_sentences, write_jsonl
from eval.table import build_markdown_table
from scripts.eval_ngram import evaluate

import numpy as np

_CORPUS = (
    "नेपालको राजधानी काठमाडौं शहर हो। "
    "भारतको राजधानी दिल्ली शहर हो। "
    "नेपालको राष्ट्रिय चरा डाँफे शहर हो। "
    "भारतको राष्ट्रिय चरा मोर शहर हो। "
    "नेपालको राष्ट्रिय पशु गाई शहर हो। "
    "भारतको राष्ट्रिय पशु बाघ शहर हो।"
)


def test_split_sentences_on_danda_and_dot() -> None:
    parts = split_sentences("एक वाक्य। दुई वाक्य . तीन वाक्य? चार वाक्य!")
    assert parts == ["एक वाक्य", "दुई वाक्य", "तीन वाक्य", "चार वाक्य"]


def test_build_cloze_examples_structure() -> None:
    examples = build_cloze_examples(_CORPUS, rng=random.Random(42), max_examples=10)
    assert examples, "expected at least one example"
    for ex in examples:
        assert ex["prefix"].strip(), "prefix must not be empty"
        assert len(ex["answer"]) == 1
        assert len(ex["distractors"]) == 3
        assert ex["answer"][0] not in ex["distractors"]
        assert ex["answer"][0] not in ex["prefix"].split()


def test_cloze_jsonl_roundtrip(tmp_path: Path) -> None:
    examples = build_cloze_examples(_CORPUS, rng=random.Random(0), max_examples=3)
    path = tmp_path / "bench.jsonl"
    write_jsonl(examples, path)
    assert load_examples(path) == examples


def test_table_renders_groups_by_model(tmp_path: Path) -> None:
    (tmp_path / "lm.json").write_text(
        json.dumps({"task": "lm_perplexity", "model": "ckpt/best.pt", "PPL": 14.47})
    )
    (tmp_path / "gen.json").write_text(
        json.dumps(
            {
                "task": "generation_quality",
                "model": "ckpt/best.pt",
                "aggregate": {"distinct-1": 0.3, "distinct-2": 0.5, "repetition": 0.2},
            }
        )
    )
    table = build_markdown_table(tmp_path)
    assert "ckpt/best.pt" in table
    assert "PPL" in table and "14.470" in table
    assert "0.300" in table and "0.200" in table


def test_table_empty_dir(tmp_path: Path) -> None:
    assert "No evaluation results" in build_markdown_table(tmp_path)


def test_table_reads_qa_accuracy(tmp_path):
    (tmp_path / "qa.json").write_text(
        json.dumps(dict(task="qa_accuracy", ckpt="base", accuracy=0.75))
    )
    assert "0.750" in build_markdown_table(tmp_path)


def test_bigram_baseline_returns_finite_metrics() -> None:
    tokens = np.array([0, 1] * 50, dtype=np.int32)
    metrics = evaluate(tokens, vocab_size=4, alpha=0.1)
    assert metrics["tokens"] == 4
    assert metrics["perplexity"] > 0
    assert 0 <= metrics["top5_accuracy"] <= 1
