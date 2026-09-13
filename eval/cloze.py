"""Semi-automatic cloze-benchmark generation from plain text (Week 1).

Derives fill-in-the-blank examples by masking frequent content words in
real Nepali sentences and sampling plausible distractors, so a QA/cloze
benchmark can be auto-built from Wikipedia/OSCAR text (issue #1).
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ClozeExample = dict[str, Any]

_SENTENCE_SPLIT_RE = re.compile(r"[।.!?]+")
_MIN_WORD_OCCURRENCES = 3
_MIN_SENTENCE_TOKENS = 5
_N_DISTRACTORS = 3


def split_sentences(text: str) -> list[str]:
    """Split text on Nepali/Sentence danda and Latin punctuation."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _word_counts(sentences: Sequence[str]) -> Counter[str]:
    return Counter(w.lower() for s in sentences for w in s.split())


def build_cloze_examples(
    text: str,
    *,
    rng: random.Random,
    max_examples: int = 1000,
) -> list[ClozeExample]:
    """Generate cloze examples by masking frequent words mid-sentence."""
    sentences = [
        s for s in split_sentences(text) if len(s.split()) >= _MIN_SENTENCE_TOKENS
    ]
    freq = _word_counts(sentences)
    anchors = {w for w, c in freq.items() if c >= _MIN_WORD_OCCURRENCES}
    pool = sorted(anchors)

    examples: list[ClozeExample] = []
    for sent in rng.sample(sentences, min(len(sentences), max_examples * 5)):
        tokens = sent.split()
        positions = [i for i, t in enumerate(tokens) if t.lower() in anchors and i > 0]
        if not positions:
            continue
        anchor = tokens[positions[0]].lower()
        prefix = " ".join(tokens[: positions[0]]) + " "
        excluded = {anchor} | {t.lower() for t in tokens[: positions[0]]}
        distractor_pool = [w for w in pool if w not in excluded]
        if not distractor_pool:
            continue
        distractors = rng.sample(
            distractor_pool, min(_N_DISTRACTORS, len(distractor_pool))
        )
        examples.append(
            {"prefix": prefix, "answer": [anchor], "distractors": distractors}
        )
        if len(examples) >= max_examples:
            break
    return examples


def write_jsonl(examples: Sequence[ClozeExample], path: Path) -> None:
    """Write examples as UTF-8 JSON Lines."""
    path.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples),
        encoding="utf-8",
    )


def load_examples(path: Path) -> list[ClozeExample]:
    """Load a JSON Lines benchmark file."""
    if not path.exists():
        raise FileNotFoundError(f"Benchmark not found: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
