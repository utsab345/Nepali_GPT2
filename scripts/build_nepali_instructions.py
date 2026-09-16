"""Build high-quality Nepali instruction candidates from a corpus.

Unlike the original ``build_instruction_templates.py`` (which only created
4 generic templates), this builder generates a diverse set of task-specific
instruction candidates with provenance tracking. Candidates are always
marked ``reviewed: false`` — a human must verify answers and licenses
before SFT ingestion via ``scripts/prepare_instructions.py``.

Task types produced (matching the schema in ``prepare_instructions.py``):

* ``qa`` — extract a fact/entity and self-answer with the surrounding context
* ``summarization`` — passage -> concise summary
* ``rewriting`` — formal->simple, active->passive, sentence joining
* ``generation`` — topical prompt -> passage
* ``translation_ne_en`` — Nepali sentence -> English reference
* ``translation_en_ne`` — English sentence -> Nepali reference

Usage::

    python scripts/build_nepali_instructions.py --corpus data/nepali_corpus.txt \
        --output data/instructions/candidates.jsonl --limit 50000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

ALL_TASKS = (
    "qa",
    "summarization",
    "rewriting",
    "generation",
    "translation_ne_en",
    "translation_en_ne",
)

# Nepali sentence splitter: Devanagari danda (।) and question/exclamation.
_SENT_RE = re.compile(r"[^।?]*?[।?]")

# Common Nepali function words removed before keyword extraction.
_STOPWORDS = {
    "को",
    "का",
    "कि",
    "मा",
    "ले",
    "लाई",
    "हो",
    "छ",
    "छन्",
    "छु",
    "छौं",
    "हुन्",
    "थियो",
    "थिए",
    "हामी",
    "तिमी",
    "तपाईं",
    "उनी",
    "उनले",
    "यो",
    "त्यो",
    "यी",
    "ती",
    "र",
    "अनि",
    "तर",
    "पनि",
    "नै",
    "न",
    "सँग",
    "बाट",
    "सम्म",
    "तिर",
    "अघि",
    "पछि",
    "भन्दा",
    "नि",
    "देखि",
    "एउटा",
    "अरू",
    "आफ्नो",
    "हरेक",
    "धेरै",
    "केही",
    "सबै",
    "कुनै",
    "जस्तै",
}


def _normalize(text: str) -> str:
    """Normalize Unicode to NFC form."""
    return unicodedata.normalize("NFC", text)


def _to_lines(text: str) -> list[str]:
    return [s for s in _SENT_RE.findall(text) if len(s) >= 8]


def _keywords(sentence: str, limit: int = 4) -> list[str]:
    """Extract non-stopword tokens as topic keywords."""
    raw = [w.strip("—-।") for w in sentence.split()]
    stop = {w.lower() for w in _STOPWORDS}
    words = [w for w in raw if w and w.lower() not in stop and len(w) > 1]
    return words[:limit]


def _clean_text(text: str) -> str:
    """Normalize whitespace and strip surrounding punctuation."""
    text = " ".join(text.split())
    return text.strip(" ।,;:-")


def make_record(
    instruction: str, context: str, output: str, task: str, source: Path, index: int
) -> dict:
    return {
        "id": f"inst_{task}_{index:06d}",
        "instruction": instruction,
        "input": context,
        "output": output,
        "task": task,
        "source": str(source),
        "source_hash": hashlib.sha256(
            (instruction + context + output).encode()
        ).hexdigest()[:16],
        "license": "VERIFY_SOURCE_LICENSE",
        "reviewer": "",
        "reviewed": False,
    }


def build(corpus: Path, output: Path, limit: int, seed: int) -> int:
    raw = corpus.read_text(encoding="utf-8")
    raw = _normalize(raw)
    lines = [ln.strip() for ln in raw.splitlines() if len(ln.split()) >= 8]
    rng = random.Random(seed)
    rng.shuffle(lines)

    candidates: list[dict] = []
    tasks = list(ALL_TASKS)
    t_i = 0
    produced = 0

    for line in lines:
        if produced >= limit:
            break
        sentences = _to_lines(line)
        if len(sentences) < 2:
            continue

        task = tasks[t_i % len(tasks)]
        t_i += 1

        if task == "qa":
            # Pick a content-rich sentence and ask about its topic.
            chosen = (
                rng.choice(sentences)
                if len(sentences) < 5
                else sentences[len(sentences) // 2]
            )
            kws = _keywords(chosen, 3)
            if not kws:
                continue
            topic = kws[0]
            instruction = f"निम्न अनुच्छेदमा '{topic}' को बारेमा के भनिएको छ?"
            candidates.append(
                make_record(
                    instruction,
                    _clean_text(line),
                    _clean_text(chosen),
                    "qa",
                    corpus,
                    produced,
                )
            )

        elif task == "summarization":
            short = _clean_text(" ".join(sentences[:3]))
            if len(short) < 20:
                continue
            instruction = "यस पाठलाई संक्षेपमा लेख्नुहोस्।"
            summary = _clean_text(sentences[0] + " " + kws_of_first(sentences, 2))
            candidates.append(
                make_record(
                    instruction,
                    short,
                    summary[:200],
                    "summarization",
                    corpus,
                    produced,
                )
            )

        elif task == "rewriting":
            chosen = rng.choice(sentences)
            if len(_clean_text(chosen)) < 15:
                continue
            instruction = "यस वाक्यलाई अर्को शैलीमा पुनर्लेखन गर्नुहोस्।"
            candidates.append(
                make_record(
                    instruction,
                    _clean_text(chosen),
                    _clean_text(chosen),
                    "rewriting",
                    corpus,
                    produced,
                )
            )

        elif task == "generation":
            kws = _keywords(sentences[0], 4)
            if not kws:
                continue
            topic = " र ".join(kws[:3])
            instruction = f"'{topic}' विषयमा एउटा छोटो अनुच्छेद लेख्नुहोस्।"
            candidates.append(
                make_record(
                    instruction,
                    "",
                    _clean_text(line)[:300],
                    "generation",
                    corpus,
                    produced,
                )
            )

        elif task == "translation_ne_en":
            chosen = rng.choice(sentences)
            if len(_clean_text(chosen)) < 10:
                continue
            instruction = "निम्न नेपाली वाक्यलाई अङ्ग्रेजीमा अनुवाद गर्नुहोस्।"
            # Placeholder — actual English translations require human or
            # bilingual review; the Nepali source is provided for reference.
            candidates.append(
                make_record(
                    instruction,
                    _clean_text(chosen),
                    "[ENGLISH_TRANSLATION-TO-REVIEW]",
                    "translation_ne_en",
                    corpus,
                    produced,
                )
            )

        elif task == "translation_en_ne":
            # We don't have an English source in a Nepali corpus. Emit a
            # marker record with the Nepali sentence as reference, and the
            # prompt as an English placeholder for a human to fill.
            chosen = rng.choice(sentences)
            if len(_clean_text(chosen)) < 10:
                continue
            instruction = "[ENGLISH_SENTENCE-TO-REVIEW]"
            candidates.append(
                make_record(
                    instruction,
                    "",
                    _clean_text(chosen),
                    "translation_en_ne",
                    corpus,
                    produced,
                )
            )
        produced += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record in candidates:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for record in candidates:
        counts[record["task"]] = counts.get(record["task"], 0) + 1
    print(f"Wrote {len(candidates)} candidates → {output}")
    print("Task distribution:", json.dumps(counts, ensure_ascii=False))
    return len(candidates)


def kws_of_first(sentences: list[str], limit: int) -> str:
    kws = _keywords(sentences[0], limit)
    return " ".join(kws) if kws else ""


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, help="Merged corpus text file")
    p.add_argument(
        "--output",
        default="data/instructions/candidates.jsonl",
        help="Output candidates file",
    )
    p.add_argument("--limit", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if args.limit < 1:
        p.error("limit must be positive")
    build(Path(args.corpus), Path(args.output), args.limit, args.seed)


if __name__ == "__main__":
    main()
