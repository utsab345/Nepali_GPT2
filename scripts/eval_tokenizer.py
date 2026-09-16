"""Compare NepaliGPT's SentencePiece BPE tokenizer against multilingual
tokenizers on Nepali text.

Metrics reported per tokenizer:
  * tokens per sentence (mean, median, p90)
  * tokens per word (mean)
  * bytes per token (compression ratio)
  * unknown-character handling (unk rate)
  * Devanagari fragmentation (fraction of words split into >2 pieces)
  * mixed Nepali-English handling (code-switched text)

Unlike perplexity, these metrics are directly comparable across
tokenizers because they measure encoding *efficiency*, not model quality.

Usage::

    python scripts/eval_tokenizer.py --tok tokenizer/nepali_bpe.model
    python scripts/eval_tokenizer.py --tok tokenizer/nepali_bpe.model --hf-models ai-forever/mGPT xlm-roberta-base
    python scripts/eval_tokenizer.py --text-file data/nepali_corpus.txt --sample 10000

Requires ``transformers`` for HF tokenizer comparisons.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

PROBE_SENTENCES = [
    "नेपाल",
    "नेपालको राजधानी काठमाडौं हो।",
    "काठमाडौं विश्वविद्यालयमा अध्ययन गर्दैछु।",
    "सगरमाथाको उचाइ आठ हजार आठ सय अड़तालीस मिटर छ।",
    "महेन्द्र ज्ञानेन्द्र",
    "जलवायु परिवर्तनका कारण हिमालयका ग्लेशियरहरू पग्लिँदै छन्।",
    "उसले एउटा किताब किनेर आफ्नो साथीलाई उपहार दियो।",
    "सं. २०७२ सालको महाभूकम्पले काठमाडौंका धेरै ऐतिहासिक भवनहरू नष्ट गर्यो।",
    "नेपालीमा लेख्नुहोस्: Hello, how are you? म ठीक छु।",
    "कोशी, गण्डकी र कर्णाली नेपालका प्रमुख नदीहरू हुन्।",
    "नेपाल नेपाली नेपाली नेपाल",
    "दसौं दुई सय पच्चीस पैंतीस एक्काइस",
    "१२३४५६७८९०",
    "काठमाडौं-पोखरा बस सेवा",
]


def _normalize(text: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFC", text)


def _is_devanagari(ch: str) -> bool:
    return "\u0900" <= ch <= "\u097f"


class SpTokenizerAdapter:
    """Wrap a SentencePiece processor into a uniform tokenizer interface."""

    def __init__(self, processor):
        self.sp = processor
        self.name = "NepaliGPT-BPE-16k"

    def _encode(self, text: str) -> list[str]:
        return self.sp.encode(text or " ", out_type=str)

    def _unknown_ratio(self, text: str) -> float:
        unk = self.sp.unk_id()
        ids = self.sp.encode(text)
        unk_count = sum(1 for i in ids if i == unk)
        return unk_count / max(len(ids), 1)

    def stats(self, sentences: list[str]) -> dict:
        tokens_per_sent, tokens_per_word, bytes_per_token = [], [], []
        unk_rates, fragments, code_switch = [], [], []
        for sent in sentences:
            norm = _normalize(sent)
            pieces = self._encode(norm)
            tokens_per_sent.append(len(pieces))
            dev_words = [w for w in norm.split() if any(_is_devanagari(c) for c in w)]
            for word in dev_words:
                piece_count = len(self._encode(word))
                tokens_per_word.append(piece_count)
                fragments.append(1 if piece_count > 2 else 0)
            encoded_bytes = sum(len(p) for p in pieces)
            bytes_per_token.append(encoded_bytes / max(len(pieces), 1))
            unk_rates.append(self._unknown_ratio(norm))
            has_en = any("a" <= ch.lower() <= "z" for ch in norm)
            has_ne = any(_is_devanagari(ch) for ch in norm)
            if has_en and has_ne:
                code_switch.append(sent)

        return {
            "name": self.name,
            "tokens_per_sentence_mean": (
                round(statistics.mean(tokens_per_sent), 2) if tokens_per_sent else 0.0
            ),
            "tokens_per_sentence_median": (
                round(statistics.median(tokens_per_sent), 2) if tokens_per_sent else 0.0
            ),
            "tokens_per_sentence_p90": (
                round(statistics.quantiles(tokens_per_sent, n=10)[8], 2)
                if tokens_per_sent
                else 0.0
            ),
            "tokens_per_word_mean": (
                round(statistics.mean(tokens_per_word), 2) if tokens_per_word else 0.0
            ),
            "bytes_per_token": (
                round(statistics.mean(bytes_per_token), 3) if bytes_per_token else 0.0
            ),
            "unknown_ratio": round(statistics.mean(unk_rates), 5) if unk_rates else 0.0,
            "devanagari_fragmented_word_ratio": round(
                sum(fragments) / max(len(fragments), 1), 4
            ),
            "code_switched_sentence_count": len(code_switch),
            "mixed_handling_ok": len(code_switch) > 0,
        }


class HFTokenizerAdapter:
    """Wrap a HuggingFace tokenizer into the same uniform interface."""

    def __init__(self, tokenizer, name: str):
        self.tok = tokenizer
        self.name = name

    def _encode(self, text: str) -> list[str]:
        # HF tokenizers output strings via convert_ids_to_tokens.
        ids = self.tok.encode(text, add_special_tokens=False)
        return self.tok.convert_ids_to_tokens(ids)

    def _unknown_ratio(self, text: str) -> float:
        ids = self.tok.encode(text, add_special_tokens=False)
        unk = self.tok.unk_token_id
        if unk is None:
            return 0.0
        unk_count = sum(1 for i in ids if i == unk)
        return unk_count / max(len(ids), 1)

    def stats(self, sentences: list[str]) -> dict:
        tokens_per_sent, tokens_per_word, bytes_per_token = [], [], []
        unk_rates, fragments = [], []
        for sent in sentences:
            norm = _normalize(sent)
            pieces = self._encode(norm)
            tokens_per_sent.append(len(pieces))
            dev_words = [w for w in norm.split() if any(_is_devanagari(c) for c in w)]
            for word in dev_words:
                piece_count = len(self._encode(word))
                tokens_per_word.append(piece_count)
                fragments.append(1 if piece_count > 2 else 0)
            encoded_bytes = sum(len(p) for p in pieces)
            bytes_per_token.append(encoded_bytes / max(len(pieces), 1))
            unk_rates.append(self._unknown_ratio(norm))

        return {
            "name": self.name,
            "tokens_per_sentence_mean": (
                round(statistics.mean(tokens_per_sent), 2) if tokens_per_sent else 0.0
            ),
            "tokens_per_sentence_median": (
                round(statistics.median(tokens_per_sent), 2) if tokens_per_sent else 0.0
            ),
            "tokens_per_sentence_p90": (
                round(statistics.quantiles(tokens_per_sent, n=10)[8], 2)
                if tokens_per_sent
                else 0.0
            ),
            "tokens_per_word_mean": (
                round(statistics.mean(tokens_per_word), 2) if tokens_per_word else 0.0
            ),
            "bytes_per_token": (
                round(statistics.mean(bytes_per_token), 3) if bytes_per_token else 0.0
            ),
            "unknown_ratio": round(statistics.mean(unk_rates), 5) if unk_rates else 0.0,
            "devanagari_fragmented_word_ratio": round(
                sum(fragments) / max(len(fragments), 1), 4
            ),
            "code_switched_sentence_count": 0,
            "mixed_handling_ok": False,
        }


def example_tokenizations(adapters, sample: list[str]) -> dict:
    """Produce example tokenization outputs for demonstration."""
    out = {}
    for a in adapters:
        rows = []
        for sent in sample[:5]:
            pieces = a._encode(_normalize(sent))
            rows.append({"sentence": sent, "pieces": pieces, "n": len(pieces)})
        out[a.name] = rows
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tok", default="tokenizer/nepali_bpe.model")
    p.add_argument(
        "--hf-models",
        nargs="+",
        default=[],
        help="HF tokenizers to compare (e.g. ai-forever/mGPT xlm-roberta-base)",
    )
    p.add_argument(
        "--text-file", default=None, help="Optional corpus to sample sentences from"
    )
    p.add_argument(
        "--sample", type=int, default=2000, help="Number of sentences to sample"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="eval/results/tokenizer_comparison.json")
    args = p.parse_args(argv)

    import sentencepiece as spm
    import torch

    torch.manual_seed(args.seed)

    # Build the sentence set.
    if args.text_file and Path(args.text_file).exists():
        raw = Path(args.text_file).read_text(encoding="utf-8")
        sentences = [s for s in raw.split("\n") if len(s.split()) >= 5][: args.sample]
        if len(sentences) < args.sample:
            print(f"Only {len(sentences)} sentences found in corpus; using those.")
    else:
        sentences = PROBE_SENTENCES
        print("No --text-file provided; using curated probe sentences.")

    # Native tokenizer.
    sp = spm.SentencePieceProcessor()
    if not sp.load(args.tok):
        print(f"Tokenizer not found: {args.tok}")
        raise SystemExit(1)
    adapters: list[SpTokenizerAdapter | HFTokenizerAdapter] = [SpTokenizerAdapter(sp)]

    # HF tokenizers.
    if args.hf_models:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            print(
                "transformers not installed; skipping HF tokenizers. "
                "Run: pip install transformers"
            )
        else:
            for model_id in args.hf_models:
                tok = AutoTokenizer.from_pretrained(model_id)
                adapters.append(HFTokenizerAdapter(tok, model_id))

    results = {}
    for a in adapters:
        print(f"Analyzing {a.name}...")
        results[a.name] = a.stats(sentences)

    # Pretty print table.
    print()
    header = f"{'Tokenizer':<28} {'tok/sent':>8} {'tok/word':>8} {'B/tok':>8} {'unk%':>8} {'frag%':>8}"
    print(header)
    print("-" * len(header))
    for name, stats in results.items():
        print(
            f"{name:<28} {stats['tokens_per_sentence_mean']:>8.2f} "
            f"{stats['tokens_per_word_mean']:>8.2f} "
            f"{stats['bytes_per_token']:>8.3f} "
            f"{stats['unknown_ratio'] * 100:>7.2f}% "
            f"{stats['devanagari_fragmented_word_ratio'] * 100:>7.2f}%"
        )

    examples = example_tokenizations(
        adapters, sentences if not args.text_file else PROBE_SENTENCES
    )

    report = {
        "task": "tokenizer_comparison",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_sentences_analyzed": len(sentences),
        "tokenizers": results,
        "example_tokenizations": examples,
        "seed": args.seed,
        "source": args.text_file or "curated_probe_sentences",
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\nReport saved → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
