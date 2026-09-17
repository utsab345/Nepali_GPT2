"""NepaliLLM-Eval: unified evaluation benchmark for Nepali language models.

Runs all benchmark categories (cloze/factual QA, grammar, commonsense,
translation, summarization) against a language model and reports a
per-category score. Supports both native NepaliGPT checkpoints and any
Hugging Face causal LM.

Two scoring modes:
  * ``selection`` — candidate ranking by conditional log-probability
    (measures how well the model scores the correct answer vs distractors)
  * ``generative`` — greedy generation with exact/fuzzy match against
    references (measures actual text production quality)

Usage::

    python scripts/eval_benchmark.py --ckpt ckpt/best.pt --tok tokenizer/nepali_bpe.model
    python scripts/eval_benchmark.py --model ai-forever/mGPT
    python scripts/eval_benchmark.py --benchmarks ne_cloze ne_grammar ne_commonsense
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

_BENCH_DIR = Path(__file__).resolve().parents[1] / "eval" / "benchmarks"

BENCHMARKS = {
    "ne_cloze": {"file": "ne_cloze.jsonl", "task": "selection"},
    "ne_grammar": {"file": "ne_grammar.jsonl", "task": "selection"},
    "ne_commonsense": {"file": "ne_commonsense.jsonl", "task": "selection"},
    "ne_translation": {"file": "ne_translation.jsonl", "task": "selection"},
    "ne_summarization": {"file": "ne_summarization.jsonl", "task": "selection"},
    "ne_wiki_qa": {"file": "ne_wiki_qa.jsonl", "task": "selection"},
    "ne_reading": {"file": "ne_reading.jsonl", "task": "selection"},
    "ne_continuation": {"file": "ne_continuation.jsonl", "task": "selection"},
}


def load_benchmark(name: str) -> list[dict]:
    """Load a benchmark JSONL file as a list of records."""
    bench = BENCHMARKS.get(name)
    if bench is None:
        raise ValueError(f"Unknown benchmark: {name}")
    path = _BENCH_DIR / bench["file"]
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "prompt" not in row or "answer" not in row:
            raise ValueError(f"{path}:{line_no}: missing prompt or answer")
        records.append(row)
    return records


def _encode(tokenizer, text: str) -> list[int]:
    """Encode text to token ids (works for SentencePiece and HF engines)."""
    if hasattr(tokenizer, "encode"):
        return (
            tokenizer.encode(text, out_type=int)
            if _is_sp(tokenizer)
            else tokenizer.encode(text)
        )
    return tokenizer(text).input_ids[0].tolist()


def _is_sp(tokenizer) -> bool:
    import sentencepiece as spm

    return isinstance(tokenizer, spm.SentencePieceProcessor)


def score_selection(
    model,
    tokenizer,
    prompt: str,
    answers: list[str],
    distractors: list[str],
    device,
    ctx_length: int,
) -> tuple[str, dict[str, float]]:
    """Score candidates by conditional log-probability; return best and per-candidate scores.

    The prompt is fed to the model, then each candidate is scored as the
    sum of per-token log-probabilities of generating the candidate given
    the prompt (and previously generated candidate tokens).
    """
    import torch

    # Encode prompt with BOS token.
    prefix = (
        [tokenizer.bos_id()] if _is_sp(tokenizer) and tokenizer.bos_id() >= 0 else []
    )
    prefix_ids = prefix + _encode(tokenizer, prompt)

    candidates = answers + distractors
    scores: dict[str, float] = {}
    with torch.no_grad():
        for candidate in candidates:
            cand_ids = _encode(tokenizer, " " + candidate)
            window = prefix_ids[-ctx_length:]
            score = 0.0
            for token_id in cand_ids:
                window_tok = torch.tensor([window], device=device)
                logits, _ = model(window_tok)
                log_probs = torch.log_softmax(logits[0, -1], dim=-1)
                score += float(log_probs[token_id])
                window.append(token_id)
                window = window[-ctx_length:]
            scores[candidate] = score

    best = max(scores, key=lambda c: scores[c])
    return best, scores


def run_native(args):
    """Run benchmark against a native NepaliGPT checkpoint."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from nepali_gpt2.generate import load_model_and_tokenizer

    model, sp, cfg, device = load_model_and_tokenizer(args.ckpt, args.tok, args.device)
    ctx = cfg["context_length"]

    all_results = {}
    for name in args.benchmarks:
        records = load_benchmark(name)
        correct = 0
        per_item = []
        t0 = time.perf_counter()
        for record in records:
            prompt = record["prompt"]
            answers = record["answer"]
            distractors = record.get("distractors", [])
            best, scores = score_selection(
                model, sp, prompt, answers, distractors, device, ctx
            )
            ok = best in answers
            correct += ok
            per_item.append(
                {
                    "id": record.get("id", ""),
                    "prompt": prompt,
                    "answer": answers,
                    "best": best,
                    "scores": {k: round(v, 4) for k, v in scores.items()},
                    "correct": ok,
                }
            )
        elapsed = time.perf_counter() - t0
        accuracy = correct / max(len(records), 1)
        all_results[name] = {
            "n_items": len(records),
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "seconds": round(elapsed, 2),
            "items": per_item,
        }
        print(f"{name}: {correct}/{len(records)} ({accuracy:.1%}) in {elapsed:.1f}s")

    return all_results


def run_hf(args):
    """Run benchmark against a Hugging Face causal LM."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(cast(Any, device))
    model.eval()
    ctx = min(getattr(model.config, "n_positions", 512), 512)

    # Wrap to make encode work with SentencePiece-style API
    class HFTokenizerAdapter:
        def __init__(self, tok):
            self.tok = tok

        def encode(self, text, out_type=None):
            return self.tok.encode(text, return_tensors="pt")[0].tolist()

        @property
        def bos_id(self):
            return self.tok.bos_token_id

    adapter = HFTokenizerAdapter(tokenizer)

    # Model adapter — need to return (logits, None) tuple like NepaliGPT
    class HFModelAdapter:
        def __init__(self, m, dev):
            self.m = m
            self.dev = dev

        def __call__(self, x):
            out = self.m(input_ids=x, return_dict=True)
            return out.logits, None

    model_adapter = HFModelAdapter(model, device)

    all_results = {}
    for name in args.benchmarks:
        records = load_benchmark(name)
        correct = 0
        per_item = []
        t0 = time.perf_counter()
        for record in records:
            prompt = record["prompt"]
            answers = record["answer"]
            distractors = record.get("distractors", [])
            best, scores = score_selection(
                model_adapter, adapter, prompt, answers, distractors, device, ctx
            )
            ok = best in answers
            correct += ok
            per_item.append(
                {
                    "id": record.get("id", ""),
                    "prompt": prompt,
                    "answer": answers,
                    "best": best,
                    "correct": ok,
                }
            )
        elapsed = time.perf_counter() - t0
        accuracy = correct / max(len(records), 1)
        all_results[name] = {
            "n_items": len(records),
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "seconds": round(elapsed, 2),
            "items": per_item,
        }
        print(f"{name}: {correct}/{len(records)} ({accuracy:.1%}) in {elapsed:.1f}s")

    return all_results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Run the NepaliLLM-Eval benchmark suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--ckpt",
        help="NepaliGPT checkpoint path (native mode)",
    )
    p.add_argument(
        "--tok",
        default="tokenizer/nepali_bpe.model",
        help="SentencePiece tokenizer path",
    )
    p.add_argument(
        "--model",
        help="HF model ID for baseline evaluation (e.g. ai-forever/mGPT)",
    )
    p.add_argument(
        "--benchmarks",
        nargs="+",
        default=sorted(BENCHMARKS),
        choices=sorted(BENCHMARKS),
        help="Which benchmarks to run",
    )
    p.add_argument("--device", default=None)
    p.add_argument(
        "--results-dir",
        default="eval/results",
        help="Where to write the JSON report",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if not args.ckpt and not args.model:
        p.error("Provide either --ckpt (native) or --model (HF baseline)")

    import torch

    torch.manual_seed(args.seed)

    if args.ckpt:
        results = run_native(args)
        engine = "native"
        label = args.ckpt
    else:
        results = run_hf(args)
        engine = "hf"
        label = args.model

    overall = sum(r["correct"] for r in results.values()) / max(
        sum(r["n_items"] for r in results.values()), 1
    )
    report = {
        "benchmark": "NepaliLLM-Eval",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "engine": engine,
        "model": label,
        "per_category": results,
        "overall_accuracy": round(overall, 4),
        "total_items": sum(r["n_items"] for r in results.values()),
        "total_correct": sum(r["correct"] for r in results.values()),
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    safe_label = label.replace("/", "-").replace(" ", "_")
    out = results_dir / f"eval_benchmark_{safe_label}_{stamp}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        f"\nOverall accuracy: {report['total_correct']}/{report['total_items']} "
        f"({overall:.1%})"
    )
    print(f"Report saved → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
