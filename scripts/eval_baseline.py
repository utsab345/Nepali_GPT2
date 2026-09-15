"""CLI: evaluate any Hugging Face causal LM as a Nepali baseline (issue #2).

Runs the same measurement as the NepaliGPT eval suite — chunked
perplexity over a text file plus generation-quality metrics — with recorded corpus identity. Token-level perplexity is not directly
comparable across different tokenizers; use identical held-out text.

Usage::

    python scripts/eval_baseline.py --model ai-forever/mGPT \
        --text data/nepali_corpus.txt

Requires ``pip install transformers``. Saves a ``llm_*.json`` report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import torch  # noqa: E402
from eval.cloze import load_examples  # noqa: E402
from eval.metrics import (  # noqa: E402
    distinct_n,
    mean_sentence_length,
    repetition_rate,
)

CHUNK = 512
DEFAULT_PROMPT = "नेपाल एक सुन्दर"


@torch.no_grad()
def perplexity_on_text(model, tokenizer, text: str, device, max_batches: int) -> float:
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    total, count, batches = 0.0, 0, 0
    use_amp = device.type == "cuda"
    for start in range(0, len(ids) - 1, CHUNK):
        if max_batches >= 0 and batches >= max_batches:
            break
        chunk = ids[start : start + CHUNK].unsqueeze(0).to(device)
        if chunk.size(1) < 2:
            continue
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(chunk, labels=chunk)
        tokens = chunk.size(1) - 1
        total += out.loss.item() * tokens
        count += tokens
        batches += 1
    if not count:
        raise ValueError("Evaluation text has no target tokens")
    return math.exp(total / count)


@torch.no_grad()
def sample_and_score(
    model,
    tokenizer,
    device,
    prompt: str,
    num_samples: int,
    max_new: int,
    temperature: float,
    top_p: float,
) -> tuple[list[list[str]], float]:
    gen_input = tokenizer(prompt, return_tensors="pt").to(device)
    seqs: list[list[str]] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(num_samples):
        out = model.generate(
            **gen_input,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        new_tokens = out[0][gen_input.input_ids.size(1) :].tolist()
        seqs.append(tokenizer.convert_ids_to_tokens(new_tokens))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    tokens_per_sec = sum(map(len, seqs)) / max(time.perf_counter() - t0, 1e-6)
    return seqs, tokens_per_sec


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate an HF causal LM as a Nepali baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--benchmark", default="eval/data/ne_cloze.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", required=True, help="HF model id (e.g. ai-forever/mGPT)")
    p.add_argument(
        "--text", default="data/nepali_corpus.txt", help="raw Nepali text to score"
    )
    p.add_argument("--max-batches", type=int, default=200, help="PPL chunks to average")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.92)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--results-dir", default="eval/results", help="where to write reports"
    )
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    text_path = Path(args.text)
    if not text_path.exists():
        print(f"Corpus not found: {text_path}")
        raise SystemExit(1)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model: Any = AutoModelForCausalLM.from_pretrained(args.model)
    model = model.to(device)
    model.eval()

    text = text_path.read_text(encoding="utf-8")
    ppl = perplexity_on_text(model, tokenizer, text, device, args.max_batches)

    seqs, tokens_per_sec = sample_and_score(
        model,
        tokenizer,
        device,
        args.prompt,
        args.num_samples,
        args.max_new,
        args.temperature,
        args.top_p,
    )
    flat = [tok for seq in seqs for tok in seq]

    examples = load_examples(Path(args.benchmark))
    if not examples:
        raise ValueError("QA benchmark is empty")
    correct = 0
    with torch.no_grad():
        for example in examples:
            context = tokenizer.encode(example["prefix"], add_special_tokens=True)
            if not context:
                context = [tokenizer.bos_token_id or tokenizer.eos_token_id]
            scores = {}
            for candidate in example["answer"] + example["distractors"]:
                window = list(context)
                score = 0.0
                for token in tokenizer.encode(
                    " " + candidate, add_special_tokens=False
                ):
                    logits = model(
                        torch.tensor([window[-CHUNK:]], device=device)
                    ).logits
                    score += torch.log_softmax(logits[0, -1], -1)[token].item()
                    window.append(token)
                scores[candidate] = score
            correct += max(scores, key=lambda c: scores[c]) in example["answer"]

    report = {
        "task": "baseline",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "params": model.num_parameters(),
        "PPL": ppl,
        "QA acc": correct / len(examples),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "device": str(device),
        "seed": args.seed,
        "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2 if sys.platform == "darwin" else 1024),
        "peak_cuda_mb": (
            torch.cuda.max_memory_allocated(device) / 1e6
            if device.type == "cuda"
            else 0
        ),
        "distinct-1": distinct_n(flat, 1),
        "distinct-2": distinct_n(flat, 2),
        "repetition": repetition_rate(flat, 4),
        "mean_sentence_token_len": mean_sentence_length(seqs),
        "tokens_per_sec": tokens_per_sec,
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out = results_dir / f'llm_{args.model.replace("/", "-")}_{stamp}.json'
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(
        f"{args.model}: PPL {ppl:.2f} | distinct-1 {report['distinct-1']:.3f} | "
        f"distinct-2 {report['distinct-2']:.3f} | repetition {report['repetition']:.3f} | "
        f"{tokens_per_sec:.1f} tok/s"
    )
    print(f"Report saved → {out}")
    return report


def main(argv: list[str] | None = None) -> int:
    run(_parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
