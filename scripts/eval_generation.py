"""CLI: generation-quality metrics for NepaliGPT completions (Week 1).

Usage::

    python scripts/eval_generation.py [--prompt "नेपाल एक सुन्दर"] [--prompt "..."] ...
                                      [--num-samples 5] [--max-new 80]

Samples ``num-samples`` completions per prompt, scores them with
``eval/metrics.py`` (distinct-1/2, repetition, sentence length) and
appends an aggregated JSON report to ``eval/results/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from eval.metrics import (  # noqa: E402
    distinct_n,
    mean_sentence_length,
    repetition_rate,
    summarize_generation,
)

from nepali_gpt2.generate import generate, load_model_and_tokenizer  # noqa: E402

DEFAULT_PROMPTS = ["नेपाल एक सुन्दर", "हाम्रो देशको इतिहास"]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate NepaliGPT generation quality (distinct-n, repetition).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--prompt", action="append", default=None, help="prompt (repeatable)"
    )
    p.add_argument("--num-samples", type=int, default=5, help="completions per prompt")
    p.add_argument(
        "--max-new", type=int, default=80, help="tokens to generate per sample"
    )
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.92)
    p.add_argument("--ckpt", default="ckpt/best.pt", help="trained checkpoint")
    p.add_argument(
        "--tok", default="tokenizer/nepali_bpe.model", help="tokenizer model"
    )
    p.add_argument(
        "--device", default=None, help="torch device (default: cuda if available)"
    )
    p.add_argument(
        "--results-dir", default="eval/results", help="where to write reports"
    )
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    model, sp, cfg, device = load_model_and_tokenizer(args.ckpt, args.tok, args.device)

    prompts = args.prompt or DEFAULT_PROMPTS
    completions: list[str] = []
    per_prompt: list[dict] = []

    for prompt in prompts:
        samples: list[str] = []
        for _ in range(args.num_samples):
            text = generate(
                model,
                sp,
                cfg,
                device,
                prompt=prompt,
                max_new=args.max_new,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            samples.append(text)
            completions.append(text)
        per_prompt.append(
            {
                "prompt": prompt,
                "samples": [
                    summarize_generation(sp.encode(t, out_type=str)) for t in samples
                ],
            }
        )

    token_sequences = [sp.encode(t, out_type=str) for t in completions]
    flat_tokens = [tok for seq in token_sequences for tok in seq]

    aggregate = {
        "samples": len(completions),
        "prompts": len(prompts),
        "distinct-1": distinct_n(flat_tokens, 1),
        "distinct-2": distinct_n(flat_tokens, 2),
        "repetition": repetition_rate(flat_tokens, 4),
        "mean_sentence_token_len": mean_sentence_length(token_sequences),
        "mean_sentence_char_len": sum(len(t) for t in completions)
        / max(len(completions), 1),
    }

    report = {
        "task": "generation_quality",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ckpt": str(args.ckpt),
        "generation_args": {
            "num_samples": args.num_samples,
            "max_new": args.max_new,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
        },
        "aggregate": aggregate,
        "per_prompt": per_prompt,
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out = results_dir / f"gen_{stamp}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(
        f"distinct-1: {aggregate['distinct-1']:.3f} | "
        f"distinct-2: {aggregate['distinct-2']:.3f} | "
        f"repetition: {aggregate['repetition']:.3f} | "
        f"mean sent len: {aggregate['mean_sentence_token_len']:.1f} tokens"
    )
    print(f"Report saved → {out}")
    return report


def main(argv: list[str] | None = None) -> int:
    run(_parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
