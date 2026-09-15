"""CLI: cloze/QA accuracy for NepaliGPT on a given benchmark (issue #1).

Scoring  the  answer  vs  the  distractors by  conditional  log-probability
of the continuation given the prefix.

Usage::

    python scripts/eval_qa.py [--benchmark eval/data/ne_cloze.jsonl]
                              [--ckpt ckpt/best.pt] [--tok tokenizer/nepali_bpe.model]

Writes a timestamped ``qa_*.json`` report to ``eval/results/``.
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

import torch  # noqa: E402
from eval.cloze import load_examples  # noqa: E402

from nepali_gpt2.generate import load_model_and_tokenizer  # noqa: E402


@torch.no_grad()
def score_candidates(
    model,
    sp,
    cfg: dict,
    device,
    prefix: str,
    candidates: list[str],
) -> dict[str, float]:
    """Conditional log-probability of each candidate given ``prefix``."""
    ctx_len = int(cfg["context_length"])
    context = [sp.bos_id()] + sp.encode(prefix.strip(), out_type=int)
    scores: dict[str, float] = {}

    for cand in candidates:
        cand_ids = sp.encode(" " + cand, out_type=int)
        window = context[-ctx_len:]
        logp = 0.0
        for tok in cand_ids:
            x = torch.tensor([window[-ctx_len:]], dtype=torch.long, device=device)
            logits, _ = model(x)
            logp += torch.log_softmax(logits[0, -1], dim=-1)[tok].item()
            window.append(tok)
        scores[cand] = logp

    return scores


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate NepaliGPT cloze/QA accuracy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--benchmark",
        default="eval/data/ne_cloze.jsonl",
        help="JSON Lines cloze benchmark",
    )
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
    examples = load_examples(Path(args.benchmark))
    model, sp, cfg, device = load_model_and_tokenizer(args.ckpt, args.tok, args.device)

    correct = 0
    details: list[dict] = []
    for ex in examples:
        candidates = ex["answer"] + ex["distractors"]
        scores = score_candidates(model, sp, cfg, device, ex["prefix"], candidates)
        predicted = max(scores, key=lambda candidate: scores[candidate])
        is_correct = predicted in ex["answer"]
        correct += int(is_correct)
        details.append(
            {
                "id": ex.get("id"),
                "prefix": ex["prefix"],
                "predicted": predicted,
                "correct": is_correct,
                "scores": {c: round(s, 4) for c, s in scores.items()},
            }
        )

    accuracy = correct / max(len(examples), 1)
    report = {
        "task": "qa_accuracy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ckpt": str(args.ckpt),
        "benchmark": str(args.benchmark),
        "n_examples": len(examples),
        "correct": correct,
        "accuracy": accuracy,
        "examples": details,
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out = results_dir / f"qa_{stamp}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"Cloze accuracy: {correct}/{len(examples)} = {accuracy:.3f}")
    print(f"Report saved → {out}")
    return report


def main(argv: list[str] | None = None) -> int:
    run(_parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
