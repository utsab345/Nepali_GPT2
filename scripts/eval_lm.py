"""CLI: next-token perplexity on the held-out validation split (Week 1).

Usage::

    python scripts/eval_lm.py [--ckpt ckpt/best.pt] [--tok tokenizer/nepali_bpe.model]
                              [--token-cache data/tokens.npy] [--max-batches 200]

Requires a trained checkpoint. Appends a timestamped JSON report to
``eval/results/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from nepali_gpt2.data.dataset import evaluate_perplexity  # noqa: E402
from nepali_gpt2.generate import load_model_and_tokenizer  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate NepaliGPT perplexity on the held-out split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", default="ckpt/best.pt", help="trained checkpoint")
    p.add_argument(
        "--tok", default="tokenizer/nepali_bpe.model", help="tokenizer model"
    )
    p.add_argument(
        "--token-cache", default="data/tokens.npy", help="token array from data-prep"
    )
    p.add_argument(
        "--max-batches", type=int, default=200, help="val batches to average over"
    )
    p.add_argument(
        "--device", default=None, help="torch device (default: cuda if available)"
    )
    p.add_argument(
        "--results-dir", default="eval/results", help="where to write reports"
    )
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> float:
    model, _, cfg, device = load_model_and_tokenizer(args.ckpt, args.tok, args.device)

    ppl = evaluate_perplexity(
        model,
        device,
        token_cache=args.token_cache,
        max_batches=args.max_batches,
    )

    report = {
        "task": "lm_perplexity",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": str(args.ckpt),
        "config_hash": hashlib.sha256(
            json.dumps(cfg, sort_keys=True).encode()
        ).hexdigest()[:12],
        "perplexity": ppl,
        "mean_loss": math.log(ppl),
        "max_batches": args.max_batches,
    }
    report["PPL"] = ppl

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out = results_dir / f"lm_{stamp}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"Perplexity (val): {ppl:.2f}  (mean loss {report['mean_loss']:.4f})")
    print(f"Report saved → {out}")
    return ppl


def main(argv: list[str] | None = None) -> int:
    run(_parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
