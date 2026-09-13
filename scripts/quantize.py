"""CLI: FP16 / INT8 quantization experiments (roadmap issue #13).

Usage::

    python scripts/quantize.py --ckpt ckpt/best.pt --tok tokenizer/nepali_bpe.model
                               --outdir ckpt/quantized [--skip-ppl]

For each precision (fp32 baseline, fp16, dynamic INT8) this reports:

* on-disk size of the saved checkpoint
* generation throughput (tokens/sec) on CPU
* perplexity on a few validation batches (needs ``data/tokens.npy``)

Appends a timestamped comparison table to ``eval/results/``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.quantization import quantize_dynamic

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from nepali_gpt2.data.dataset import evaluate_perplexity  # noqa: E402
from nepali_gpt2.generate import generate, load_model_and_tokenizer  # noqa: E402

PROMPT = "नेपाल एक सुन्दर हिमाली देश हो।"
MAX_NEW = 32


def _model_bytes(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def _make_variants(model) -> dict[str, torch.nn.Module]:
    fp16 = copy.deepcopy(model).half()
    int8 = quantize_dynamic(copy.deepcopy(model), {torch.nn.Linear}, dtype=torch.qint8)
    return {"fp32": model, "fp16": fp16, "int8": int8}


def _generation_speed(model, sp, cfg, device: str) -> float:
    torch.manual_seed(42)
    t0 = time.monotonic()
    out = generate(model, sp, cfg, torch.device(device), prompt=PROMPT, max_new=MAX_NEW)
    dt = time.monotonic() - t0
    return len(sp.encode(out, out_type=int)) / dt if dt else 0.0


def run(args: argparse.Namespace) -> None:
    # Quantization runs on CPU; load there regardless of any CUDA default.
    model, sp, cfg, _ = load_model_and_tokenizer(
        args.ckpt, args.tok, args.device or "cpu"
    )
    device = args.device or "cpu"

    variants = _make_variants(model)
    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[
        :12
    ]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cache = Path(args.token_cache)
    ppl_kwargs = {} if not cache.exists() else {"token_cache": str(cache)}
    if args.skip_ppl or not cache.exists():
        if not cache.exists() and not args.skip_ppl:
            print(f"note: {args.token_cache} not found — skipping perplexity")
        ppl_kwargs = {}

    table = {}
    for name, variant in variants.items():
        state_path = outdir / f"model_{name}.pt"
        torch.save(
            {"cfg": cfg, "precision": name, "model": variant.state_dict()}, state_path
        )

        speed = _generation_speed(variant, sp, cfg, device)
        ppl = None
        if ppl_kwargs:
            ppl = evaluate_perplexity(
                variant,
                torch.device(device),
                max_batches=args.ppl_batches,
                **ppl_kwargs,
            )

        table[name] = {
            "size_mb": round(state_path.stat().st_size / 1e6, 2),
            "param_bytes_mb": round(_model_bytes(variant) / 1e6, 2),
            "tokens_per_second": round(speed, 2),
            "perplexity": round(ppl, 3) if ppl else None,
        }

    report = {
        "task": "quantization_experiment",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": str(args.ckpt),
        "config_hash": config_hash,
        "device": device,
        "variants": table,
        "notes": "INT8 uses torch.quantization.quantize_dynamic on Linear layers (CPU).",
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out = results_dir / f"quant_{stamp}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"{'variant':<6}{'size(MB)':>10}{'tok/s':>9}{'PPL':>10}")
    print("-" * 40)
    for name, row in table.items():
        print(
            f"{name:<6}{row['size_mb']:>10.2f}{row['tokens_per_second']:>9.2f}"
            f"{row['perplexity'] if row['perplexity'] else '-':>10}"
        )
    print(f"Report saved → {out}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quantize NepaliGPT to FP16/INT8 and compare size, speed, PPL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", default="ckpt/best.pt")
    p.add_argument("--tok", default="tokenizer/nepali_bpe.model")
    p.add_argument("--outdir", default="ckpt/quantized")
    p.add_argument("--device", default="cpu")
    p.add_argument("--token-cache", default="data/tokens.npy")
    p.add_argument("--ppl-batches", type=int, default=2)
    p.add_argument("--skip-ppl", action="store_true")
    p.add_argument("--results-dir", default="eval/results")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(_parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
