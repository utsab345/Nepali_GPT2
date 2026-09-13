"""CLI: inference latency / throughput / memory benchmark (roadmap issue #12).

Usage::

    python scripts/bench_inference.py --ckpt ckpt/best.pt --tok tokenizer/nepali_bpe.model
                                      --n 20 --max-new 64 --prompt-len 64

Reports per-sample latency (p50/p95), tokens/sec, prefill latency and peak
memory (RSS + CUDA). Appends a timestamped JSON report to ``eval/results/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from nepali_gpt2.generate import generate, load_model_and_tokenizer  # noqa: E402

_BASE_PROMPT = "नेपाल एक सुन्दर हिमाली देश हो। "


def _peak_rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    idx = min(len(values) - 1, int(round(q * (len(values) - 1))))
    return sorted(values)[idx]


def _make_prompt(sp, length: int) -> str:
    """A prompt of roughly ``length`` BPE tokens (windowed by the model)."""
    per = max(len(sp.encode(_BASE_PROMPT, out_type=int)), 1)
    raw = _BASE_PROMPT * (length // per + 2)
    return sp.decode(sp.encode(raw, out_type=int)[:length])


def run(args: argparse.Namespace) -> None:
    started = time.monotonic()
    model, sp, cfg, device = load_model_and_tokenizer(args.ckpt, args.tok, args.device)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    load_s = time.monotonic() - started
    base_rss_kb = _peak_rss_kb()
    n_params = model.num_params()

    prompt = _make_prompt(sp, args.prompt_len)
    ids = torch.tensor(
        [[sp.bos_id()] + sp.encode(prompt, out_type=int)],
        dtype=torch.long,
        device=device,
    )[:, -cfg["context_length"] :]

    # Prefill: pure single for-the-whole-prompt forward pass latency.
    with torch.no_grad():
        prefill = []
        for _ in range(max(args.warmup, 1)):
            t0 = time.monotonic()
            model(ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            prefill.append((time.monotonic() - t0) * 1000)

    # Decode: N independent generations of up to max_new tokens.
    gen_times: list[float] = []
    gen_tokens: list[int] = []
    for _ in range(args.n):
        torch.manual_seed(args.seed)
        t0 = time.monotonic()
        out = generate(
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
        dt = time.monotonic() - t0
        gen_times.append(dt)
        gen_tokens.append(len(sp.encode(out, out_type=int)))

    total_s = sum(gen_times)
    tokens_per_s = sum(gen_tokens) / total_s if total_s else 0.0

    report = {
        "task": "inference_benchmark",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": str(args.ckpt),
        "config_hash": hashlib.sha256(
            json.dumps(cfg, sort_keys=True).encode()
        ).hexdigest()[:12],
        "device": str(device),
        "param_count": n_params,
        "load_seconds": round(load_s, 3),
        "n_samples": args.n,
        "max_new": args.max_new,
        "prompt_len_tokens": args.prompt_len,
        "threads": args.threads if args.threads > 0 else "auto",
        "prefill_p50_ms": round(_percentile(prefill, 0.5), 2),
        "gen_p50_ms": round(_percentile(gen_times, 0.5) * 1000, 2),
        "gen_p95_ms": round(_percentile(gen_times, 0.95) * 1000, 2),
        "tokens_per_second": round(tokens_per_s, 2),
        "mean_tokens_per_sample": round(statistics.fmean(gen_tokens), 1),
        "peak_rss_mb": round(_peak_rss_kb() / 1024, 1),
        "baseline_rss_after_load_mb": round(base_rss_kb / 1024, 1),
        "peak_cuda_mb": (
            round(torch.cuda.max_memory_allocated() / 1e6, 1)
            if torch.cuda.is_available()
            else 0.0
        ),
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out = results_dir / f"bench_infer_{stamp}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"device={report['device']}  params={n_params:,}")
    print(
        f"load={load_s:.1f}s  prefill(p50)={report['prefill_p50_ms']}ms  "
        f"gen(p50/p95)={report['gen_p50_ms']}/{report['gen_p95_ms']}ms"
    )
    print(
        f"throughput={tokens_per_s:.1f} tok/s  "
        f"mem(RSS)={report['peak_rss_mb']}MB  mem(CUDA)={report['peak_cuda_mb']}MB"
    )
    print(f"Report saved → {out}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark NepaliGPT inference latency, throughput and memory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", default="ckpt/best.pt")
    p.add_argument("--tok", default="tokenizer/nepali_bpe.model")
    p.add_argument("--device", default=None)
    p.add_argument("--n", type=int, default=20, help="number of generations to time")
    p.add_argument("--max-new", type=int, default=64, help="tokens per generation")
    p.add_argument("--prompt-len", type=int, default=64, help="prompt length in tokens")
    p.add_argument("--warmup", type=int, default=2, help="prefill warmup passes")
    p.add_argument("--threads", type=int, default=0, help="torch threads (0 = auto)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--results-dir", default="eval/results")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(_parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
