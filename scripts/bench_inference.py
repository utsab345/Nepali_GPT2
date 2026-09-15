"""Benchmark fixed-length batched decoding on CPU or CUDA (no KV cache)."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
from nepali_gpt2.generate import load_model_and_tokenizer  # noqa: E402


def _percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lo = int(index)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (
        index - lo
    )


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def decode(model, ids, context_length, max_new):
    """Fixed token count includes EOS; measures model throughput, not text quality."""
    for _ in range(max_new):
        logits, _ = model(ids[:, -context_length:])
        ids = torch.cat((ids, logits[:, -1].argmax(-1, keepdim=True)), dim=1)
    return ids


def benchmark_case(model, cfg, device, batch_size, prompt_len, max_new, n, warmup):
    if min(batch_size, prompt_len, max_new, n) < 1 or warmup < 0:
        raise ValueError(
            "Batch, prompt, generation length and samples must be positive"
        )
    if prompt_len > cfg["context_length"]:
        raise ValueError("Prompt length exceeds model context")
    ids = torch.ones((batch_size, prompt_len), dtype=torch.long, device=device)
    model.eval()
    for _ in range(warmup):
        decode(model, ids, cfg["context_length"], max_new)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    prefill, latency = [], []
    for _ in range(n):
        synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            model(ids)
        synchronize(device)
        prefill.append(time.perf_counter() - start)
        start = time.perf_counter()
        decode(model, ids, cfg["context_length"], max_new)
        synchronize(device)
        latency.append(time.perf_counter() - start)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(
        batch_size=batch_size,
        prompt_len_tokens=prompt_len,
        max_new=max_new,
        n_samples=n,
        prefill_p50_ms=_percentile(prefill, 0.5) * 1000,
        gen_p50_ms=_percentile(latency, 0.5) * 1000,
        gen_p95_ms=_percentile(latency, 0.95) * 1000,
        tokens_per_second=batch_size * max_new * n / sum(latency),
        generated_tokens=batch_size * max_new * n,
        process_peak_rss_mb=rss / (1024**2 if sys.platform == "darwin" else 1024),
        peak_cuda_mb=(
            torch.cuda.max_memory_allocated(device) / 1e6
            if device.type == "cuda"
            else 0
        ),
    )


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="ckpt/best.pt")
    p.add_argument("--tok", default="tokenizer/nepali_bpe.model")
    p.add_argument("--device", default=None)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--prompt-len", type=int, nargs="+", default=[64])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", default="eval/results")
    return p.parse_args(argv)


def run(args):
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    start = time.perf_counter()
    model, _, cfg, device = load_model_and_tokenizer(args.ckpt, args.tok, args.device)
    synchronize(device)
    load_s = time.perf_counter() - start
    cases = [
        benchmark_case(
            model, cfg, device, batch, length, args.max_new, args.n, args.warmup
        )
        for batch in args.batch_sizes
        for length in args.prompt_len
    ]
    report = dict(
        task="inference_benchmark",
        model=args.ckpt,
        device=str(device),
        timestamp=datetime.now(timezone.utc).isoformat(),
        cfg=cfg,
        param_count=model.num_params(),
        load_seconds=load_s,
        torch_version=str(torch.__version__),
        threads=torch.get_num_threads(),
        cases=cases,
        notes="Greedy fixed-length decoding; EOS does not stop. RSS is process lifetime high-water mark; no KV cache.",
    )
    if len(cases) == 1:
        report.update(cases[0])
    directory = Path(args.results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    out = directory / f"bench_infer_{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    run(_parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
