import torch
from scripts.bench_inference import benchmark_case

from nepali_gpt2.model import NepaliGPT
from test_model import tiny_cfg


def test_batched_decode_counts_actual_token_steps():
    cfg = tiny_cfg()
    row = benchmark_case(NepaliGPT(cfg), cfg, torch.device("cpu"), 2, 8, 3, 2, 1)
    assert row["generated_tokens"] == 12
    assert row["tokens_per_second"] > 0
    assert row["gen_p95_ms"] >= row["gen_p50_ms"] > 0
    assert row["peak_cuda_mb"] == 0
