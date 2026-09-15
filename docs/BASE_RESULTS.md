# NepaliGPT base results

Measured from the supplied Colab checkpoint (`33,661,952` parameters) on an AMD
Ryzen 5 5500U CPU with one PyTorch thread. CUDA and held-out corpus data were
unavailable, so no perplexity is claimed here.

| Batch | Prompt tokens | Decode p50 (ms) | Decode p95 (ms) | Tokens/s |
|---:|---:|---:|---:|---:|
| 1 | 32 | 3625.14 | 3859.49 | 8.76 |
| 1 | 128 | 8775.78 | 9248.98 | 3.61 |
| 4 | 32 | 10770.27 | 10927.33 | 11.87 |
| 4 | 128 | 33105.25 | 33738.85 | 3.86 |

Quantization measurements are in `docs/measurements/quant_20260915_113923Z.json`.
FP32/FP16/INT8/INT4 checkpoint sizes were 134.69/67.36/67.42/50.93 MB; CPU
throughput was 8.43/1.86/15.72/3.78 tokens/s respectively. INT4 is a packed
reference implementation without an optimized kernel. Memory values are process
high-water marks and are not isolated per-variant measurements.

The base model scored 3/5 on the repository's five-item cloze smoke set. This is
not a broad language-quality or instruction-following evaluation.
