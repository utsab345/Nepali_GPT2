# 🇳🇵 NepaliGPT

> *An open decoder-only language model trained from scratch for Nepali.*

**41M tokens · 33.7M parameters · 16K Nepali BPE · 14.47 PPL**

[![CI](https://github.com/utsab345/Nepali_GPT2/actions/workflows/ci.yml/badge.svg)](https://github.com/utsab345/Nepali_GPT2/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/utsab345/Nepali_GPT2)](https://github.com/utsab345/Nepali_GPT2/releases)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-NepaliGPT--base-yellow)](https://huggingface.co/utsabdahal34/NepaliGPT-base)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`[Demo](space/app.py)` · `[Model](https://huggingface.co/utsabdahal34/NepaliGPT-base)` · `[Dataset](docs/DATASET_CARD.md)` · `[Technical Report](docs/TECHNICAL_REPORT.md)` · `[Benchmarks](eval/benchmarks/)`

---

**Sample generation** — the base model, greedy, default settings:

> **Prompt**: `नेपाल एक सुन्दर`
>
> नेपाल एक सुन्दर देश हो। यहाँ मुख्यतया धान, गहुँ, उखु, आलु, तोरी तथा विभिन्न किसिमका तरकारीहरू उत्पादन गरिन्छ। नेपालको राजधानी काठमाडौं हो र यो सगरमाथा, लुम्बिनी जस्ता पर्यटकीय स्थलहरूका लागि विश्वभरि प्रसिद्ध छ।

NepaliGPT is an **end-to-end** Nepali LLM project: it downloads its own
corpus, trains a Devanagari-aware SentencePiece tokenizer, trains a GPT-2
style transformer **from random initialisation** (no transfer learning), and
ships evaluation, quantization, instruction tuning, and a serving API. A
visitor can go from this README to generated Nepali text in about 30 seconds.

---

## Why NepaliGPT?

- **~30M speakers, little NLP attention** — most multilingual tokenizers
  fragment Devanagari script into broken subwords. NepaliGPT trains its own
  16K BPE on Nepali text, so `नेपाल` and `काठमाडौं` become natural tokens.
- **Fully from scratch** — corpus → tokenizer → model. No pretrained
  weights from any other model.
- **Small enough to run anywhere** — 34M parameters, dynamic-INT8 CPU
  inference at ~16 tokens/s on a laptop.
- **Priced for research** — train to convergence in ~113 minutes on a T4,
  and ablate every design choice yourself.

---

## Results

### Base model (v1.0)

| Metric | Value |
|---|---|
| Final validation loss | **2.9960** |
| Perplexity | **14.47** |
| Parameters | **33.66M** |
| Training time | ~113 min (Tesla T4) |
| Training steps | 15,000 |
| Vocab | 16K Nepali BPE |
| Context | 512 |

### Training curve

![Training Loss](assets/train-test.png)

### Benchmark results (NepaliLLM-Eval, 145 curated items)

| Category | Items | NepaliGPT-base |
|---|---:|---:|
| Factual cloze | 40 | _run `scripts/eval_benchmark.py`_ |
| Grammar | 30 | — |
| Commonsense | 25 | — |
| Translation | 20 | — |
| Summarization | 10 | — |
| Wikipedia QA | 20 | — |
| **Overall** | **145** | — |

Full details: [NepaliLLM-Eval](eval/benchmarks/README.md) and
[Technical Report §6](docs/TECHNICAL_REPORT.md).

### CPU inference (Ryzen 5 5500U, 1 thread)

| Precision | Prompt | Throughput | Cloze (5-item) |
|---|---:|---:|---:|
| FP32 | 32 tok | 8.76 tok/s | 3/5 |
| **Dynamic INT8** | **32 tok** | **15.72 tok/s** | **3/5** |
| Packed INT4 | 32 tok | 3.78 tok/s | 3/5 |

Baseline comparisons (mGPT, smoothed bigram) run with
[`scripts/eval_baseline.py`](scripts/eval_baseline.py). Precision caveats in
[`docs/BASE_RESULTS.md`](docs/BASE_RESULTS.md).

---

## Model family

| Model | Params | Use-case |
|---|---:|---|
| NepaliGPT-small | ~17M | Experiments, CPU serving |
| **NepaliGPT-base** | **~34M** | **Primary release** |
| NepaliGPT-Instruct | ~34M | Instruction following (SFT on base) |

---

## Quickstart

```bash
pip install -e .
python examples/quickstart.py            # ~30s to Nepali generation
```

Or, to see the full pipeline:

```bash
# 1. Data (requires HF + Kaggle credentials)
export HF_TOKEN=hf_...
export KAGGLE_USERNAME=... KAGGLE_KEY=...
python -m nepali_gpt2 data-prep           # download corpus + train tokenizer

# 2. Train (base model, ~113 min on a T4)
python -m nepali_gpt2 train

# 3. Generate
python -m nepali_gpt2 generate --prompt "नेपाल एक सुन्दर"
```

See [scripts/run_eval_pipeline.sh](scripts/run_eval_pipeline.sh) for a
one-command evaluation of any checkpoint.

---

## Documentation

| Doc | What it covers |
|---|---|
| [Technical Report](docs/TECHNICAL_REPORT.md) | Paper-style: dataset, tokenizer, architecture, training, eval, ablations, limitations |
| [Dataset Card](docs/DATASET_CARD.md) | Corpus sources, processing, splits, contamination |
| [Benchmarks](eval/benchmarks/README.md) | NepaliLLM-Eval suite: 145 items, 6 categories |
| [Reproduction](docs/REPRODUCE.md) | Full step-by-step reproduction guide |
| [Base Results](docs/BASE_RESULTS.md) | Measured CPU/quantization benchmarks |
| [API](api/README.md) | FastAPI + streaming + Docker |
| [Ablations](scripts/run_ablations.py) | vocab×context×pos×size matrix runner |

---

## Repository structure

```
nepali-gpt2/
├── src/nepali_gpt2/       # Core package (model, train, generate, sft, quantize)
│   └── data/              # Corpus download + SentencePiece tokenizer
├── scripts/               # CLI entrypoints + experiments
├── eval/                  # NepaliLLM-Eval benchmark + metrics
│   └── benchmarks/        # 145 curated items, 6 categories
├── api/                   # FastAPI inference + streaming SSE
├── docker/                # CPU serving image
├── space/                 # HF Space (Gradio demo + Training Explorer)
├── docs/                  # Technical report, dataset card, results
├── examples/              # Quickstart
├── tests/                 # 15 pytest files
└── web/                   # Static browser client
```

---

## Instruction tuning

NepaliGPT-Instruct is fine-tuned on a curated Nepali instruction dataset
using response-only loss masking:

1. Generate candidates from the corpus:
   ```bash
   python scripts/build_nepali_instructions.py \
       --corpus data/nepali_corpus.txt --limit 50000
   ```
2. Review (mark `reviewed: true`), then validate/split:
   ```bash
   python scripts/prepare_instructions.py data/instructions/candidates.jsonl
   ```
3. Fine-tune:
   ```bash
   python scripts/sft_train.py --config configs/sft.json \
       --ckpt ckpt/best.pt --train-data data/instructions/train.jsonl
   ```
4. Compare base vs instruct:
   ```bash
   python scripts/compare_instruct.py \
       --base ckpt/best.pt --instruct ckpt/instruct/best.pt \
       --tok tokenizer/nepali_bpe.model --data data/instructions/val.jsonl
   ```

---

## Evaluation

| Tool | Measures |
|---|---|
| `scripts/eval_lm.py` | Held-out perplexity |
| `scripts/eval_generation.py` | Distinct-1/2, repetition, sentence length |
| `scripts/eval_qa.py` | Cloze/QA accuracy vs distractors |
| `scripts/eval_benchmark.py` | Full NepaliLLM-Eval suite (6 categories) |
| `scripts/eval_baseline.py` | Any HF causal LM as baseline (mGPT, XGLM…) |
| `scripts/eval_tokenizer.py` | Tokenizer efficiency comparison |
| `scripts/benchmark_table.py` | Aggregates all results into Markdown |

Run everything with a single command:

```bash
bash scripts/run_eval_pipeline.sh --ckpt ckpt/best.pt --tok tokenizer/nepali_bpe.model
```

---

## Serving

```bash
NEPALIGPT_DEVICE=cpu uvicorn api.main:app --reload
```

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/generate` | text completion |
| `POST` | `/generate/stream` | SSE token streaming |
| `POST` | `/next_token` | top-k next tokens + probabilities |
| `POST` | `/tokenize` | SentencePiece pieces/ids/decoded |
| `GET` | `/health` | liveness + model info |
| `GET` | `/metrics` | counters, latency, errors |

Docker: `docker build -f docker/Dockerfile -t nepaligpt-api .`

---

## Training observability

Every training run now records a structured `training_metrics.json` and a
multi-panel chart (`ckpt/training_curves.png`) — loss, learning rate,
gradient norms, and tokens/sec:

```bash
python -m nepali_gpt2 train        # produces ckpt/training_curves.png
```

Consecutive checkpoints (`ckpt/step_%06d.pt`) can be exported for the
**Training Explorer** Gradio demo:

```bash
python scripts/export_training_progress.py \
    --ckpt-dir ckpt --tok tokenizer/nepali_bpe.model \
    --output space/training_progress.json
GRADIO_SERVER_NAME=0.0.0.0 python space/training_explorer.py
```

---

## Development

```bash
pip install -e ".[dev]" && pre-commit install
ruff check . && black --check . && mypy && pytest
```

CI runs lint, format, type checks, tests, a smoke-inference job, and eval
sanity checks on every push/PR.

---

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{dahal_nepaligpt,
  author = {Utsab Dahal},
  title = {NepaliGPT},
  year = {2026},
  url = {https://github.com/utsab345/Nepali_GPT2},
  version = {1.0.0}
}
```