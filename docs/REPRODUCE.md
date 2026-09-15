# Reproducing NepaliGPT

Run commands from the repository root with Python 3.11 or 3.12.
The README's historical training result is not a new measurement from this change.

## Environment and base model

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m nepali_gpt2 data-prep
python -m nepali_gpt2 train --model-size base --seed 42
python scripts/eval_lm.py --ckpt ckpt/best.pt
python scripts/eval_qa.py --ckpt ckpt/best.pt
python scripts/eval_generation.py --ckpt ckpt/best.pt
```

Data preparation requires the upstream dataset credentials described in the README.
Archive the raw corpus, tokenizer, raw-int32 token cache (despite its `.npy` suffix),
command, package versions, Git commit and hardware alongside each run.
Train small separately with `--model-size small --ckpt-dir ckpt/small`.

## Instruction dataset and SFT

Source JSONL records use these fields:

```json
{"instruction":"नेपालको राजधानी के हो?","input":"","output":"काठमाडौं।","task":"qa","source":"author-created","license":"CC0-1.0","reviewed":true,"reviewer":"actual-reviewer-name"}
```

Record real provenance and actual review; the example above demonstrates the schema.
Supported tasks are `qa`, `summarization`, `rewriting`, `translation_ne_en`,
`translation_en_ne`, and `generation`. Obtain translations from parallel sources or
bilingual reviewers; copying a sentence is not a translation dataset. Check factual
answers, faithful summaries, preserved rewrite meaning, language direction and personal
data. Keep evaluation sources separate from training. The preparer rejects unreviewed
records and duplicate prompts before a seeded 95/5 split.

```bash
python scripts/prepare_instructions.py reviewed/source1.jsonl reviewed/source2.jsonl
python scripts/sft_train.py --config configs/sft.json --ckpt ckpt/best.pt
python scripts/compare_instruct.py --base ckpt/best.pt --instruct ckpt/instruct/best.pt \
  --tok tokenizer/nepali_bpe.model --data data/instructions/val.jsonl
```

SFT masks instruction and padding labels. Overlong examples are rejected rather than
silently dropping answers. Use the `format_prompt` function in `nepali_gpt2.sft` at
inference too. Add `distractors` to held-out QA records for candidate accuracy.
Human reviewers must fill instruction-following scores; exact match and diversity
alone do not establish instruction-following quality.

## Benchmarking and quantization

```bash
python scripts/bench_inference.py --device cpu --batch-sizes 1 4 8 --prompt-len 32 128 512
python scripts/bench_inference.py --device cuda --batch-sizes 1 4 8 --prompt-len 32 128 512
python scripts/quantize.py --device cpu --token-cache data/tokens.npy
python scripts/benchmark_table.py
```

Benchmarking uses fixed-length greedy decoding, including EOS steps, without a KV cache.
Throughput counts token steps directly. CPU RSS is a process lifetime high-water mark;
use separate processes to compare isolated memory. INT4 is a packed-weight reference
that dequantizes for matrix multiplication, so reduced file size need not mean speedup.
Quantized files can be reloaded with the ordinary loader on CPU.

## Baselines

```bash
pip install transformers
python scripts/eval_baseline.py --model ai-forever/mGPT --text data/heldout.txt
```

Use the same held-out documents and QA set for all models. Never pass the full training
corpus as held-out evaluation. PPL and subword diversity depend on tokenization;
PPL across different vocabularies is not a direct quality ranking. Record revisions,
text hashes and hardware. No baseline scores are claimed until this is run.

## Ablations

Change one factor at a time relative to base/16k/512/learned, retaining the same raw
training and validation documents, seed, optimization settings and raw-text budget:

| Factor | Values |
| --- | --- |
| Vocabulary | 8000, 16000, 32000 |
| Context | 128, 256, 512 |
| Position | learned, rope |
| Model size | small, base |

`train` accepts `--vocab-size`, `--context-length`, and `--position-encoding`.
For vocabulary variants, retrain SentencePiece and regenerate each cache; changing
only the model vocabulary is invalid. Keep tokenizer/cache/checkpoint paths separate.
For example, `python -m nepali_gpt2 train --position-encoding rope --ckpt-dir ckpt/rope`.
Run each checkpoint through LM and inference evaluation; record wall-clock training
seconds, validation loss/PPL, tokens/sec and hardware. Compare token-level PPL only
within a shared tokenizer; for vocabulary comparisons also report byte-normalized NLL.
The experiment runs and byte-normalized evaluator remain outstanding.

## Publishing and release

```bash
pip install -e '.[hub]'
python scripts/export_hf.py --ckpt ckpt/best.pt --tok tokenizer/nepali_bpe.model \
  --repo-id utsabdahal34/NepaliGPT-base --metadata model-metadata.json --outdir dist/base
```

The metadata JSON must contain nonempty `data`, `training`, `evaluation`, and
`limitations` text based on actual run records. Add the real benchmark results, never
placeholder scores. Review the exported README and artifacts, then authenticate locally
and rerun with `--upload`. Repeat for `NepaliGPT-small` and `NepaliGPT-Instruct`.
Exports include both native weights and standard Transformers GPT-2 safetensors.
SentencePiece handles input encoding in the model-card example. Conversion supports
learned-position checkpoints and is verified against native logits; RoPE export is rejected.

Before tagging v1.0.0, run `ruff check .`, `black --check .`, `mypy`, and `pytest`, verify
published artifact downloads and reproduction commands, and attach measured reports.
No tag or published release is created by these instructions.
