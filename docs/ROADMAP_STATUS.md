# Roadmap completion status

The supplied Colab base checkpoint and tokenizer are now available locally under
`ckpt/colab-base/` (large files remain git-ignored). The corpus, small checkpoint,
Instruct checkpoint and reviewed instruction dataset are still missing.
No issue should be closed based on scaffolding alone.

| Issue | Implemented or corrected here | Outstanding acceptance work |
| --- | --- | --- |
| #2 Baselines | Token-weighted PPL, actual generated-token throughput, QA scoring, memory and corpus hash | Run multilingual baseline on the same held-out corpus; publish measured table |
| #3 Instructions | Reviewed-record validation, provenance, deduplication, deterministic train/validation split | Collect 10k–50k examples across six tasks, review quality and source licenses |
| #4 SFT | Response-only labels, JSON config/CLI overrides, validation, best/last checkpoints | Run on collected dataset and pretrained checkpoint |
| #5 Base vs Instruct | Paired seeded prompts, QA scoring, completion metrics and human-rating fields | Train Instruct, run comparison and rate instruction following |
| #6 Base publishing | Validated native checkpoint bundle and model card exporter | Base artifacts received; supply small checkpoint and authenticate to upload to `utsabdahal34` |
| #7 Instruct publishing | Standard Transformers GPT-2 weight conversion and Instruct model-card support | Supply trained Instruct checkpoint and publish |
| #12 Inference | CPU/CUDA synchronization, batch/sequence grid, warmup, p50/p95, exact token counts, load time and memory | Run production checkpoints on target CPU/GPU |
| #13 Quantization | FP32/FP16/dynamic INT8 and packed INT4 reference, reload support, real base-model speed/size and cloze quality reports | Optimized INT4 kernel and held-out PPL delta remain optional follow-up |
| #14 Ablations | Learned/RoPE positions and configurable vocabulary/context; controlled experiment plan | Train matched runs and report PPL, training time, speed and interpretation |
| #15 Tests | Fresh-clone real tokenizer round trip, model/causal/seed tests, SFT and quantization regression tests | See validation summary below |
| #16 Release | Changelog, reproduction/run instructions, and v1.0.0 GitHub release | Future v1.1/v2.0 releases can follow new artifacts |

## Required inputs

- Small pretrained checkpoint and matching tokenizer (base artifacts have been received).
- Original corpus/token caches and a separate held-out text file for baseline evaluation.
- Reviewed instruction source JSONL, including paired Nepali/English examples.
- Hugging Face login with write access to `utsabdahal34` (authenticate locally; do not put credentials in files).

The public Hugging Face account had no listed models when inspected in this session.
Image datasets listed under that account do not supply the missing language artifacts.

## Validation

- CPU test suite covers native model, real SentencePiece, generation, SFT masks,
  evaluation, INT4 error bounds, all precision reloads, and Transformers parity.
- End-to-end CLI smoke checks passed for SFT, a batch/sequence benchmark grid,
  quantization, local Hugging Face export and paired evaluation using a tiny
  synthetic checkpoint. These are software checks, not model-quality results.
- Base-model CPU measurements are recorded in `docs/BASE_RESULTS.md`; CUDA is unavailable.
