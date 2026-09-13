# Evaluation suite (roadmap Week 1)

Planned benchmark story to replace the single perplexity number:

| Scope | Metric |
|---|---|
| Language modeling | next-token PPL on held-out Nepali text |
| Generation quality | distinct-1 / distinct-2, repetition rate, avg sentence length |
| Understanding | small Nepali QA / cloze set (curated or semi-automatic from Wikipedia/OSCAR) |

Baselines: mGPT (or a small multilingual Llama) on the same metrics.

Scripts live in `scripts/eval*.py`; aggregate results are logged to
`results/` with timestamps and config hashes, then rendered as a Markdown
table via `scripts/benchmark_table.py`.

## Status

- [x] PPL evaluation — `python scripts/eval_lm.py`
- [x] Generation metrics (distinct-n, repetition) — `python scripts/eval_generation.py`
- [x] QA / cloze benchmark — `eval/data/ne_cloze.jsonl` (curated gold) +
      `scripts/build_qa_benchmark.py` (auto generation) +
      `python scripts/eval_qa.py` (accuracy)
- [x] Baseline runner — `python scripts/eval_baseline.py --model ai-forever/mGPT`
- [x] Results table — `python scripts/benchmark_table.py`

Baseline benchmark numbers still need to be produced on a GPU machine
(CUDA available); the commits here add all the measurement tooling.