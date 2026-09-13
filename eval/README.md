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
table for the README.

## Status

- [ ] PPL evaluation
- [ ] Generation metrics (distinct-n, repetition)
- [ ] QA / cloze benchmark dataset
- [ ] Baseline comparison table