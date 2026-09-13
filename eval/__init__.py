"""Evaluation metrics and suites for NepaliGPT (roadmap Week 1).

Automatic, reproducible measurements of model quality on Nepali text:

* Language modeling — next-token perplexity on the held-out split.
* Generation quality — vocabulary diversity (distinct-1/2), repetition
  rate, and average sentence length of sampled completions.

The metric functions in ``metrics`` are pure and dependency-free (no
torch), so they are unit-testable on their own and reusable for the
baseline comparisons (mGPT, multilingual Llama, …) planned for Week 1.
Inference-backed runners (PPL, generation sampling) live in
``scripts/eval_lm.py`` and ``scripts/eval_generation.py``.
"""
