#!/usr/bin/env bash
# Run the full NepaliGPT evaluation pipeline against a checkpoint.
#
# Usage:
#   bash scripts/run_eval_pipeline.sh               # uses ckpt/best.pt
#   bash scripts/run_eval_pipeline.sh --ckpt ckpt/instruct/best.pt --tok tokenizer/nepali_bpe.model
#
# Runs (in order):
#   1. eval_lm.py          — held-out perplexity
#   2. eval_generation.py  — distinct-1/2, repetition, sentence length
#   3. eval_qa.py          — cloze/QA accuracy
#   4. eval_benchmark.py   — full NepaliLLM-Eval suite (145 items, 6 categories)
#   5. eval_tokenizer.py   — tokenizer efficiency comparison
#   6. benchmark_table.py  — aggregate all results into Markdown
#
# Outputs land in eval/results/ and a summary table at eval/results/pipeline.md

set -euo pipefail

CKPT="${1:-ckpt/best.pt}"
TOK="${2:-tokenizer/nepali_bpe.model}"
RESULTS_DIR="eval/results"
mkdir -p "$RESULTS_DIR"

echo "======================================"
echo "NepaliGPT Evaluation Pipeline"
echo "Checkpoint: $CKPT"
echo "Tokenizer:  $TOK"
echo "======================================"

STEP=1
run() {
    echo ""
    echo "--- Step $STEP: $* ---"
    "$@"
    STEP=$((STEP + 1))
}

run python scripts/eval_lm.py --ckpt "$CKPT" --tok "$TOK" \
    --results-dir "$RESULTS_DIR"
run python scripts/eval_generation.py --ckpt "$CKPT" --tok "$TOK" \
    --results-dir "$RESULTS_DIR"
run python scripts/eval_qa.py --ckpt "$CKPT" --tok "$TOK" \
    --results-dir "$RESULTS_DIR"
run python scripts/eval_benchmark.py --ckpt "$CKPT" --tok "$TOK" \
    --results-dir "$RESULTS_DIR" --benchmarks \
    ne_cloze ne_grammar ne_commonsense ne_translation ne_summarization ne_wiki_qa
run python scripts/eval_tokenizer.py --tok "$TOK"

echo ""
echo "======================================"
echo "Aggregating results..."
echo "======================================"
python scripts/benchmark_table.py --results-dir "$RESULTS_DIR" \
    --output "$RESULTS_DIR/pipeline.md"

echo ""
echo "Pipeline complete. Results in:"
echo "  $RESULTS_DIR/"
echo "Summary table: $RESULTS_DIR/pipeline.md"