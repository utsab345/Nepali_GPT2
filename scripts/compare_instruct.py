"""Paired base/Instruct evaluation on held-out instructions.

Compares a base (pretrained) checkpoint against its Instruct (SFT) sibling
on the same prompts and reports:

  * exact-match rate (reference-following)
  * QA accuracy (with distractors, if present)
  * generation-quality metrics (distinct-1/2, repetition, length)
  * per-task breakdown
  * before/after examples with a human-rating hook

Usage::

    python scripts/compare_instruct.py \
        --base ckpt/best.pt \
        --instruct ckpt/instruct/best.pt \
        --tok tokenizer/nepali_bpe.model \
        --data data/instructions/val.jsonl \
        --out eval/results/base_vs_instruct.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]
import torch  # noqa: E402
from eval.metrics import summarize_generation  # noqa: E402

from nepali_gpt2.generate import generate, load_model_and_tokenizer  # noqa: E402
from nepali_gpt2.sft import format_prompt  # noqa: E402
from scripts.eval_qa import score_candidates  # noqa: E402


def _invoke(model, sp, cfg, device, prompt: str, max_new: int, seed: int) -> str:
    torch.manual_seed(seed)
    return generate(
        model,
        sp,
        cfg,
        device,
        prompt=prompt,
        max_new=max_new,
        temperature=0.7,
        top_k=50,
        top_p=0.92,
    )


def evaluate_checkpoint(
    ckpt_path: str,
    tok_path: str,
    device: str | None,
    rows: list[dict],
    max_new: int,
    seed: int,
) -> dict:
    model, sp, cfg, device = load_model_and_tokenizer(ckpt_path, tok_path, device)
    details = []
    for i, row in enumerate(rows):
        prompt = format_prompt(row["instruction"], row.get("input", ""))
        text = _invoke(model, sp, cfg, device, prompt, max_new, seed + i)
        item = dict(
            id=row.get("id", i),
            task=row.get("task", "unknown"),
            prompt=prompt,
            reference=row["output"],
            output=text,
            exact_match=text.strip() == row["output"].strip(),
            metrics=summarize_generation(sp.encode(text, out_type=str)),
            human_instruction_following_score=None,
        )
        if row.get("distractors"):
            scores = score_candidates(
                model, sp, cfg, device, prompt, [row["output"]] + row["distractors"]
            )
            item["qa_correct"] = max(scores, key=lambda c: scores[c]) == row["output"]
        details.append(item)

    exact = [d["exact_match"] for d in details]
    qa = [d["qa_correct"] for d in details if "qa_correct" in d]
    distinct1 = [d["metrics"]["distinct-1"] for d in details]
    distinct2 = [d["metrics"]["distinct-2"] for d in details]
    repetition = [d["metrics"]["repetition"] for d in details]
    lengths = [d["metrics"]["num_tokens"] for d in details]

    return {
        "checkpoint": ckpt_path,
        "examples": details,
        "exact_match": sum(exact) / len(exact) if exact else None,
        "qa_accuracy": sum(qa) / len(qa) if qa else None,
        "gen_quality": {
            "distinct1_mean": st.mean(distinct1) if distinct1 else None,
            "distinct2_mean": st.mean(distinct2) if distinct2 else None,
            "repetition_mean": st.mean(repetition) if repetition else None,
            "output_len_mean": st.mean(lengths) if lengths else None,
            "output_len_median": st.median(lengths) if lengths else None,
        },
        "per_task": {
            task: {
                "n": sum(1 for d in details if d["task"] == task),
                "exact_match": (
                    sum(d["exact_match"] for d in details if d["task"] == task)
                    / max(sum(1 for d in details if d["task"] == task), 1)
                ),
            }
            for task in sorted({d["task"] for d in details})
        },
    }


def render_markdown(result: dict) -> str:
    """Render a human-readable before/after comparison table."""
    rows = result["results"]
    lines = [
        "| Metric | Base | Instruct | Δ |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("exact_match", "Exact match"),
        ("qa_accuracy", "QA accuracy"),
    ):
        base = rows["base"].get(key)
        inst = rows["instruct"].get(key)
        if base is None or inst is None:
            continue
        delta = inst - base
        lines.append(f"| {label} | {base:.1%} | {inst:.1%} | {delta:+.1%} |")
    for key, label in (
        ("distinct1_mean", "Distinct-1"),
        ("distinct2_mean", "Distinct-2"),
        ("repetition_mean", "Repetition rate"),
        ("output_len_mean", "Mean output length"),
    ):
        base = rows["base"]["gen_quality"].get(key)
        inst = rows["instruct"]["gen_quality"].get(key)
        if base is None or inst is None:
            continue
        delta = inst - base
        lines.append(f"| {label} | {base:.3f} | {inst:.3f} | {delta:+.3f} |")

    examples = []
    for i, (b, ins) in enumerate(
        zip(rows["base"]["examples"], rows["instruct"]["examples"])
    ):
        if ins.get("exact_match", False):
            continue  # show only interesting before/after difference
        examples.append(
            f"\n### Example {i + 1}: `{b['task']}`\n\n"
            f"**Prompt**\n```\n{b['prompt']}\n```\n\n"
            f"**Reference**\n```\n{b['reference']}\n```\n\n"
            f"**Base**\n```\n{b['output']}\n```\n\n"
            f"**Instruct**\n```\n{ins['output']}\n```\n"
        )
    if examples:
        lines.append("\n## Before / after examples")
        lines.extend(examples[:5])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--instruct", required=True)
    p.add_argument("--tok", required=True)
    p.add_argument("--data", required=True, help="Held-out instruction JSONL")
    p.add_argument("--out", default="eval/results/base_vs_instruct.json")
    p.add_argument(
        "--out-markdown",
        default="eval/results/base_vs_instruct.md",
        help="Optional Markdown comparison output",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new", type=int, default=80)
    args = p.parse_args(argv)
    rows = [
        json.loads(line)
        for line in Path(args.data).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Evaluation dataset is empty")

    results: dict = {"seed": args.seed, "max_new": args.max_new, "results": {}}
    for name in ("base", "instruct"):
        print(f"Evaluating {name}...")
        results["results"][name] = evaluate_checkpoint(
            getattr(args, name), args.tok, args.device, rows, args.max_new, args.seed
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")

    markdown = render_markdown(results)
    md_path = Path(args.out_markdown)
    if md_path:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(markdown, encoding="utf-8")
        print(f"Markdown comparison -> {md_path}")

    print(markdown)
    print(f"JSON report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
