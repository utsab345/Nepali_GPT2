"""Paired base/Instruct QA and generation evaluation on held-out instructions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]
import torch  # noqa: E402
from eval.metrics import summarize_generation  # noqa: E402
from scripts.eval_qa import score_candidates  # noqa: E402

from nepali_gpt2.generate import generate, load_model_and_tokenizer  # noqa: E402
from nepali_gpt2.sft import format_prompt  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--instruct", required=True)
    p.add_argument("--tok", required=True)
    p.add_argument(
        "--data", required=True, help="Held-out instruction JSONL, optional distractors"
    )
    p.add_argument("--out", default="eval/results/base_vs_instruct.json")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new", type=int, default=80)
    args = p.parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.data).read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Evaluation dataset is empty")
    results = {}
    for name in ("base", "instruct"):
        model, sp, cfg, device = load_model_and_tokenizer(
            getattr(args, name), args.tok, args.device
        )
        details = []
        for i, row in enumerate(rows):
            prompt = format_prompt(row["instruction"], row.get("input", ""))
            torch.manual_seed(args.seed + i)
            text = generate(model, sp, cfg, device, prompt=prompt, max_new=args.max_new)
            item = dict(
                id=row.get("id", i),
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
                item["qa_correct"] = (
                    max(scores, key=lambda c: scores[c]) == row["output"]
                )
            details.append(item)
        qa = [d["qa_correct"] for d in details if "qa_correct" in d]
        results[name] = dict(
            checkpoint=getattr(args, name),
            examples=details,
            exact_match=sum(d["exact_match"] for d in details) / len(details),
            qa_accuracy=sum(qa) / len(qa) if qa else None,
        )
        del model
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(dict(seed=args.seed, results=results), indent=2, ensure_ascii=False)
        + "\n"
    )
    print(out)


if __name__ == "__main__":
    main()
