"""Export training-progress snapshots for the Training Explorer demo.

Reads consecutive checkpoints (``step_%06d.pt`` files saved by training)
and produces a single JSON file with, per step:

  * train/val loss (if the checkpoint carries it)
  * a fixed-prompt generation sample
  * top-k next-token probabilities for a probe prompt
  * token-level entropy (optional)

This powers the Gradio Training Explorer (``space/training_explorer.py``),
which lets a visitor slide across training steps and watch generation
quality evolve.

Usage::

    python scripts/export_training_progress.py \
        --ckpt-dir ckpt \
        --tok tokenizer/nepali_bpe.model \
        --output space/training_progress.json \
        --prompt "नेपालको राजधानी"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_PROMPTS = ["नेपालको राजधानी", "हाम्रो देशको इतिहास", "काठमाडौं"]


def find_checkpoints(ckpt_dir: Path) -> list[Path]:
    """Return step checkpoints sorted by step number."""
    pattern = re.compile(r"step_(\d+)\.pt$")
    found = []
    for p in ckpt_dir.glob("step_*.pt"):
        m = pattern.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    found.sort()
    return [p for _, p in found]


def export(
    ckpt_dir: Path,
    tok_path: Path,
    output: Path,
    prompt: str,
    max_new: int,
    top_n: int,
    device: str | None,
) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import torch

    from nepali_gpt2.generate import generate, load_model_and_tokenizer, next_words

    checkpoints = find_checkpoints(ckpt_dir)
    if not checkpoints:
        print(f"No step_*.pt checkpoints found in {ckpt_dir}")
        raise SystemExit(1)
    print(f"Found {len(checkpoints)} checkpoints: {[p.name for p in checkpoints]}")

    torch.manual_seed(42)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    steps = []
    for path in checkpoints:
        model, sp, cfg, dev = load_model_and_tokenizer(str(path), str(tok_path), device)
        gen = generate(
            model,
            sp,
            cfg,
            dev,
            prompt=prompt,
            max_new=max_new,
            temperature=0.8,
            top_k=50,
            top_p=0.92,
        )
        preds = next_words(model, sp, cfg, dev, prompt, top_n=top_n)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        val_loss = ckpt.get("val_loss")
        entry = {
            "checkpoint": path.name,
            "step": int(ckpt.get("step", 0)),
            "val_loss": round(val_loss, 4) if val_loss else None,
            "ppl": round(__import__("math").exp(val_loss), 2) if val_loss else None,
            "generation": gen,
            "next_words": [{"word": w, "prob": round(p, 4)} for w, p in preds],
        }
        steps.append(entry)
        print(
            f"  {path.name}: val={entry['val_loss']} ppl={entry['ppl']} gen='{gen[:60]}…'"
        )

    report = {
        "prompt": prompt,
        "max_new": max_new,
        "n_checkpoints": len(steps),
        "steps": steps,
        "generated_at": __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved → {output}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-dir", default="ckpt")
    p.add_argument("--tok", default="tokenizer/nepali_bpe.model")
    p.add_argument("--output", default="space/training_progress.json")
    p.add_argument("--prompt", default=_PROMPTS[0])
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    export(
        Path(args.ckpt_dir),
        Path(args.tok),
        Path(args.output),
        args.prompt,
        args.max_new,
        args.top_n,
        args.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
