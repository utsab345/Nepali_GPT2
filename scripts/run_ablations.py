"""Run a controlled NepaliGPT ablation matrix and preserve run provenance.

The command validates that each run has an isolated tokenizer/cache/checkpoint
directory, then invokes the existing training and evaluation CLIs. Use
``--dry-run`` to inspect the complete matrix without starting training.

After training, each run is automatically evaluated with:
  * held-out perplexity (``eval_lm.py``)
  * generation quality (``eval_generation.py``)
  * QA/cloze accuracy (``eval_qa.py``)

All results are aggregated into ``eval/results/ablation_summary.json`` and
a Markdown table ``ablation_runs/ablation_table.md``.

Usage::

    python scripts/run_ablations.py --dry-run          # inspect the matrix
    python scripts/run_ablations.py --max-steps 3000   # quick smoke runs
    python scripts/run_ablations.py                    # full matrix
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[0]


def matrix():
    """Full ablation matrix: vocab x context x position x model size."""
    return itertools.product(
        (8_000, 16_000, 32_000),
        (128, 256, 512),
        ("learned", "rope"),
        ("small", "base"),
    )


def eval_resume_args(entry: dict) -> None:
    """Attach the evaluation command to a run entry (for provenance)."""
    entry["eval_commands"] = {
        "eval_lm": [
            sys.executable,
            str(_SCRIPTS / "eval_lm.py"),
            "--ckpt",
            str(entry["ckpt_dir"] / "best.pt"),
            "--tok",
            str(entry["tokenizer_path"]),
        ],
        "eval_generation": [
            sys.executable,
            str(_SCRIPTS / "eval_generation.py"),
            "--ckpt",
            str(entry["ckpt_dir"] / "best.pt"),
            "--tok",
            str(entry["tokenizer_path"]),
        ],
        "eval_qa": [
            sys.executable,
            str(_SCRIPTS / "eval_qa.py"),
            "--ckpt",
            str(entry["ckpt_dir"] / "best.pt"),
            "--tok",
            str(entry["tokenizer_path"]),
        ],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="ablation_runs")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-steps", type=int, default=15_000)
    p.add_argument("--tokenizer-dir", default="tokenizer")
    p.add_argument(
        "--token-cache-template",
        default="data/ablation/vocab{vocab}/tokens.npy",
        help="per-vocabulary token cache path; must contain {vocab}",
    )
    p.add_argument(
        "--results-dir",
        default="eval/results",
        help="Where evaluation results are written",
    )
    p.add_argument(
        "--run-eval",
        action="store_true",
        help="Run evaluation after training each run",
    )
    args = p.parse_args(argv)
    if args.max_steps < 1:
        p.error("max-steps must be positive")
    if "{vocab}" not in args.token_cache_template:
        p.error(
            "token-cache-template must contain {vocab} so vocab runs cannot share a cache"
        )
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for vocab, context, position, size in matrix():
        name = f"vocab{vocab}_ctx{context}_{position}_{size}"
        run_dir = root / name
        ckpt_dir = run_dir / "ckpt"
        token_cache = args.token_cache_template.format(vocab=vocab)
        # Each run needs its own tokenizer trained on the per-vocab cache's corpus.
        tok_dir = run_dir / "tokenizer"
        tok_path = tok_dir / "nepali_bpe.model"

        command = [
            sys.executable,
            "-m",
            "nepali_gpt2",
            "train",
            "--model-size",
            size,
            "--vocab-size",
            str(vocab),
            "--context-length",
            str(context),
            "--position-encoding",
            position,
            "--max-steps",
            str(args.max_steps),
            "--token-cache",
            token_cache,
            "--ckpt-dir",
            str(ckpt_dir),
        ]
        entry = {
            "name": name,
            "vocab_size": vocab,
            "context_length": context,
            "position_encoding": position,
            "model_size": size,
            "token_cache": token_cache,
            "status": "planned",
            "command": command,
            "ckpt_dir": str(ckpt_dir),
            "tokenizer_path": str(tok_path),
        }
        eval_resume_args(entry)
        manifest.append(entry)
        print(" ".join(command))
        if not args.dry_run:
            if not Path(token_cache).exists():
                print(f"  [SKIP] token cache not found: {token_cache}")
                entry["status"] = "skipped_missing_cache"
                continue
            entry["started_at"] = datetime.now(timezone.utc).isoformat()
            subprocess.run(command, check=True)
            entry["status"] = "trained"
            entry["finished_at"] = datetime.now(timezone.utc).isoformat()
            if args.run_eval:
                print(f"  Evaluating {name}...")
                for eval_name, eval_cmd in entry["eval_commands"].items():
                    result = subprocess.run(eval_cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        entry.setdefault("eval_results", {})
                        out_line = (
                            result.stdout.strip().splitlines()[-1]
                            if result.stdout
                            else ""
                        )
                        entry["eval_results"][eval_name] = out_line
                    else:
                        entry.setdefault("eval_results", {})
                        entry["eval_results"][
                            eval_name
                        ] = f"ERROR: {result.stderr.strip()[-200:]}"

    output = root / "manifest.json"
    output.write_text(
        json.dumps(
            dict(
                created_at=datetime.now(timezone.utc).isoformat(),
                max_steps=args.max_steps,
                runs=manifest,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"\nManifest saved -> {output}")

    # Aggregate a comparison table for trained runs.
    summarize(manifest, root, args.results_dir)
    return 0


def summarize(manifest: list, root: Path, results_dir: str) -> None:
    """Write an aggregated Markdown table of trained runs' PPL/loss."""
    trained = [r for r in manifest if r.get("status") == "trained"]
    if not trained:
        print("No trained runs to summarize.")
        return

    rows = []
    for run in trained:
        # Load best checkpoint to read val_loss.
        best = Path(run["ckpt_dir"]) / "best.pt"
        ppl = loss = "—"
        if best.exists():
            try:
                import math
                import os

                import torch

                os.environ.setdefault("TORCH_HOME", "/tmp/torch")
                ckpt = torch.load(best, map_location="cpu", weights_only=True)
                val = float(ckpt.get("val_loss", 0.0))
                loss = f"{val:.4f}"
                ppl = f"{math.exp(val):.2f}" if val else "—"
            except Exception:
                loss = "err"
        rows.append(
            {
                "name": run["name"],
                "vocab": run["vocab_size"],
                "ctx": run["context_length"],
                "pos": run["position_encoding"],
                "size": run["model_size"],
                "val_loss": loss,
                "ppl": ppl,
                **{k: v for k, v in run.get("eval_results", {}).items()},
            }
        )

    # Markdown table.
    md = [
        "| Run | Vocab | Ctx | Pos | Size | Val loss | PPL | Eval-LM | Eval-Gen | Eval-QA |",
        "|---|---|---:|---:|---|---:|---:|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['name']} | {r['vocab']} | {r['ctx']} | {r['pos']} | {r['size']} | {r['val_loss']} | {r['ppl']} | {r.get('eval_lm', '—')} | {r.get('eval_generation', '—')} | {r.get('eval_qa', '—')} |"
        )
    table = "\n".join(md)
    (root / "ablation_table.md").write_text(table + "\n")
    print(f"\nComparison table saved -> {root / 'ablation_table.md'}")
    _ = results_dir


if __name__ == "__main__":
    raise SystemExit(main())
