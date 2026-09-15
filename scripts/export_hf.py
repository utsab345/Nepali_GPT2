"""Export a validated native NepaliGPT bundle; optionally upload to Hugging Face."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
import torch  # noqa: E402

from nepali_gpt2.generate import load_model_and_tokenizer  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tok", required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--upload", action="store_true")
    args = p.parse_args(argv)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    for field in ("data", "training", "evaluation", "limitations"):
        if not isinstance(metadata.get(field), str) or not metadata[field].strip():
            p.error(f"metadata must contain {field} text based on actual run records")
    model, sp, cfg, _ = load_model_and_tokenizer(args.ckpt, args.tok, "cpu")
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    if checkpoint.get("precision", "fp32") != "fp32":
        p.error("Export the original FP32 checkpoint")
    from nepali_gpt2.hf_export import to_transformers

    converted = to_transformers(model, sp.bos_id(), sp.eos_id(), sp.pad_id())
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            model=model.state_dict(),
            cfg=cfg,
            step=checkpoint.get("step"),
            val_loss=checkpoint.get("val_loss"),
        ),
        outdir / "model.pt",
    )
    shutil.copyfile(args.tok, outdir / "tokenizer.model")
    (outdir / "nepali_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    converted.save_pretrained(outdir)
    card = f"""---
language:
- ne
library_name: transformers
tags:
- text-generation
- nepali
---
# {args.repo_id}

NepaliGPT decoder-only model exported as standard Transformers GPT-2 weights.
Install `transformers`, `sentencepiece`, `torch`, and `huggingface_hub` for the
example below. The additional native `model.pt` requires the source package
from https://github.com/utsab345/Nepali_GPT2.

## Usage

```python
from huggingface_hub import hf_hub_download
import sentencepiece as spm
import torch
from transformers import AutoModelForCausalLM

repo = "{args.repo_id}"
tokenizer = spm.SentencePieceProcessor(model_file=hf_hub_download(repo, "tokenizer.model"))
model = AutoModelForCausalLM.from_pretrained(repo)
ids = torch.tensor([[tokenizer.bos_id()] + tokenizer.encode("नेपाल एक सुन्दर")])
output = model.generate(ids, max_new_tokens=40, do_sample=False)
print(tokenizer.decode(output[0].tolist()))
```

For an instruction-tuned checkpoint, construct the prompt with
`nepali_gpt2.sft.format_prompt(instruction, context)`.

## Architecture

```json
{json.dumps(cfg, indent=2)}
```
"""
    for field in ("data", "training", "evaluation", "limitations"):
        card += f"\n## {field.title()}\n\n{metadata[field]}\n"
    (outdir / "README.md").write_text(card, encoding="utf-8")
    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo_id, exist_ok=True)
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(outdir),
            allow_patterns=[
                "model.pt",
                "tokenizer.model",
                "config.json",
                "nepali_config.json",
                "generation_config.json",
                "*.safetensors",
                "*.safetensors.index.json",
                "README.md",
            ],
        )
    print(outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
