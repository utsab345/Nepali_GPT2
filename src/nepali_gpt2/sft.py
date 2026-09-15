"""Response-only supervised fine-tuning for instruction JSONL records."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from nepali_gpt2.generate import load_model_and_tokenizer
from nepali_gpt2.train import build_optimizer, seed_everything


def format_prompt(instruction: str, context: str = "") -> str:
    prompt = f"### निर्देशन:\n{instruction.strip()}\n"
    if context.strip():
        prompt += f"### सन्दर्भ:\n{context.strip()}\n"
    return prompt + "### उत्तर:\n"


class InstructionDataset(Dataset):
    """Right-pad examples and supervise only response tokens (including EOS)."""

    def __init__(self, path: str, sp, max_length: int):
        if max_length < 2 or min(sp.bos_id(), sp.eos_id(), sp.pad_id()) < 0:
            raise ValueError("Need max_length >= 2 and BOS, EOS, PAD tokenizer IDs")
        self.items = []
        for line_no, line in enumerate(
            Path(path).read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("instruction", "output"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"{path}:{line_no}: {key} must be nonempty text")
            context = row.get("input", "")
            if not isinstance(context, str):
                raise ValueError(f"{path}:{line_no}: input must be text")
            prefix = [sp.bos_id()] + sp.encode(
                format_prompt(row["instruction"], context)
            )
            answer = sp.encode(row["output"].strip()) + [sp.eos_id()]
            # Reject silent loss of instructions or reference answers.
            if len(prefix) + len(answer) > max_length + 1:
                raise ValueError(
                    f"{path}:{line_no}: example exceeds max_length={max_length}"
                )
            tokens = prefix + answer
            labels = [-100] * len(prefix) + answer
            x = tokens[:-1] + [sp.pad_id()] * (max_length - len(tokens) + 1)
            y = labels[1:] + [-100] * (max_length - len(tokens) + 1)
            self.items.append((torch.tensor(x), torch.tensor(y)))
        if not self.items:
            raise ValueError(f"Empty instruction dataset: {path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def response_loss(model, x, y):
    logits, _ = model(x)
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), y.flatten(), ignore_index=-100
    )


@torch.no_grad()
def validation_loss(model, loader, device):
    model.eval()
    total = count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        n = int((y != -100).sum())
        total += response_loss(model, x, y).item() * n
        count += n
    return total / count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON defaults; explicit CLI flags override")
    parser.add_argument("--ckpt", default="ckpt/best.pt")
    parser.add_argument("--tok", default="tokenizer/nepali_bpe.model")
    parser.add_argument("--train-data", default="data/instructions/train.jsonl")
    parser.add_argument("--val-data", default="data/instructions/val.jsonl")
    parser.add_argument("--outdir", default="ckpt/instruct")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config:
        defaults = json.loads(Path(preliminary.config).read_text())
        unknown = defaults.keys() - {a.dest for a in parser._actions}
        if unknown:
            parser.error(f"Unknown config keys: {sorted(unknown)}")
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.max_length) < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, max-length and lr must be positive")
    seed_everything(args.seed)
    model, sp, cfg, device = load_model_and_tokenizer(args.ckpt, args.tok, args.device)
    if args.max_length > cfg["context_length"]:
        parser.error("max-length exceeds checkpoint context length")
    train = InstructionDataset(args.train_data, sp, args.max_length)
    val = InstructionDataset(args.val_data, sp, args.max_length)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    optimizer = build_optimizer(model, args.lr, 0.01)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    best, step = math.inf, 0
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = response_loss(model, x.to(device), y.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
        value = validation_loss(model, val_loader, device)
        checkpoint = dict(
            cfg=cfg,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            step=step,
            epoch=epoch + 1,
            val_loss=value,
            sft_args=vars(args),
        )
        torch.save(checkpoint, outdir / "last.pt")
        if value < best:
            best = value
            torch.save(checkpoint, outdir / "best.pt")
        print(f"epoch={epoch + 1} step={step} response_val_loss={value:.4f}")
    return 0
