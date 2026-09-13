"""Train a NepaliGPT model from the cached token array.

Generates ``ckpt/best.pt`` (lowest validation loss) and periodic
crash-recovery checkpoints, plus a training-loss plot in ``loss.png``.

Design notes
------------
* Mixed precision (``torch.amp``) is enabled automatically on CUDA and
  disabled on CPU, so the script runs unchanged on either backend.
* The LR schedule (linear warm-up -> cosine decay) and optimiser (AdamW
  with weight decay on 2-D params only) follow the GPT-2 / nanoGPT recipe.
* ``torch.compile`` is applied on CUDA as a free inference/training speed-up.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nepali_gpt2.config import MODEL_CONFIGS, TRAIN_DEFAULTS
from nepali_gpt2.data.dataset import TokenDataset, eval_loss
from nepali_gpt2.model import NepaliGPT


def make_lr_fn(
    lr: float,
    warmup: int,
    total_steps: int,
    min_lr_ratio: float,
):
    """Return a warmup-then-cosine-decay learning rate schedule.

    The returned function maps a step number to the LR for that step:
    linearly from ``~0`` at step 0 to ``lr`` at ``warmup``, then a cosine
    decay down to ``lr * min_lr_ratio`` at ``total_steps``.

    Warm-up stabilises early optimisation (large gradients right after
    random init); the cosine tail lets the model settle into a minimum.
    """

    def lr_schedule(step: int) -> float:
        if step < warmup:
            return lr * (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        cosine = 0.5 * (1 + math.cos(math.pi * t))
        return lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine)

    return lr_schedule


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """AdamW with weight decay applied to 2-D parameters only.

    Weight matrices (>= 2-D) get L2-style decay; biases and LayerNorm
    gains (1-D) are left untouched, matching the GPT-2 / nanoGPT
    convention. Decaying biases makes optimisation noisier for little gain.
    """
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
    )


def seed_everything(seed: int) -> None:
    """Make the run reproducible across RNGs (Python, NumPy, PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {props.total_memory / 1e9:.1f} GB")

    cfg_dict = MODEL_CONFIGS[args.model_size]
    ctx = cfg_dict["context_length"]

    # Token cache produced by data.prep.
    token_cache = Path(args.token_cache)
    if not token_cache.exists():
        raise FileNotFoundError(
            f"Token cache not found: {token_cache}\nRun `python -m nepali_gpt2 data-prep` first."
        )
    # Open the cache as a read-only memmap: we never need the whole array
    # in RAM, and NumPy slices double as windows for the DataLoader.
    arr = np.memmap(token_cache, dtype=np.int32, mode="r")
    print(f"Token array: {len(arr):,} tokens")

    # Hold-out 5% of the corpus for validation. Training only ever sees the
    # first 95%; the tail is reserved for the val-loss / perplexity numbers.
    split = int(0.95 * len(arr))
    train_loader = DataLoader(
        TokenDataset(arr[:split], ctx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TokenDataset(arr[split:], ctx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    print(
        f"Train batches : {len(train_loader):,}  |  Val batches : {len(val_loader):,}"
    )

    model = NepaliGPT(cfg_dict).to(device)
    print(f"Parameters    : {model.num_params() / 1e6:.2f} M")

    # torch.compile gives a free speed-up on modern Triton-equipped GPUs.
    # Guarded so CPU-only machines (where it occasionally regresses) skip it.
    if hasattr(torch, "compile") and torch.cuda.is_available():
        model = torch.compile(model)
        print("torch.compile applied")

    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    lr_fn = make_lr_fn(args.lr, args.warmup_steps, args.max_steps, args.min_lr_ratio)
    # FP16 autocast + gradient scaling on CUDA. `enabled=False` on CPU makes
    # the whole AMP dance a no-op there, so one code path serves both.
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True)

    def save_checkpoint(path: Path, step: int, val_loss: float) -> None:
        # The config dict is saved alongside weights so checkpoints remain
        # loadable without matching the source tag (see generate.load_*).
        torch.save(
            {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
                "cfg": cfg_dict,
            },
            path,
        )

    train_losses, val_losses, steps_logged = [], [], []
    global_step, best_val = 0, float("inf")

    model.train()
    t0 = time.time()
    print(
        f"\nTraining up to {args.max_steps:,} steps "
        f"(eval every {args.eval_every}, {args.eval_batches} val batches)\n"
    )

    for epoch in range(1, args.epochs + 1):
        for x, y in train_loader:
            if global_step >= args.max_steps:
                break

            x, y = x.to(device), y.to(device)

            cur_lr = lr_fn(global_step)
            for g in optimizer.param_groups:
                g["lr"] = cur_lr

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                _, loss = model(x, y)

            # GradScaler scales the loss up in FP16 to keep small gradients
            # representable; the backward/lr/update order below is the one
            # PyTorch recommends to avoid losing precision.
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1

            if global_step % args.eval_every == 0 or global_step == 1:
                v_loss = eval_loss(
                    model, val_loader, device, args.eval_batches, use_amp
                )
                t_loss = eval_loss(
                    model, train_loader, device, args.eval_batches, use_amp
                )

                train_losses.append(t_loss)
                val_losses.append(v_loss)
                steps_logged.append(global_step)

                elapsed = (time.time() - t0) / 60
                print(
                    f"Ep {epoch:02d} | Step {global_step:6d} | "
                    f"train={t_loss:.4f} | val={v_loss:.4f} | "
                    f"lr={cur_lr:.2e} | {elapsed:.1f}m"
                )

                if v_loss < best_val:
                    best_val = v_loss
                    save_checkpoint(ckpt_dir / "best.pt", global_step, v_loss)
                    print(f"  best model saved (val={v_loss:.4f})")

            if global_step % args.save_every == 0:
                save_checkpoint(
                    ckpt_dir / f"step_{global_step:06d}.pt", global_step, best_val
                )
                print(f"  [checkpoint saved at step {global_step}]")

        if global_step >= args.max_steps:
            break

    elapsed = (time.time() - t0) / 60
    print(f"\nDone! Best val loss: {best_val:.4f} | Time: {elapsed:.1f}m")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps_logged, train_losses, label="Train", color="steelblue")
    ax.plot(steps_logged, val_losses, label="Val", color="tomato")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("NepaliGPT — Training Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss.png", dpi=120)
    print("Loss curve saved → loss.png")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train NepaliGPT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model-size",
        default=TRAIN_DEFAULTS["model_size"],
        choices=sorted(MODEL_CONFIGS),
        help="architecture preset",
    )
    for key in TRAIN_DEFAULTS:
        if key == "model_size":
            continue
        p.add_argument(
            f"--{key.replace('_', '-')}",
            type=type(TRAIN_DEFAULTS[key]),
            default=TRAIN_DEFAULTS[key],
        )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI wrapper: parse arguments and run the training loop."""
    train(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
