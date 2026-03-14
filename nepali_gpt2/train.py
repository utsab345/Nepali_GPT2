import os
import math
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from nepali_gpt2.model import NepaliGPT



DEFAULTS = dict(
    token_cache  = "data/tokens.npy",
    ckpt_dir     = "ckpt",
    model_size   = "base",        
    epochs       = 10,
    max_steps    = 15_000,
    batch_size   = 32,
    lr           = 5e-4,
    min_lr_ratio = 0.1,            
    weight_decay = 0.1,
    grad_clip    = 1.0,
    warmup_steps = 500,
    eval_every   = 500,
    eval_batches = 100,
    save_every   = 5_000,
    seed         = 42,
)



class TokenDataset(Dataset):
    def __init__(self, data: np.ndarray, ctx: int):
        self.data = data
        self.ctx  = ctx

    def __len__(self):
        return len(self.data) - self.ctx

    def __getitem__(self, i):
        x = torch.from_numpy(self.data[i     : i + self.ctx    ].astype(np.int64))
        y = torch.from_numpy(self.data[i + 1 : i + self.ctx + 1].astype(np.int64))
        return x, y



def make_lr_fn(lr, warmup, total_steps, min_lr_ratio):
    def lr_schedule(step):
        if step < warmup:
            return lr * (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        cosine = 0.5 * (1 + math.cos(math.pi * t))
        return lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine)
    return lr_schedule



@torch.no_grad()
def eval_loss(model, loader, device, max_batches, use_amp):
    model.eval()
    total = count = 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            _, loss = model(x, y)
        total += loss.item()
        count += 1
    model.train()
    return total / max(count, 1)



def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load("tokenizer/nepali_bpe.model")
    PAD_ID = sp.pad_id()  # 0

    token_cache = Path(args.token_cache)
    assert token_cache.exists(), f"Token cache not found: {token_cache}\nRun data_prep.py first."
    arr = np.memmap(token_cache, dtype=np.int32, mode="r")
    print(f"Token array: {len(arr):,} tokens")

    cfg_dict = NepaliGPT.CONFIGS[args.model_size]
    CTX   = cfg_dict["context_length"]

    split      = int(0.95 * len(arr))
    train_ds   = TokenDataset(arr[:split], CTX)
    val_ds     = TokenDataset(arr[split:], CTX)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True, drop_last=False)

    print(f"Train batches : {len(train_loader):,}  |  Val batches : {len(val_loader):,}")

    model = NepaliGPT(cfg_dict).to(device)
    print(f"Parameters    : {model.num_params() / 1e6:.2f} M")
    print(f"Config        : {cfg_dict}")

    USE_COMPILE = hasattr(torch, "compile") and torch.cuda.is_available()
    if USE_COMPILE:
        model = torch.compile(model)
        print("torch.compile applied ✓")

    decay    = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() <  2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay,    "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    lr_fn    = make_lr_fn(args.lr, args.warmup_steps, args.max_steps, args.min_lr_ratio)
    use_amp  = torch.cuda.is_available()
    scaler   = torch.amp.GradScaler("cuda", enabled=use_amp)


    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True)

    # Training 
    train_losses, val_losses, steps_logged = [], [], []
    global_step = 0
    best_val    = float("inf")

    model.train()
    t0 = time.time()
    print(f"\nTraining up to {args.max_steps:,} steps "
          f"(eval every {args.eval_every}, {args.eval_batches} val batches)\n")

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

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1

            if global_step % args.eval_every == 0 or global_step == 1:
                v_loss = eval_loss(model, val_loader,   device, args.eval_batches, use_amp)
                t_loss = eval_loss(model, train_loader, device, args.eval_batches, use_amp)

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
                    torch.save(
                        {
                            "step": global_step,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "val_loss": v_loss,
                            "cfg": cfg_dict,
                        },
                        ckpt_dir / "best.pt",
                    )
                    print(f"  ★ best model saved (val={v_loss:.4f})")

            if global_step % args.save_every == 0:
                torch.save(
                    {
                        "step": global_step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "val_loss": best_val,
                        "cfg": cfg_dict,
                    },
                    ckpt_dir / f"step_{global_step:06d}.pt",
                )
                print(f"  [checkpoint saved at step {global_step}]")

        if global_step >= args.max_steps:
            break

    elapsed = (time.time() - t0) / 60
    print(f"\nDone! Best val loss: {best_val:.4f} | Time: {elapsed:.1f}m")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps_logged, train_losses, label="Train", color="steelblue")
    ax.plot(steps_logged, val_losses,   label="Val",   color="tomato")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title("NepaliGPT — Training Loss")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss.png", dpi=120)
    print("Loss curve saved → loss.png")



def parse_args():
    p = argparse.ArgumentParser(description="Train NepaliGPT")
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
