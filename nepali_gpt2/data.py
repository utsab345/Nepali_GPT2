"""Dataset, evaluation and perplexity helpers shared by train and generate."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TokenDataset(Dataset):
    """Chunked causal-language-model dataset over cached token ids.

    Each item maps the window ``X[i : i + ctx]`` to its shifted target
    ``X[i + 1 : i + ctx + 1]``, i.e. every position predicts the next token.
    """

    def __init__(self, data: np.ndarray, ctx: int) -> None:
        self.data = data
        self.ctx = ctx

    def __len__(self) -> int:
        return len(self.data) - self.ctx

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.data[i: i + self.ctx].astype(np.int64))
        y = torch.from_numpy(self.data[i + 1: i + self.ctx + 1].astype(np.int64))
        return x, y


@torch.no_grad()
def eval_loss(
    model,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    use_amp: bool,
) -> float:
    """Mean cross-entropy loss over a subset of ``loader``.

    The model is switched back to ``train()`` mode on return.
    """
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


@torch.no_grad()
def evaluate_perplexity(
    model,
    device: torch.device,
    token_cache: str = "data/tokens.npy",
    ctx: int = 512,
    batch_size: int = 32,
    max_batches: int = 200,
    use_amp: bool = True,
) -> float:
    """Mean perplexity of ``model`` over the held-out 5% validation split."""
    cache = Path(token_cache)
    if not cache.exists():
        raise FileNotFoundError(f"Token cache not found: {cache}")

    arr = np.memmap(cache, dtype=np.int32, mode="r")
    split = int(0.95 * len(arr))
    ds = TokenDataset(arr[split:], ctx)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model.eval()
    total = count = 0
    for i, (x, y) in enumerate(dl):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=(use_amp and torch.cuda.is_available())):
            _, loss = model(x, y)
        total += loss.item()
        count += 1

    return math.exp(total / max(count, 1))