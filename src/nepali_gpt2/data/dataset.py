"""Dataset, evaluation and perplexity helpers shared by train and generate.

Keeping these in one module avoids duplicating the token-array handling
between training (which needs loss snapshots) and inference (which needs
perplexity). The held-out split (5% of the corpus) is defined here too so
train.py and generate.py always evaluate on the same slice of data.
"""

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

    Note: the token cache was written in document order (bos..eos per
    line), so consecutive windows can span a document boundary. This is
    acceptable — the boundary tokens simply become predict-one-another
    examples, just as sentence boundaries do within a document.
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

    Averaged over ``max_batches`` (or the whole loader, whichever is
    smaller) so evaluations stay bounded even on very long runs.

    The model is switched back to ``train()`` mode on return, so callers
    can evaluate mid-loop without restoring training mode themselves.
    """
    model.eval()
    total = count = 0
    for i, (x, y) in enumerate(loader):
        if max_batches >= 0 and i >= max_batches:
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
    """Mean perplexity of ``model`` over the held-out 5% validation split.

    Uses the *same* `0.95 / 0.05` split as ``train.py`` so the reported
    number is comparable across runs and to the validation loss curve.
    Recomputes exp(mean loss) over at most ``max_batches`` for speed;
    set ``max_batches`` higher (or ``-1``) for full-corpus numbers.
    """
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
        if max_batches >= 0 and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=(use_amp and torch.cuda.is_available())):
            _, loss = model(x, y)
        total += loss.item()
        count += 1

    return math.exp(total / max(count, 1))
