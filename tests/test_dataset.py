"""Regression tests for dataset evaluation boundaries."""

import math

import numpy as np
import torch

from nepali_gpt2.data.dataset import evaluate_perplexity


class ConstantLossModel(torch.nn.Module):
    def __init__(self, loss: float) -> None:
        super().__init__()
        self.loss = torch.tensor(loss)
        self.calls = 0

    def forward(self, _x, _y):
        self.calls += 1
        return None, self.loss


def test_negative_max_batches_evaluates_the_full_validation_loader(tmp_path) -> None:
    path = tmp_path / "tokens.npy"
    np.asarray(np.arange(120, dtype=np.int32)).tofile(path)
    model = ConstantLossModel(2.0)

    result = evaluate_perplexity(
        model,
        torch.device("cpu"),
        token_cache=str(path),
        ctx=4,
        batch_size=8,
        max_batches=-1,
        use_amp=False,
    )

    assert model.calls == 1
    assert math.isclose(result, math.exp(2.0))
