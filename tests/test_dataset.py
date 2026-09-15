import math

import numpy as np
import pytest
import torch

from nepali_gpt2.data.dataset import TokenDataset, evaluate_perplexity


class UniformModel(torch.nn.Module):
    def forward(self, x, y):
        return None, torch.tensor(math.log(10.0))


def test_perplexity_unlimited_restores_eval_mode(tmp_path):
    cache = tmp_path / "tokens.npy"
    np.ones(400, dtype=np.int32).tofile(cache)
    model = UniformModel().eval()
    value = evaluate_perplexity(
        model, torch.device("cpu"), str(cache), ctx=4, batch_size=3, max_batches=-1
    )
    assert value == pytest.approx(10.0)
    assert not model.training


def test_empty_validation_rejected(tmp_path):
    cache = tmp_path / "tokens.npy"
    np.ones(20, dtype=np.int32).tofile(cache)
    assert len(TokenDataset(np.ones(2), 4)) == 0
    with pytest.raises(ValueError, match="no target tokens"):
        evaluate_perplexity(UniformModel(), torch.device("cpu"), str(cache), ctx=4)
