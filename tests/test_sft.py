import json

import pytest
import torch

from nepali_gpt2.model import NepaliGPT
from nepali_gpt2.sft import InstructionDataset, response_loss
from test_model import tiny_cfg


class Tokenizer:
    def bos_id(self):
        return 1

    def eos_id(self):
        return 2

    def pad_id(self):
        return 0

    def encode(self, text):
        return [3 + ord(c) % 20 for c in text]


def test_response_mask_and_training(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(dict(instruction="नेपाल", output="देश")))
    ds = InstructionDataset(str(path), Tokenizer(), 64)
    x, y = ds[0]
    assert (y != -100).sum() == 4  # three response characters plus EOS
    assert y[y != -100].tolist() == Tokenizer().encode("देश") + [2]
    model = NepaliGPT(tiny_cfg())
    loss = response_loss(model, x[None], y[None])
    assert torch.isfinite(loss)
    loss.backward()
    assert model.head.weight.grad is not None


def test_overlong_example_rejected(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(dict(instruction="नेपाल", output="देश")))
    with pytest.raises(ValueError, match="exceeds"):
        InstructionDataset(str(path), Tokenizer(), 8)
