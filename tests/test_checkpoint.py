from collections import OrderedDict

import pytest
import torch

from nepali_gpt2.checkpoint import normalize_state_dict
from nepali_gpt2.model import NepaliGPT
from test_model import tiny_cfg


def legacy_state(model):
    state = OrderedDict()
    for name, value in model.state_dict().items():
        for old, new in (("wq", "Wq"), ("wk", "Wk"), ("wv", "Wv")):
            name = name.replace(f".attn.{old}.", f".attn.{new}.")
        state["_orig_mod." + name] = value
    for i, block in enumerate(model.blocks):
        state[f"_orig_mod.blocks.{i}.attn.mask"] = block.attn.mask.clone()
    return state


def test_original_colab_checkpoint_preserves_logits():
    cfg = tiny_cfg()
    original = NepaliGPT(cfg).eval()
    restored = NepaliGPT(cfg).eval()
    restored.load_state_dict(normalize_state_dict(legacy_state(original), cfg))
    ids = torch.randint(1, cfg["vocab_size"], (1, 8))
    with torch.no_grad():
        torch.testing.assert_close(original(ids)[0], restored(ids)[0])


def test_rejects_noncausal_saved_mask():
    cfg = tiny_cfg()
    state = legacy_state(NepaliGPT(cfg))
    state["_orig_mod.blocks.0.attn.mask"].zero_()
    with pytest.raises(ValueError, match="Unexpected causal mask"):
        normalize_state_dict(state, cfg)


def test_rejects_colliding_legacy_names():
    cfg = tiny_cfg()
    state = legacy_state(NepaliGPT(cfg))
    state["_orig_mod.blocks.0.attn.wq.weight"] = state[
        "_orig_mod.blocks.0.attn.Wq.weight"
    ]
    with pytest.raises(ValueError, match="Conflicting checkpoint"):
        normalize_state_dict(state, cfg)
