import pytest
import torch

from nepali_gpt2.hf_export import to_transformers
from nepali_gpt2.model import NepaliGPT
from test_model import tiny_cfg


@pytest.mark.parametrize("bias", [False, True])
def test_transformers_logits_match_native(bias, tmp_path):
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM

    source = NepaliGPT(dict(tiny_cfg(), qkv_bias=bias)).eval()
    target = to_transformers(source, 1, 2, 0)
    ids = torch.randint(1, 1000, (2, 12))
    with torch.no_grad():
        torch.testing.assert_close(
            source(ids)[0], target(ids).logits, atol=1e-5, rtol=1e-5
        )
    target.save_pretrained(tmp_path)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(
            source(ids)[0], loaded(ids).logits, atol=1e-5, rtol=1e-5
        )
