import pytest
import torch

from nepali_gpt2.model import NepaliGPT
from nepali_gpt2.quantization import Int4Linear, convert
from test_model import tiny_cfg


def test_int4_odd_width_error_bound():
    linear = torch.nn.Linear(7, 5).eval()
    quantized = Int4Linear(linear)
    x = torch.eye(7)
    error = (linear(x) - quantized(x)).abs()
    assert torch.all(error <= quantized.scale.T / 2 + 1e-6)
    assert quantized.packed.numel() == 20


@pytest.mark.parametrize("precision", ["fp32", "fp16", "int8", "int4"])
def test_precision_state_roundtrip(precision):
    cfg = tiny_cfg()
    model = convert(NepaliGPT(cfg), precision)
    other = convert(NepaliGPT(cfg), precision)
    other.load_state_dict(model.state_dict())
    ids = torch.ones((1, 4), dtype=torch.long)
    with torch.no_grad():
        a, _ = model(ids)
        b, _ = other(ids)
    assert torch.isfinite(a).all()
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize("precision", ["fp32", "fp16", "int8", "int4"])
def test_checkpoint_loader_preserves_quantization_metadata(
    precision, tokenizer, tmp_path
):
    from nepali_gpt2.generate import load_model_and_tokenizer

    cfg = dict(tiny_cfg(), vocab_size=tokenizer.get_piece_size())
    source = convert(NepaliGPT(cfg), precision)
    checkpoint = tmp_path / "model.pt"
    token_path = tmp_path / "tokenizer.model"
    token_path.write_bytes(tokenizer.serialized_model_proto())
    torch.save(
        dict(cfg=cfg, model=source.state_dict(), precision=precision), checkpoint
    )
    restored, _, _, _ = load_model_and_tokenizer(
        str(checkpoint), str(token_path), "cpu"
    )
    ids = torch.ones((1, 4), dtype=torch.long)
    with torch.no_grad():
        torch.testing.assert_close(source(ids)[0], restored(ids)[0])
