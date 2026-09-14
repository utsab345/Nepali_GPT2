"""Smoke tests for the NepaliGPT architecture.

Run with:  pytest
Requires ``torch`` (included in requirements.txt).

These are fast unit tests, not full integration tests: they exercise the
forward pass, weight tying, parameter counts and config sanity on a tiny
model so any regression shows up in seconds rather than after a 2-hour
training run.
"""

import torch

from nepali_gpt2.config import MODEL_CONFIGS, TRAIN_DEFAULTS
from nepali_gpt2.model import NepaliGPT


def tiny_cfg() -> dict:
    # Keep dims small (emb_dim=32, ctx=64) so every test runs in well under
    # a second on CPU — speed matters more than realism for smoke tests.
    return dict(
        vocab_size=1_000,
        context_length=64,
        emb_dim=32,
        n_heads=4,
        n_layers=2,
        drop_rate=0.0,
        qkv_bias=False,
    )


def test_forward_shapes() -> None:
    cfg = tiny_cfg()
    model = NepaliGPT(cfg).eval()

    x = torch.randint(0, cfg["vocab_size"], (2, cfg["context_length"]))
    logits, loss = model(x, x)

    assert logits.shape == (2, cfg["context_length"], cfg["vocab_size"])
    assert loss is not None and loss.shape == ()


def test_logits_without_targets() -> None:
    model = NepaliGPT(tiny_cfg()).eval()
    x = torch.randint(0, 1_000, (1, 32))
    logits, loss = model(x)
    assert logits.shape == (1, 32, 1_000)
    assert loss is None


def test_forward_rejects_sequences_longer_than_context() -> None:
    cfg = tiny_cfg()
    model = NepaliGPT(cfg).eval()
    x = torch.randint(0, cfg["vocab_size"], (1, cfg["context_length"] + 1))

    try:
        model(x)
    except ValueError as exc:
        assert "context_length" in str(exc)
    else:
        raise AssertionError("expected overlong input to be rejected")


def test_weight_tying() -> None:
    model = NepaliGPT(tiny_cfg())
    assert model.head.weight is model.tok_emb.weight


def test_num_params() -> None:
    assert NepaliGPT(tiny_cfg()).num_params() > 0


def test_presets_valid() -> None:
    for name, cfg in MODEL_CONFIGS.items():
        model = NepaliGPT(cfg)
        assert model.num_params() > 0, f"empty model for preset {name}"


def test_defaults_have_model_size() -> None:
    assert TRAIN_DEFAULTS["model_size"] in MODEL_CONFIGS
