"""Model-level behaviour tests: causal masking, generation determinism,
next-word ranking and a smoke training step (roadmap issue #15).

These exercise the real model + decoding loop with a tiny in-memory
``SentencePiece``-like double, so nothing depends on a checkpoint or a
trained tokenizer file. Pure CPU, runs in well under a second.
"""

from __future__ import annotations

import torch

from nepali_gpt2.generate import generate, next_words
from nepali_gpt2.model import NepaliGPT


class FakeSP:
    """Minimal stand-in for ``sentencepiece.SentencePieceProcessor``.

    Maps characters to ids (bos=1, eos=2, pad=0) so the decode loop and
    ``next_words`` can be exercised without a trained ``.model`` file.
    """

    def __init__(self) -> None:
        self._c2i: dict[str, int] = {}
        self._i2c: dict[int, str] = {}

    def bos_id(self) -> int:
        return 1

    def eos_id(self) -> int:
        return 2

    def encode(self, text: str, out_type: type = int) -> list[int]:
        ids = []
        for ch in text:
            if ch not in self._c2i:
                nxt = len(self._c2i) + 3
                self._c2i[ch] = nxt
                self._i2c[nxt] = ch
            ids.append(self._c2i[ch])
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self._i2c.get(int(i), chr(0xE000 + int(i))) for i in ids)

    def id_to_piece(self, i: int) -> str:
        return self._i2c.get(int(i), "▁unk")


def tiny_cfg() -> dict:
    return dict(
        vocab_size=1_000,
        context_length=64,
        emb_dim=32,
        n_heads=4,
        n_layers=2,
        drop_rate=0.0,
        qkv_bias=False,
    )


def test_causal_mask_ignores_future_tokens() -> None:
    cfg = tiny_cfg()
    model = NepaliGPT(cfg).eval()
    first = 16
    x1 = torch.randint(1, cfg["vocab_size"], (1, 32))
    x2 = x1.clone()
    x2[0, first:] = torch.randint(1, cfg["vocab_size"], (1, 32 - first))

    with torch.no_grad():
        logits1, _ = model(x1)
        logits2, _ = model(x2)

    # Positions <= first only see identical prefixes, so their logits must match.
    assert torch.allclose(logits1[0, :first], logits2[0, :first], atol=1e-5)
    assert not torch.allclose(logits1[0, first:], logits2[0, first:], atol=1e-5)


def test_generate_deterministic_with_fixed_seed() -> None:
    torch.manual_seed(123)
    cfg = tiny_cfg()
    model = NepaliGPT(cfg).eval()
    sp = FakeSP()

    torch.manual_seed(7)
    out_a = generate(model, sp, cfg, torch.device("cpu"), prompt="नेपाल", max_new=20)
    torch.manual_seed(7)
    out_b = generate(model, sp, cfg, torch.device("cpu"), prompt="नेपाल", max_new=20)

    assert out_a == out_b
    assert out_a  # non-empty


def test_generate_respects_max_new_bound() -> None:
    torch.manual_seed(123)
    cfg = tiny_cfg()
    model = NepaliGPT(cfg).eval()
    sp = FakeSP()

    out = generate(
        model,
        sp,
        cfg,
        torch.device("cpu"),
        prompt="नेपाल",
        max_new=16,
        temperature=1e-6,
        top_k=1,
    )
    assert 0 < len(sp.encode(out, out_type=int)) <= 16


def test_generate_rejects_invalid_sampling_settings() -> None:
    cfg = tiny_cfg()
    model = NepaliGPT(cfg).eval()
    sp = FakeSP()

    for kwargs in ({"temperature": 0}, {"top_p": 0}, {"top_p": 1.1}, {"max_new": -1}):
        try:
            generate(model, sp, cfg, torch.device("cpu"), **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_next_words_returns_sorted_probabilities() -> None:
    torch.manual_seed(123)
    cfg = tiny_cfg()
    model = NepaliGPT(cfg).eval()
    sp = FakeSP()

    preds = next_words(model, sp, cfg, torch.device("cpu"), prompt="ने", top_n=5)
    assert len(preds) == 5
    tokens, probs = zip(*preds)
    assert all(isinstance(t, str) for t in tokens)
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert list(probs) == sorted(probs, reverse=True)


def test_next_words_rejects_non_positive_top_n() -> None:
    cfg = tiny_cfg()
    model = NepaliGPT(cfg).eval()
    sp = FakeSP()

    for top_n in (0, -1):
        try:
            next_words(model, sp, cfg, torch.device("cpu"), prompt="ने", top_n=top_n)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for top_n={top_n}")


def test_smoke_train_step_reduces_loss() -> None:
    cfg = tiny_cfg()
    model = NepaliGPT(cfg).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randint(1, cfg["vocab_size"], (2, 24))
    losses = []
    for _ in range(4):
        opt.zero_grad()
        _, loss = model(x, x)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(value)).item() for value in losses)
    assert losses[-1] < losses[0]
