"""Decoder-only Transformer (GPT-2 style) architecture for Nepali.

NepaliGPT is a pre-LayerNorm, decoder-only causal language model:

* Token + learned positional embeddings (context length 512)
* Stack of causal multi-head self-attention + GELU feed-forward blocks
* Pre-LayerNorm before attention and MLP, residual adds after dropout
* Weight tying between the input embedding and the output head
* Deterministic Normal(0, 0.02) initialisation
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention with a learned output projection."""

    def __init__(
        self,
        emb_dim: int,
        ctx_len: int,
        dropout: float,
        n_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        assert emb_dim % n_heads == 0, "emb_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // n_heads

        self.wq = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.wk = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.wv = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.drop = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(ctx_len, ctx_len), diagonal=1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.mask[:t, :t].bool(), float("-inf"))
        att = self.drop(torch.softmax(att, dim=-1))

        out = (att @ v).transpose(1, 2).contiguous().view(b, t, self.emb_dim)
        return self.proj(out)


class FeedForward(nn.Module):
    """Position-wise 2-layer MLP with GELU, expanding the width 4x."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            nn.Dropout(cfg["drop_rate"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block: attention then feed-forward."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(
            cfg["emb_dim"], cfg["context_length"],
            cfg["drop_rate"], cfg["n_heads"], cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg["emb_dim"])
        self.ln2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class NepaliGPT(nn.Module):
    """GPT-2 style decoder-only language model.

    Args:
        cfg: dict with keys ``vocab_size``, ``context_length``, ``emb_dim``,
            ``n_heads``, ``n_layers``, ``drop_rate`` and ``qkv_bias``.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])
        self.blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.ln_f = nn.LayerNorm(cfg["emb_dim"])
        self.head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # Weight tying: share input and output embedding projection.
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0.0, 0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the model.

        Args:
            idx: Long tensor of shape ``(B, T)`` with input token ids.
            targets: optional Long tensor of shape ``(B, T)`` used to compute
                cross-entropy loss (``ignore_index`` = pad id, i.e. 0).

        Returns:
            logits of shape ``(B, T, vocab_size)`` and a scalar loss (or
            ``None`` when ``targets`` is ``None``).
        """
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,  # pad id from the SentencePiece tokenizer
            )
        return logits, loss

    def num_params(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)