
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, ctx_len, dropout, n_heads, qkv_bias=False):
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"
        self.n_heads  = n_heads
        self.d_out    = d_out
        self.head_dim = d_out // n_heads
        self.Wq = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wk = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wv = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.proj   = nn.Linear(d_out, d_out)
        self.drop   = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(ctx_len, ctx_len), diagonal=1)
        )

    def forward(self, x):
        B, T, _ = x.shape
        q = self.Wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.mask[:T, :T].bool(), float("-inf"))
        att = self.drop(torch.softmax(att, dim=-1))
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, self.d_out)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            nn.Dropout(cfg["drop_rate"]),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            cfg["emb_dim"], cfg["emb_dim"],
            cfg["context_length"], cfg["drop_rate"],
            cfg["n_heads"], cfg["qkv_bias"]
        )
        self.ff   = FeedForward(cfg)
        self.ln1  = nn.LayerNorm(cfg["emb_dim"])
        self.ln2  = nn.LayerNorm(cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class NepaliGPT(nn.Module):


    CONFIGS = {
        "small": dict(
            vocab_size=16000, context_length=512,
            emb_dim=384, n_heads=6, n_layers=6,
            drop_rate=0.1, qkv_bias=False,
        ),
        "base": dict(
            vocab_size=16000, context_length=512,
            emb_dim=512, n_heads=8, n_layers=8,
            drop_rate=0.1, qkv_bias=False,
        ),
        "large": dict(
            vocab_size=16000, context_length=512,
            emb_dim=768, n_heads=12, n_layers=12,
            drop_rate=0.1, qkv_bias=False,
        ),
    }

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg     = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop    = nn.Dropout(cfg["drop_rate"])
        self.blocks  = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.ln_f    = nn.LayerNorm(cfg["emb_dim"])
        self.head    = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Args:
            idx     : (B, T) integer token ids
            targets : (B, T) integer token ids for loss computation (optional)
        Returns:
            logits  : (B, T, vocab_size)
            loss    : scalar cross-entropy loss (or None if targets not given)
        """
        B, T = idx.shape
        pos    = torch.arange(T, device=idx.device)
        x      = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x      = self.blocks(x)
        x      = self.ln_f(x)
        logits = self.head(x)
        loss   = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,   
            )
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
