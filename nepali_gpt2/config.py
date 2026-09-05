"""Centralised configuration: model sizes and training hyper-parameters.

All hard-coded defaults live here so the scripts stay thin and a single
setting can be changed without touching training or generation code.
"""

from __future__ import annotations

from typing import Any, Dict

#: Architecture presets, keyed by size. Each value is a valid ``NepaliGPT``
#: config dict and is also embedded into checkpoints under ``ckpt["cfg"]``.
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
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

#: Default CLI options shared by the training entry point.
TRAIN_DEFAULTS: Dict[str, Any] = dict(
    token_cache="data/tokens.npy",
    ckpt_dir="ckpt",
    model_size="base",
    epochs=10,
    max_steps=15_000,
    batch_size=32,
    lr=5e-4,
    min_lr_ratio=0.1,
    weight_decay=0.1,
    grad_clip=1.0,
    warmup_steps=500,
    eval_every=500,
    eval_batches=100,
    save_every=5_000,
    seed=42,
)