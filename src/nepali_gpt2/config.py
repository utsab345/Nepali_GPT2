"""Centralised configuration: model sizes and training hyper-parameters.

Keep all tunableconstants here rather than scattering magic numbers through
the scripts. A single change in this file (e.g. ``drop_rate`` for better
regularisation) propagates everywhere without touching training or
generation code. The same config dict is embedded in every checkpoint
under ``ckpt["cfg"]``, so checkpoints are self-describing and can be
reloaded without matching a code version.
"""

from __future__ import annotations

from typing import Any

#: Architecture presets, keyed by size. Each value is a valid ``NepaliGPT``
#: config dict and is also embedded into checkpoints under ``ckpt["cfg"]``.
#:
#: The three presets scale depth and width together (6/8/12 layers paired
#: with 384/512/768 embedding dims) so that training cost grows roughly
#: linearly with model size while keeping a 16k vocab fixed.
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "small": dict(
        vocab_size=16000,
        context_length=512,
        emb_dim=384,
        n_heads=6,
        n_layers=6,
        drop_rate=0.1,
        qkv_bias=False,
    ),
    "base": dict(
        vocab_size=16000,
        context_length=512,
        emb_dim=512,
        n_heads=8,
        n_layers=8,
        drop_rate=0.1,
        qkv_bias=False,
    ),
    "large": dict(
        vocab_size=16000,
        context_length=512,
        emb_dim=768,
        n_heads=12,
        n_layers=12,
        drop_rate=0.1,
        qkv_bias=False,
    ),
}

#: Default CLI options shared by the training entry point. Each key maps to
#: a ``--kebab-case`` argument; values double as the argparse defaults.
TRAIN_DEFAULTS: dict[str, Any] = dict(
    token_cache="data/tokens.npy",
    ckpt_dir="ckpt",
    model_size="base",
    epochs=10,  # upper bound on passes over the data
    max_steps=15_000,  # hard stop — the effective training budget
    batch_size=32,
    lr=5e-4,  # peak LR reached at the end of warm-up
    min_lr_ratio=0.1,  # cosine decay floor (10% of peak LR)
    weight_decay=0.1,  # applied to 2-D params only (see train.build_optimizer)
    grad_clip=1.0,
    warmup_steps=500,
    eval_every=500,  # steps between validation-loss snapshots
    eval_batches=100,  # batches averaged per evaluation
    save_every=5_000,  # periodic crash-recovery checkpoint cadence
    seed=42,
)
