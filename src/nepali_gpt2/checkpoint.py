"""Compatibility for the original Colab checkpoint state dictionaries."""

from __future__ import annotations

import torch


def normalize_state_dict(state, cfg):
    """Update known legacy names in place, preserving quantization metadata.

    Only validated deterministic masks are omitted. Unknown weights still fail
    strict model loading; ambiguous old/new aliases are rejected.
    """
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state, "_orig_mod.")
    for index in range(cfg["n_layers"]):
        prefix = f"blocks.{index}.attn."
        mask_key = prefix + "mask"
        if mask_key in state:
            mask = state[mask_key]
            ctx = cfg["context_length"]
            expected = torch.triu(torch.ones(ctx, ctx, device=mask.device), diagonal=1)
            if mask.shape != expected.shape or not torch.equal(mask, expected):
                raise ValueError(f"Unexpected causal mask in checkpoint: {mask_key}")
            del state[mask_key]
        for old, new in (("Wq", "wq"), ("Wk", "wk"), ("Wv", "wv")):
            for suffix in ("weight", "bias"):
                old_key, new_key = (
                    prefix + old + "." + suffix,
                    prefix + new + "." + suffix,
                )
                if old_key in state:
                    if new_key in state:
                        raise ValueError(
                            f"Conflicting checkpoint weights: {old_key}, {new_key}"
                        )
                    state[new_key] = state.pop(old_key)
    return state
