"""Convert learned-position NepaliGPT weights to standard Transformers GPT-2."""

from __future__ import annotations

import torch


def to_transformers(model, bos_id, eos_id, pad_id):
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = model.cfg
    if cfg.get("position_encoding", "learned") != "learned":
        raise ValueError("Standard GPT-2 export does not support RoPE")
    config = GPT2Config(
        vocab_size=cfg["vocab_size"],
        n_positions=cfg["context_length"],
        n_embd=cfg["emb_dim"],
        n_layer=cfg["n_layers"],
        n_head=cfg["n_heads"],
        activation_function="gelu",
        resid_pdrop=cfg["drop_rate"],
        embd_pdrop=cfg["drop_rate"],
        attn_pdrop=cfg["drop_rate"],
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        use_cache=True,
    )
    converted = GPT2LMHeadModel(config).eval()
    with torch.no_grad():
        converted.transformer.wte.weight.copy_(model.tok_emb.weight)
        converted.transformer.wpe.weight.copy_(model.pos_emb.weight)
        converted.transformer.ln_f.load_state_dict(model.ln_f.state_dict())
        for source, target in zip(model.blocks, converted.transformer.h):
            target.ln_1.load_state_dict(source.ln1.state_dict())
            target.ln_2.load_state_dict(source.ln2.state_dict())
            target.attn.c_attn.weight.copy_(
                torch.cat(
                    [
                        source.attn.wq.weight,
                        source.attn.wk.weight,
                        source.attn.wv.weight,
                    ],
                    dim=0,
                ).T
            )
            target.attn.c_attn.bias.zero_()
            if cfg["qkv_bias"]:
                target.attn.c_attn.bias.copy_(
                    torch.cat(
                        [source.attn.wq.bias, source.attn.wk.bias, source.attn.wv.bias]
                    )
                )
            for linear, conv in (
                (source.attn.proj, target.attn.c_proj),
                (source.ff.net[0], target.mlp.c_fc),
                (source.ff.net[2], target.mlp.c_proj),
            ):
                conv.weight.copy_(linear.weight.T)
                conv.bias.copy_(linear.bias)
    return converted
