"""Inference utilities and CLI for NepaliGPT.

Three modes are provided:

* ``generate`` — autoregressive text completion
* ``next_words`` — top-N next-token probabilities for a prompt
* ``eval`` — mean perplexity on the held-out validation split

Both generate modes are wrapped in ``@torch.no_grad()``: decoding is pure
inference, and disabling gradient tracking also reduces memory so longer
generations fit without CUDA OOM.
"""

from __future__ import annotations

import argparse
import sys

import sentencepiece as spm
import torch

from nepali_gpt2.data.dataset import evaluate_perplexity
from nepali_gpt2.model import NepaliGPT

DEFAULT_PROMPT = "नेपाल एक सुन्दर"


def load_model_and_tokenizer(
    ckpt_path: str = "ckpt/best.pt",
    tok_path: str = "tokenizer/nepali_bpe.model",
    device: str | None = None,
) -> tuple[NepaliGPT, spm.SentencePieceProcessor, dict, torch.device]:
    """Load a trained checkpoint and its SentencePiece tokenizer.

    The checkpoint is expected to be one saved by ``train.py``: it carries
    a ``cfg`` key so the model is reconstructed from the exact architecture
    that created it (not the current code presets).

    Returns:
        model (eval mode), tokenizer, model config dict, and device.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # map_location lets a CUDA-trained checkpoint load onto CPU and vice
    # versa, so serve/inference scripts can share the same weights.
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]

    model = NepaliGPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sp = spm.SentencePieceProcessor()
    if not sp.load(tok_path):
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")

    print(f"Loaded checkpoint: step={ckpt['step']}, val_loss={ckpt['val_loss']:.4f}")
    return model, sp, cfg, device


@torch.no_grad()
def generate(
    model,
    sp,
    cfg: dict,
    device,
    prompt: str = DEFAULT_PROMPT,
    max_new: int = 80,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.92,
) -> str:
    """Autoregressively complete ``prompt`` with top-k / top-p sampling.

    Decoding runs token-by-token: at each step we feed the last ``ctx``
    tokens through the model, sample one token from the (truncated) next-
    token distribution, append it, and repeat until ``max_new`` tokens or an
    EOS token. The prompt is never included in the output.
    """
    bos_id, eos_id = sp.bos_id(), sp.eos_id()
    ctx = cfg["context_length"]

    ids = torch.tensor(
        [[bos_id] + sp.encode(prompt, out_type=int)],
        dtype=torch.long,
        device=device,
    )

    for _ in range(max_new):
        # Slice the window to `ctx` so prompts longer than the context still
        # fit through the (fixed-size) positional embedding table.
        logits, _ = model(ids[:, -ctx:])
        logits = logits[:, -1, :] / temperature

        # Top-k: hard-truncate the distribution to the k most likely tokens.
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        # Top-p (nucleus): keep the smallest set whose cumulative mass
        # reaches p — a tighter, data-adaptive budget than fixed top-k.
        if top_p < 1.0:
            sorted_logits, sort_idx = torch.sort(logits, descending=True)
            cum = torch.cumsum(torch.softmax(sorted_logits, -1), -1)
            sorted_logits[cum - torch.softmax(sorted_logits, -1) > top_p] = float(
                "-inf"
            )
            logits = torch.zeros_like(logits).scatter_(1, sort_idx, sorted_logits)

        next_id = torch.multinomial(torch.softmax(logits, -1), 1)
        if next_id.item() == eos_id:
            break
        ids = torch.cat([ids, next_id], dim=1)

    return sp.decode(ids[0, 1:].tolist())


@torch.no_grad()
def next_words(
    model,
    sp,
    cfg: dict,
    device,
    prompt: str,
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """Return the top-``n`` most probable next (word, probability) pairs.

    Unlike the sampler, this is a single forward pass (no decoding loop):
    it takes one last-token softmax and reports the highest-probability
    continuations, which is what a keyboard/completion UI wants.
    """
    bos_id = sp.bos_id()
    ctx = cfg["context_length"]

    ids = torch.tensor(
        [[bos_id] + sp.encode(prompt, out_type=int)],
        dtype=torch.long,
        device=device,
    )[:, -ctx:]

    logits, _ = model(ids)
    probs = torch.softmax(logits[0, -1], dim=-1)
    top_probs, top_ids = torch.topk(probs, top_n)

    return [
        # SentencePiece marks word starts with "▁"; drop it and trim so the
        # printed word reads like normal Nepali text.
        (sp.id_to_piece(i.item()).replace("▁", " ").strip(), p.item())
        for i, p in zip(top_ids, top_probs)
    ]


def run(args: argparse.Namespace) -> None:
    model, sp, cfg, device = load_model_and_tokenizer(args.ckpt, args.tok, args.device)

    if args.mode == "generate":
        out = generate(
            model,
            sp,
            cfg,
            device,
            prompt=args.prompt,
            max_new=args.max_new,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(f"\nPrompt   : {args.prompt}")
        print(f"Generated: {out}")

    elif args.mode == "next_words":
        preds = next_words(model, sp, cfg, device, args.prompt, top_n=args.top_n)
        print(f'\n"{args.prompt}" →')
        for word, prob in preds:
            bar = "|" * int(prob * 50)
            print(f"  {word:<15} {prob:.3f}  {bar}")

    elif args.mode == "eval":
        ppl = evaluate_perplexity(model, device)
        print(f"\nPerplexity (val): {ppl:.2f}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NepaliGPT inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument(
        "--mode", default="generate", choices=["generate", "next_words", "eval"]
    )
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.92)
    p.add_argument(
        "--top-n", type=int, default=10, help="Number of next-word predictions to show"
    )
    p.add_argument("--ckpt", default="ckpt/best.pt")
    p.add_argument("--tok", default="tokenizer/nepali_bpe.model")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI wrapper: parse arguments and dispatch the requested mode."""
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
