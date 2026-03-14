import math
import argparse
import torch
import numpy as np
from pathlib import Path

from nepali_gpt2.model import NepaliGPT



def load_model_and_tokenizer(
    ckpt_path: str = "ckpt/best.pt",
    tok_path:  str = "tokenizer/nepali_bpe.model",
    device:    str = None,
):
    """
    Load a trained NepaliGPT checkpoint and its SentencePiece tokenizer.

    Returns:
        model  : NepaliGPT (eval mode)
        sp     : SentencePieceProcessor
        cfg    : model config dict
        device : torch.device
    """
    import sentencepiece as spm

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt["cfg"]

    model = NepaliGPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sp = spm.SentencePieceProcessor()
    sp.load(tok_path)

    print(f"Loaded checkpoint: step={ckpt['step']}, val_loss={ckpt['val_loss']:.4f}")
    return model, sp, cfg, device



@torch.no_grad()
def generate(
    model,
    sp,
    cfg: dict,
    device,
    prompt:      str   = "नेपाल",
    max_new:     int   = 80,
    temperature: float = 0.8,
    top_k:       int   = 50,
    top_p:       float = 0.92,
) -> str:

    BOS_ID = sp.bos_id()
    EOS_ID = sp.eos_id()
    CTX    = cfg["context_length"]

    ids = torch.tensor(
        [[BOS_ID] + sp.encode(prompt, out_type=int)],
        dtype=torch.long, device=device,
    )

    for _ in range(max_new):
        ids_in = ids[:, -CTX:]
        logits, _ = model(ids_in)
        logits = logits[:, -1, :] / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        if top_p < 1.0:
            sorted_logits, sort_idx = torch.sort(logits, descending=True)
            cum = torch.cumsum(torch.softmax(sorted_logits, -1), -1)
            sorted_logits[cum - torch.softmax(sorted_logits, -1) > top_p] = float("-inf")
            logits = torch.zeros_like(logits).scatter_(1, sort_idx, sorted_logits)

        next_id = torch.multinomial(torch.softmax(logits, -1), 1)
        if next_id.item() == EOS_ID:
            break
        ids = torch.cat([ids, next_id], dim=1)

    generated = ids[0, 1:].tolist()   
    return sp.decode(generated)


#  Next-word prediction 

@torch.no_grad()
def next_words(model, sp, cfg: dict, device, prompt: str, top_n: int = 10):
    """
    Return the top-n most probable next tokens after `prompt`.

    Returns:
        List of (word, probability) tuples sorted by probability (desc).
    """
    BOS_ID = sp.bos_id()
    CTX    = cfg["context_length"]

    ids = torch.tensor(
        [[BOS_ID] + sp.encode(prompt, out_type=int)],
        dtype=torch.long, device=device,
    )
    ids = ids[:, -CTX:]
    logits, _ = model(ids)
    probs = torch.softmax(logits[0, -1], dim=-1)
    top_p, top_i = torch.topk(probs, top_n)
    return [
        (sp.id_to_piece(i.item()).replace("▁", " ").strip(), p.item())
        for i, p in zip(top_i, top_p)
    ]



@torch.no_grad()
def evaluate_perplexity(
    model,
    device,
    token_cache: str = "data/tokens.npy",
    ctx:         int = 512,
    batch_size:  int = 32,
    max_batches: int = 200,
    use_amp:     bool = True,
) -> float:
    """Compute perplexity on the held-out 5% validation split."""
    from torch.utils.data import DataLoader
    from nepali_gpt2.train import TokenDataset

    arr   = np.memmap(token_cache, dtype=np.int32, mode="r")
    split = int(0.95 * len(arr))
    ds    = TokenDataset(arr[split:], ctx)
    dl    = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model.eval()
    total = count = 0
    for i, (x, y) in enumerate(dl):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=(use_amp and torch.cuda.is_available())):
            _, loss = model(x, y)
        total += loss.item()
        count += 1

    avg_loss   = total / max(count, 1)
    perplexity = math.exp(avg_loss)
    return perplexity



def main():
    p = argparse.ArgumentParser(description="NepaliGPT inference")
    p.add_argument("--prompt",      default="नेपाल एक सुन्दर")
    p.add_argument("--mode",        default="generate",
                   choices=["generate", "next_words", "eval"])
    p.add_argument("--max_new",     type=int,   default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k",       type=int,   default=50)
    p.add_argument("--top_p",       type=float, default=0.92)
    p.add_argument("--top_n",       type=int,   default=10,
                   help="Number of next-word predictions to show")
    p.add_argument("--ckpt",   default="ckpt/best.pt")
    p.add_argument("--tok",    default="tokenizer/nepali_bpe.model")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    model, sp, cfg, device = load_model_and_tokenizer(args.ckpt, args.tok, args.device)

    if args.mode == "generate":
        out = generate(model, sp, cfg, device,
                       prompt=args.prompt, max_new=args.max_new,
                       temperature=args.temperature,
                       top_k=args.top_k, top_p=args.top_p)
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


if __name__ == "__main__":
    main()
