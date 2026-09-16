"""NepaliGPT quickstart — from installation to Nepali generation in ~30 seconds.

    pip install -e .
    python examples/quickstart.py

This loads the base checkpoint from ckpt/ and prints a sample generation.
If no checkpoint exists yet, it prints help text describing how to get one.
"""

from __future__ import annotations

from pathlib import Path

PROMPTS = [
    "नेपाल एक सुन्दर",
    "हाम्रो देशको इतिहास",
    "काठमाडौंमा",
    "हिमालयको फेदीमा",
]


def main() -> None:
    ckpt = Path("ckpt/best.pt")
    tok = Path("tokenizer/nepali_bpe.model")

    if not ckpt.exists():
        print("NepaliGPT checkpoint not found. To generate text:")
        print("  1. Download the base checkpoint from HuggingFace:")
        print("     https://huggingface.co/utsabdahal34/NepaliGPT-base")
        print(
            "  2. Place it at ckpt/best.pt with the tokenizer at tokenizer/nepali_bpe.model"
        )
        print("  3. Re-run this script.\n")
        print("Or retrain from scratch (requires HF + Kaggle credentials):")
        print("  python -m nepali_gpt2 data-prep")
        print("  python -m nepali_gpt2 train")
        return

    if not tok.exists():
        print(f"Tokenizer not found: {tok}")
        return

    from nepali_gpt2 import generate, load_model_and_tokenizer

    model, sp, cfg, device = load_model_and_tokenizer(
        ckpt_path=str(ckpt), tok_path=str(tok)
    )

    for prompt in PROMPTS:
        text = generate(
            model,
            sp,
            cfg,
            device,
            prompt=prompt,
            max_new=60,
            temperature=0.8,
            top_k=50,
            top_p=0.92,
        )
        print(f"\nPrompt : {prompt}")
        print(f"Output : {text}")


if __name__ == "__main__":
    main()
