"""NepaliGPT Training Explorer — see how the model learned.

Loads exported training-progress JSON (from export_training_progress.py),
lets the visitor slide across training steps, and shows:
  * prompt -> generation at that step
  * training/validation loss and perplexity
  * top next-token probabilities

Also tokenizer visualizer: enter Nepali text and see SentencePiece token
boundaries, token ids, and the decoded sentence back.

Requires: gradio
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
PROGRESS_FILE = ROOT / "training_progress.json"
TOKENIZER_FILE = ROOT / "nepali_bpe.model"


def _load_progress() -> dict:
    if not PROGRESS_FILE.exists():
        return {
            "prompt": "नेपालको राजधानी",
            "steps": [
                {
                    "checkpoint": "No data",
                    "step": 0,
                    "val_loss": None,
                    "ppl": None,
                    "generation": "Run scripts/export_training_progress.py first.",
                    "next_words": [],
                }
            ],
        }
    return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))


def get_step_view(progress: dict, step_index: int) -> tuple[str, float, str, str]:
    """Return (generation, loss, ppl, next words) for a given step index."""
    steps = progress.get("steps", [])
    if not steps:
        return "No data", 0.0, 0.0, "No data"
    entry = steps[min(step_index, len(steps) - 1)]
    gen = entry.get("generation", "")
    loss = entry.get("val_loss") or 0.0
    ppl = entry.get("ppl") or 0.0
    nw = "\n".join(
        f"{w['word']:<20} {w['prob']:.4f}" for w in entry.get("next_words", [])
    )
    return gen, loss, ppl, nw


def analyze_tokenizer(text: str) -> str:
    """Show SentencePiece tokenization for arbitrary input text."""
    import sentencepiece as spm

    if not TOKENIZER_FILE.exists():
        return "Tokenizer file not found — run data-prep first."
    sp = spm.SentencePieceProcessor()
    if not sp.load(str(TOKENIZER_FILE)):
        return "Failed to load tokenizer."

    pieces = sp.encode(text, out_type=str)
    ids = sp.encode(text, out_type=int)
    tokens = list(zip(pieces, ids))
    table = "\n".join(
        f"{i:>2d}  {p:<24}  {tid:>5d}" for i, (p, tid) in enumerate(tokens)
    )
    decoded = sp.decode(ids)
    n_tokens = len(tokens)
    return (
        f"**Sentence**: {text}\n\n"
        f"**{n_tokens} tokens**\n\n"
        f"```\n{table}\n```\n\n"
        f"**Decoded**: {decoded}"
    )


def build_app() -> gr.Blocks:
    progress = _load_progress()
    n_steps = max(len(progress.get("steps", [])), 1)
    prompt = progress.get("prompt", "नेपालको राजधानी")

    with gr.Blocks(title="NepaliGPT Training Explorer") as demo:
        gr.Markdown(
            f"# 🇳🇵 NepaliGPT Training Explorer\n"
            f"**Prompt**: `{prompt}` — watch generation quality evolve "
            f"across {n_steps} training steps."
        )
        with gr.Row():
            step_slider = gr.Slider(
                minimum=0,
                maximum=n_steps - 1,
                step=1,
                value=0,
                label="Training step index (0 = first checkpoint)",
            )
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Generation")
                out_gen = gr.Textbox(label="Model output", lines=6, interactive=False)
            with gr.Column():
                gr.Markdown("### Metrics")
                with gr.Row():
                    out_loss = gr.Number(label="Validation loss", interactive=False)
                    out_ppl = gr.Number(label="Perplexity", interactive=False)
                out_nw = gr.Textbox(
                    label="Next-token probabilities", lines=8, interactive=False
                )

        step_slider.change(
            lambda i: get_step_view(progress, int(i)),
            inputs=[step_slider],
            outputs=[out_gen, out_loss, out_ppl, out_nw],
        )
        demo.load(
            lambda: get_step_view(progress, 0),
            outputs=[out_gen, out_loss, out_ppl, out_nw],
        )

        gr.Markdown("---")
        gr.Markdown("## Tokenizer Explorer")
        tok_input = gr.Textbox(
            label="Nepali text",
            value="काठमाडौं विश्वविद्यालयमा अध्ययन गर्दैछु।",
        )
        tok_out = gr.Markdown()
        tok_btn = gr.Button("Show tokenization")
        tok_btn.click(analyze_tokenizer, inputs=[tok_input], outputs=[tok_out])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
