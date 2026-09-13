"""Gradio demo UI for NepaliGPT (roadmap Week 3, issue #10).

Talks to the FastAPI service (``api/main.py``) over HTTP, so it can be
deployed to a Hugging Face Space and pointed at any hosted instance:

    GRADIO_API_URL=http://localhost:8000 GRADIO_PORT=7860 python api/demo.py

Requires: ``pip install gradio httpx``
"""

from __future__ import annotations

import os

import gradio as gr
import httpx

API_URL = os.environ.get("GRADIO_API_URL", "http://localhost:8000")
PORT = int(os.environ.get("GRADIO_PORT", "7860"))
TIMEOUT = 120.0


def generate(prompt: str, max_new: int, temperature: float, top_p: float) -> str:
    try:
        resp = httpx.post(
            f"{API_URL}/generate",
            json={
                "prompt": prompt,
                "max_new": max_new,
                "temperature": temperature,
                "top_p": top_p,
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["text"]
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


def next_tokens(prompt: str, top_n: int) -> str:
    try:
        resp = httpx.post(
            f"{API_URL}/next_token",
            json={"prompt": prompt, "top_n": top_n},
            timeout=30.0,
        )
        resp.raise_for_status()
        return "\n".join(
            f"{p['token']:<15} {p['probability']:.3f}"
            for p in resp.json()["predictions"]
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="NepaliGPT — नेपाली भाषा मोडेल") as demo:
        gr.Markdown(
            "# NepaliGPT\nGPT-2 style Nepali language model, served via FastAPI."
        )
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="नेपाल एक सुन्दर")
                max_new = gr.Slider(10, 200, value=80, step=10, label="Max tokens")
                temperature = gr.Slider(
                    0.1, 1.5, value=0.8, step=0.05, label="Temperature"
                )
                top_p = gr.Slider(0.5, 1.0, value=0.92, step=0.01, label="Top-p")
                generate_btn = gr.Button("Generate")
            with gr.Column():
                output = gr.Textbox(label="Generated text", lines=8)
        generate_btn.click(generate, [prompt, max_new, temperature, top_p], output)

        gr.Markdown("## Next-word prediction")
        with gr.Row():
            next_prompt = gr.Textbox(label="Prompt", value="काठमाडौं")
            top_n = gr.Slider(1, 20, value=10, step=1, label="Top-n")
            predict_btn = gr.Button("Predict next words")
            predictions = gr.Textbox(label="Predictions (word + probability)", lines=6)
        predict_btn.click(next_tokens, [next_prompt, top_n], predictions)
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=PORT)
