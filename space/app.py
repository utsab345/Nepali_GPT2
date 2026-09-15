"""NepaliGPT public Gradio demo."""

import os

import gradio as gr
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM

MODEL_ID = os.getenv("MODEL_ID", "utsabdahal34/NepaliGPT-base")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = spm.SentencePieceProcessor(
    model_file=hf_hub_download(MODEL_ID, "tokenizer.model")
)
MODEL = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE).eval()


@torch.inference_mode()
def generate(prompt: str, max_new: int, temperature: float, top_p: float) -> str:
    if not prompt.strip():
        return "कृपया एउटा prompt लेख्नुहोस्।"
    ids = torch.tensor(
        [[TOKENIZER.bos_id()] + TOKENIZER.encode(prompt)], dtype=torch.long, device=DEVICE
    )
    output = MODEL.generate(
        ids,
        max_new_tokens=int(max_new),
        do_sample=temperature > 0,
        temperature=max(float(temperature), 1e-5),
        top_p=float(top_p),
        pad_token_id=TOKENIZER.pad_id(),
        eos_token_id=TOKENIZER.eos_id(),
    )
    return TOKENIZER.decode(output[0].tolist())


with gr.Blocks(title="NepaliGPT") as demo:
    gr.Markdown(
        "# NepaliGPT\n"
        "A GPT-style Nepali language model trained from scratch. "
        "[Source code](https://github.com/utsab345/Nepali_GPT2) · "
        "[Model](https://huggingface.co/utsabdahal34/NepaliGPT-base)"
    )
    prompt = gr.Textbox(label="Prompt", value="नेपाल एक सुन्दर", lines=3)
    with gr.Row():
        max_new = gr.Slider(8, 128, value=64, step=8, label="Max new tokens")
        temperature = gr.Slider(0, 1.5, value=0.8, step=0.05, label="Temperature")
        top_p = gr.Slider(0.5, 1, value=0.92, step=0.01, label="Top-p")
    button = gr.Button("Generate", variant="primary")
    output = gr.Textbox(label="Generated text", lines=10)
    button.click(generate, [prompt, max_new, temperature, top_p], output)


if __name__ == "__main__":
    demo.launch()
