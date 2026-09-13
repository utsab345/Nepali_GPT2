"""NepaliGPT: a GPT-2 style causal language model trained on Nepali text.

Public API:
    NepaliGPT                    — the decoder-only transformer model
    MODEL_CONFIGS                — available architecture presets
    load_model_and_tokenizer    — load a checkpoint + SentencePiece tokenizer
    generate                    — autoregressive text completion
    next_words                  — top-N next-token probabilities

Everything else (training, data prep, CLI) lives in the ``train``,
``data.prep`` and ``__main__`` submodules and is imported as needed so
importing this package stays cheap.
"""

from nepali_gpt2.config import MODEL_CONFIGS
from nepali_gpt2.generate import generate, load_model_and_tokenizer, next_words
from nepali_gpt2.model import NepaliGPT

# Keep in sync with pyproject.toml.
__version__ = "1.0.0"

__all__ = [
    "NepaliGPT",
    "MODEL_CONFIGS",
    "load_model_and_tokenizer",
    "generate",
    "next_words",
    "__version__",
]
