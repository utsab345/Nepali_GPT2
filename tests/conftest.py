"""Real tokenizer fixture for offline integration tests."""

import pytest
import sentencepiece as spm


@pytest.fixture
def tokenizer(tmp_path):
    texts = ["नेपाल", "काठमाडौं", "एक सुन्दर देश", "नमस्ते धेरै मीठो"]
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(texts * 20), encoding="utf-8")
    prefix = str(tmp_path / "tokenizer")
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=prefix,
        vocab_size=64,
        model_type="bpe",
        hard_vocab_limit=False,
        pad_id=0,
        bos_id=2,
        eos_id=3,
        unk_id=1,
    )
    return spm.SentencePieceProcessor(model_file=prefix + ".model")
