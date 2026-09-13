"""Tokenizer round-trip test (roadmap issue #15).

The SentencePiece model is produced by ``data-prep`` and is git-ignored
(``tokenizer/nepali_bpe.model``), so this test skips when the file is not
present rather than failing in CI or fresh clones.
"""

from pathlib import Path

import pytest

_TOK = Path("tokenizer/nepali_bpe.model")


@pytest.mark.skipif(
    not _TOK.exists(), reason="tokenizer model not present (run data-prep)"
)
def test_tokenizer_round_trip() -> None:
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    assert sp.load(str(_TOK))

    for text in ("नेपाल", "काठमाडौं", "एक सुन्दर देश", "नमस्ते धेरै मीठो"):
        ids = sp.encode(text, out_type=int)
        assert ids
        decoded = sp.decode(ids).replace("▁", "")
        assert text in decoded
