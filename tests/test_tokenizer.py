"""Exercise a real SentencePiece tokenizer without downloaded artifacts."""


def test_tokenizer_round_trip(tokenizer):
    for text in ("नेपाल", "काठमाडौं", "एक सुन्दर देश", "नमस्ते धेरै मीठो"):
        ids = tokenizer.encode(text)
        assert ids and tokenizer.unk_id() not in ids
        assert tokenizer.decode(ids) == text
    assert [
        tokenizer.pad_id(),
        tokenizer.bos_id(),
        tokenizer.eos_id(),
        tokenizer.unk_id(),
    ] == [0, 2, 3, 1]
