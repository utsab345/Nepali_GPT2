# NepaliGPT Training Dataset

> Dataset card for the Nepali text corpus used to train NepaliGPT-base,
> NepaliGPT-small, and the planned NepaliGPT-Instruct models.

## Summary

| Field | Value |
|---|---|
| **Name** | NepaliGPT Corpus |
| **Language** | Nepali (ne) |
| **Total lines** | ~700,000 |
| **Total characters** | ~150M |
| **Total tokens** (16k BPE) | ~41M |
| **Unique tokens** | ~14,200 / 16,000 vocab |
| **Type-token ratio** | ~0.35 |
| **Sources** | Nepali Wikipedia, OSCAR Nepali |
| **Licenses** | CC BY-SA 4.0 (Wikipedia), CC0 (OSCAR) |
| **Created** | 2026-09 |

## Sources

### 1. Nepali Wikipedia

| Field | Value |
|---|---|
| **Source** | `wikimedia/wikipedia` (HuggingFace Datasets) |
| **Dump date** | 2023-11-01 |
| **Language code** | `ne` |
| **Articles downloaded** | ≤200,000 |
| **License** | CC BY-SA 4.0 |
| **Access** | Public, no authentication required |

Nepali Wikipedia is the highest-quality Nepali text source available. It
contains edited, structured articles on diverse topics (geography, history,
science, culture, biography) with consistent formatting and relatively
low noise.

### 2. OSCAR Nepali

| Field | Value |
|---|---|
| **Source** | `hsebarp/oscar-corpus-nepali` (Kaggle) |
| **Lines downloaded** | ≤500,000 |
| **License** | CC0 / research use |
| **Access** | Requires Kaggle API credentials |

OSCAR provides web-crawled Nepali text from diverse internet sources. It
includes news, blogs, forums, and general web content. The quality is more
variable than Wikipedia, but it provides broader topical and register
coverage.

## Processing Pipeline

### 1. Download

Both corpora are downloaded via streaming APIs (HuggingFace `datasets`
for Wikipedia, Kaggle CLI for OSCAR). No full dataset is loaded into
memory during download.

### 2. Filtering

- **OSCAR**: Lines shorter than 30 characters are discarded (headers,
  menus, fragments, navigation text). This removes ~15–20% of raw lines.
- **Wikipedia**: Empty articles are discarded. All article text is
  preserved (including markup headers and section boundaries).

### 3. Merging

Both corpora are concatenated into `data/nepali_corpus.txt` with one
article/line per line. Wikipedia articles come first, followed by OSCAR
lines. No shuffling is applied at the corpus level — the temporal order
is preserved for reproducible train/val splits.

### 4. Unicode Normalization

All text is encoded as UTF-8. SentencePiece training applies implicit
byte-level normalization. No explicit NFC/NFD normalization is applied
before tokenization — SentencePiece handles Devanagari character
boundaries correctly with `character_coverage=0.9995`.

### 5. Deduplication

Exact-line deduplication is **not** applied to the merged corpus.
Potential duplicate content across Wikipedia and OSCAR is accepted as-is.
For the 41M-token corpus, deduplication would remove a negligible
fraction and risks losing valid content.

### 6. Tokenization

A SentencePiece BPE tokenizer is trained on the merged corpus:

| Parameter | Value |
|---|---|
| Vocabulary size | 16,000 |
| Model type | BPE |
| Character coverage | 0.9995 |
| Special tokens | `<pad>` (0), `<unk>` (1), `<s>` (2), `</s>` (3) |
| Training samples | 1,000,000 (shuffled) |
| Threads | All available CPU cores |

### 7. Token Cache

Tokenized text is stored as a memory-mapped `int32` array
(`data/tokens.npy`, ~164 MB). Each article/line is wrapped with BOS/EOS
tokens. The cache enables efficient random-access training without
re-tokenization.

## Train/Validation Split

| Split | Fraction | Tokens (approx.) |
|---|---|---|
| Train | 95% | ~39M |
| Validation | 5% | ~2M |

The split is **temporal** (not shuffled): the first 95% of the token
array is used for training, and the last 5% for validation. This avoids
information leakage where the model sees validation content during
training.

## Statistics

### Corpus Composition

```
Wikipedia articles:    ~200,000  (28.6% of lines)
OSCAR lines:           ~500,000  (71.4% of lines)
Total lines:           ~700,000
Total characters:      ~150M
```

### Token Distribution

```
Total tokens:          ~41M
Tokens per line (avg): ~58
Vocabulary utilization: 14,200 / 16,000 (88.8%)
Type-token ratio:       ~0.35
```

### Content Topics

- **Wikipedia**: Nepal geography, history, politics, culture, religion,
  biography, science, sports, language
- **OSCAR**: News, blogs, forums, social media, general web content,
  mixed registers

## Contamination Considerations

The evaluation benchmark (`eval/benchmarks/`) is constructed from
hand-written Nepali questions and curated examples, **not** extracted
from the training corpus. There is no risk of benchmark contamination
from the training data.

For future benchmarks derived from Wikipedia, a content-hash check
should be performed against the training corpus to verify no overlap.

## Known Limitations

1. **Noisy OSCAR data**: Web-crawled text contains formatting artifacts,
   incomplete sentences, and occasional non-Nepali content.
2. **Wikipedia bias**: Wikipedia over-represents formal/encyclopedic
   registers and under-represents conversational Nepali.
3. **Limited domain coverage**: No legal, medical, or technical Nepali
   text is included. Domain-specific models would need additional data.
4. **No parallel data**: The corpus is monolingual. Translation pairs
   for the instruction dataset are generated separately.
5. **Temporal cutoff**: Wikipedia dump is from 2023-11-01. Events after
   this date are not represented.

## Reproducibility

To reproduce the dataset:

```bash
# Requires HF_TOKEN and KAGGLE credentials
export HF_TOKEN=hf_...
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

python -m nepali_gpt2 data-prep
```

This downloads both corpora, merges them, trains the tokenizer, and
produces the token cache. All outputs are cached — re-running is a
no-op unless files are deleted.

### Checksums

After running `data-prep`, verify the token cache:

```bash
sha256sum data/tokens.npy
sha256sum tokenizer/nepali_bpe.model
```

Record these in `docs/measurements/dataset_checksums.json` for
reproducibility.

## Citation

```bibtex
@software{dahal_nepaligpt_data,
  author = {Utsab Dahal},
  title = {NepaliGPT Training Dataset},
  year = {2026},
  url = {https://github.com/utsab345/Nepali_GPT2},
  note = {Nepali Wikipedia + OSCAR, ~41M tokens}
}
```
