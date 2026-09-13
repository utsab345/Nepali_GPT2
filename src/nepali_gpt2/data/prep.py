"""Build the Nepali training corpus and train a SentencePiece tokenizer.

Pipeline (all outputs cached, re-running is a no-op):

1. Download Nepali Wikipedia (~200k articles) from HuggingFace
2. Download the OSCAR Nepali corpus (~500k lines) from Kaggle
3. Merge both into ``data/nepali_corpus.txt``
4. Train a 16k-vocab SentencePiece BPE tokenizer
5. Tokenize and cache all tokens as ``data/tokens.npy``

Memory notes
------------
Each downloader streams from disk/network instead of loading the corpus
into RAM, and tokenization writes chunked ``.npy`` files that are later
concatenated into a single ``memmap``. This keeps peak memory flat even
though the merged corpus is several hundred MB of raw text and the token
cache is ~164 MB.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import sentencepiece as spm

DATA_DIR = Path("data")
TOK_DIR = Path("tokenizer")
TOK_PREFIX = str(TOK_DIR / "nepali_bpe")
CORPUS_FILE = DATA_DIR / "nepali_corpus.txt"
TOKEN_CACHE = DATA_DIR / "tokens.npy"

WIKI_FILE = DATA_DIR / "wiki_ne.txt"
WEB_FILE = DATA_DIR / "web_ne.txt"

VOCAB_SIZE = 16_000
WIKI_MAX_LINES = 200_000
OSCAR_MAX_LINES = 500_000
TOKENIZE_CHUNK = 2_000_000


def setup_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    TOK_DIR.mkdir(exist_ok=True)


# ---- Wikipedia -----------------------------------------------------------

def download_wikipedia() -> None:
    """Download Nepali Wikipedia via HuggingFace datasets (streaming)."""
    if WIKI_FILE.exists():
        print("Wikipedia corpus already downloaded ✓")
        return

    # `datasets` is only needed for the download — keep it an optional,
    # lazily-imported dependency so the rest of this module works without it.
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    print(f"Downloading Nepali Wikipedia (up to {WIKI_MAX_LINES:,} articles)…")
    # streaming=True pulls rows one at a time — we never need to materialise
    # the full dataset in memory, and we cap the number of articles written.
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.ne",
        split="train",
        streaming=True,
    )

    written = 0
    with open(WIKI_FILE, "w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text", "").strip()
            if not text:
                continue
            f.write(text + "\n")
            written += 1
            if written % 50_000 == 0:
                print(f"  Wikipedia: {written:,} articles")
            if written >= WIKI_MAX_LINES:
                break

    print(f"Wikipedia corpus saved → {WIKI_FILE}  ({written:,} articles)")


# ---- OSCAR web corpus ----------------------------------------------------

def download_oscar() -> None:
    """Download the OSCAR Nepali corpus from Kaggle (requires credentials)."""
    if WEB_FILE.exists():
        print("OSCAR corpus already downloaded ✓")
        return

    oscar_zip = DATA_DIR / "oscar-corpus-nepali.zip"
    oscar_dir = DATA_DIR / "oscar_raw"

    if not oscar_zip.exists():
        print("Downloading OSCAR Nepali corpus from Kaggle…")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", "hsebarp/oscar-corpus-nepali",
             "-p", str(DATA_DIR)],
            check=True,
        )
        print("Download complete ✓")

    if not oscar_dir.exists():
        print("Extracting…")
        oscar_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(oscar_zip, "r") as zf:
            zf.extractall(oscar_dir)
        print("Extraction complete ✓")

    txt_files = sorted(
        # The dataset ships its text either as loose .txt files or as
        # .jsonl — handle both so we don't depend on the exact archive layout.
        glob.glob(str(oscar_dir / "**" / "*.txt"), recursive=True)
        + glob.glob(str(oscar_dir / "**" / "*.jsonl"), recursive=True)
        + glob.glob(str(oscar_dir / "*.txt"), recursive=False)
    )
    print(f"Found {len(txt_files)} file(s): {[Path(p).name for p in txt_files]}")

    written = 0
    with open(WEB_FILE, "w", encoding="utf-8") as out_f:
        for fpath in txt_files:
            if written >= OSCAR_MAX_LINES:
                break
            is_jsonl = fpath.endswith(".jsonl")
            with open(fpath, encoding="utf-8") as in_f:
                for line in in_f:
                    if written >= OSCAR_MAX_LINES:
                        break
                    if is_jsonl:
                        try:
                            text = json.loads(line).get("text", "").strip()
                        except Exception:
                            # A malformed JSONL row shouldn't kill the whole
                            # pipeline — fall back to the raw line.
                            text = line.strip()
                    else:
                        text = line.strip()
                    # Skip terse rows (headers, menus, fragments): they add
                    # noise and barely help the language model.
                    if len(text) > 30:
                        out_f.write(text + "\n")
                        written += 1
                        if written % 100_000 == 0:
                            print(f"  OSCAR: {written:,}")

    print(f"OSCAR corpus saved → {WEB_FILE}  ({written:,} lines)")


# ---- Merge ---------------------------------------------------------------

def merge_corpora() -> None:
    if CORPUS_FILE.exists():
        print("Merged corpus already exists ✓")
        return

    files = [f for f in (WIKI_FILE, WEB_FILE) if f.exists()]
    if not files:
        raise FileNotFoundError("No corpus files found — run download steps first.")

    with open(CORPUS_FILE, "w", encoding="utf-8") as out:
        for f in files:
            with open(f, encoding="utf-8") as src:
                for line in src:
                    out.write(line)

    print(f"Merged corpus saved → {CORPUS_FILE}")


# ---- Tokenizer -----------------------------------------------------------

def train_tokenizer() -> spm.SentencePieceProcessor:
    if not Path(TOK_PREFIX + ".model").exists():
        print(f"Training SentencePiece BPE tokenizer (vocab={VOCAB_SIZE})…")
        # SentencePiece markers are fixed ids 0-3 (pad/unk/bos/eos). The
        # loss in model.py ignores id 0 (pad) — these ids must stay stable
        # or every checkpoint becomes unreadable, so don't reorder them.
        spm.SentencePieceTrainer.train(
            input=str(CORPUS_FILE),
            model_prefix=TOK_PREFIX,
            vocab_size=VOCAB_SIZE,
            model_type="bpe",
            character_coverage=0.9995,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            pad_piece="<pad>", unk_piece="<unk>",
            bos_piece="<s>", eos_piece="</s>",
            num_threads=os.cpu_count(),
            # SentencePiece shuffles internally; training on the full lexicon
            # can take a while, so cap samples for reproducible quick retrains.
            input_sentence_size=1_000_000,
            shuffle_input_sentence=True,
        )
        print("Tokenizer saved ✓")
    else:
        print("Tokenizer already trained ✓")

    sp = spm.SentencePieceProcessor()
    sp.load(TOK_PREFIX + ".model")
    return sp


# ---- Tokenize ------------------------------------------------------------

def tokenize_corpus(sp: spm.SentencePieceProcessor) -> None:
    if TOKEN_CACHE.exists():
        print("Token cache already exists ✓")
        return

    bos_id, eos_id = sp.bos_id(), sp.eos_id()

    print("Tokenizing corpus (chunked)…")
    tmp_dir = DATA_DIR / "tok_chunks"
    tmp_dir.mkdir(exist_ok=True)

    # Buffer tokens in ~2M-token chunks, flush each to its own .npy, then
    # concatenate them into a single memmap. This bounds peak memory to a
    # few chunks regardless of corpus size.
    buf, chunk_files = [], []
    chunk_idx = total_toks = total_lines = 0

    def flush_chunk(data, idx: int) -> Path:
        path = tmp_dir / f"chunk_{idx:04d}.npy"
        np.save(path, np.array(data, dtype=np.int32))
        return path

    with open(CORPUS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            buf.extend([bos_id] + sp.encode(line, out_type=int) + [eos_id])
            total_lines += 1

            if len(buf) >= TOKENIZE_CHUNK:
                chunk_files.append(flush_chunk(buf, chunk_idx))
                total_toks += len(buf)
                chunk_idx += 1
                buf.clear()

            if total_lines % 100_000 == 0:
                print(f"  {total_lines:,} lines → ~{total_toks + len(buf):,} tokens")

    if buf:
        chunk_files.append(flush_chunk(buf, chunk_idx))
        total_toks += len(buf)

    print(f"Merging {len(chunk_files)} chunks ({total_toks:,} tokens)…")
    # Write straight through a memmap so we never build the full array twice;
    # each chunk is copied in and deleted immediately after.
    merged = np.memmap(TOKEN_CACHE, dtype=np.int32, mode="w+", shape=(total_toks,))
    pos = 0
    for path in chunk_files:
        part = np.load(path)
        merged[pos: pos + len(part)] = part
        pos += len(part)
        os.remove(path)
    merged.flush()
    del merged
    tmp_dir.rmdir()
    print(f"Saved {total_toks:,} tokens → {TOKEN_CACHE}")


def main(argv: list[str] | None = None) -> None:
    setup_dirs()
    download_wikipedia()
    download_oscar()
    merge_corpora()
    sp = train_tokenizer()
    tokenize_corpus(sp)
    print("\nData preparation complete ✓")


if __name__ == "__main__":
    main()