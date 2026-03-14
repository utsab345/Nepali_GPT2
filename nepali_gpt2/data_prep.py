import os
import json
import glob
import zipfile
import subprocess
import numpy as np
import sentencepiece as spm
from pathlib import Path

DATA_DIR    = Path("data")
TOK_DIR     = Path("tokenizer")
TOK_PREFIX  = str(TOK_DIR / "nepali_bpe")
CORPUS_FILE = DATA_DIR / "nepali_corpus.txt"
TOKEN_CACHE = DATA_DIR / "tokens.npy"

WIKI_FILE   = DATA_DIR / "wiki_ne.txt"
WEB_FILE    = DATA_DIR / "web_ne.txt"

VOCAB_SIZE       = 16_000
WIKI_MAX_LINES   = 200_000
OSCAR_MAX_LINES  = 500_000
TOKENIZE_CHUNK   = 2_000_000


def setup_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    TOK_DIR.mkdir(exist_ok=True)


#  Wikipedia 

def download_wikipedia():
    """Download Nepali Wikipedia via HuggingFace datasets (streaming)."""
    if WIKI_FILE.exists():
        print("Wikipedia corpus already downloaded ✓")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    print(f"Downloading Nepali Wikipedia (up to {WIKI_MAX_LINES:,} articles)…")
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
            if text:
                f.write(text + "\n")
                written += 1
            if written % 50_000 == 0 and written > 0:
                print(f"  Wikipedia: {written:,} articles")
            if written >= WIKI_MAX_LINES:
                break

    print(f"Wikipedia corpus saved → {WIKI_FILE}  ({written:,} articles)")


#  OSCAR corpus 

def download_oscar():
    """Download OSCAR Nepali corpus from Kaggle (requires credentials)."""
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
        glob.glob(str(oscar_dir / "**" / "*.txt"), recursive=True) +
        glob.glob(str(oscar_dir / "**" / "*.jsonl"), recursive=True) +
        glob.glob(str(oscar_dir / "*.txt"), recursive=False)
    )
    print(f"Found {len(txt_files)} file(s): {[Path(p).name for p in txt_files]}")

    written = 0
    with open(WEB_FILE, "w", encoding="utf-8") as out_f:
        for fpath in txt_files:
            if written >= OSCAR_MAX_LINES:
                break
            if fpath.endswith(".jsonl"):
                with open(fpath, encoding="utf-8") as in_f:
                    for line in in_f:
                        if written >= OSCAR_MAX_LINES:
                            break
                        try:
                            obj  = json.loads(line)
                            text = obj.get("text", "").strip()
                        except Exception:
                            text = line.strip()
                        if len(text) > 30:
                            out_f.write(text + "\n")
                            written += 1
                            if written % 100_000 == 0:
                                print(f"  OSCAR: {written:,}")
            else:
                with open(fpath, encoding="utf-8") as in_f:
                    for line in in_f:
                        if written >= OSCAR_MAX_LINES:
                            break
                        text = line.strip()
                        if len(text) > 30:
                            out_f.write(text + "\n")
                            written += 1
                            if written % 100_000 == 0:
                                print(f"  OSCAR: {written:,}")

    print(f"OSCAR corpus saved → {WEB_FILE}  ({written:,} lines)")


# Merge 

def merge_corpora():
    if CORPUS_FILE.exists():
        print("Merged corpus already exists ✓")
        return

    files = [f for f in [WIKI_FILE, WEB_FILE] if f.exists()]
    assert files, "No corpus files found — run download steps first."

    with open(CORPUS_FILE, "w", encoding="utf-8") as out:
        for f in files:
            with open(f, encoding="utf-8") as src:
                for line in src:
                    out.write(line)

    print(f"Merged corpus saved → {CORPUS_FILE}")


# Tokenizer 

def train_tokenizer() -> spm.SentencePieceProcessor:
    if not Path(TOK_PREFIX + ".model").exists():
        print(f"Training SentencePiece BPE tokenizer (vocab={VOCAB_SIZE})…")
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
            input_sentence_size=1_000_000,
            shuffle_input_sentence=True,
        )
        print("Tokenizer saved ✓")
    else:
        print("Tokenizer already trained ✓")

    sp = spm.SentencePieceProcessor()
    sp.load(TOK_PREFIX + ".model")
    return sp


#  Tokenize corpus 

def tokenize_corpus(sp: spm.SentencePieceProcessor):
    if TOKEN_CACHE.exists():
        print("Token cache already exists ✓")
        return

    BOS_ID = sp.bos_id()
    EOS_ID = sp.eos_id()

    print("Tokenizing corpus (chunked)…")
    tmp_dir     = DATA_DIR / "tok_chunks"
    tmp_dir.mkdir(exist_ok=True)

    buf         = []
    chunk_idx   = 0
    total_toks  = 0
    total_lines = 0
    chunk_files = []

    def flush_chunk(buf, idx):
        path = tmp_dir / f"chunk_{idx:04d}.npy"
        np.save(path, np.array(buf, dtype=np.int32))
        return path

    with open(CORPUS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            buf.extend([BOS_ID] + sp.encode(line, out_type=int) + [EOS_ID])
            total_lines += 1

            if len(buf) >= TOKENIZE_CHUNK:
                chunk_files.append(flush_chunk(buf, chunk_idx))
                total_toks += len(buf)
                chunk_idx  += 1
                buf.clear()

            if total_lines % 100_000 == 0:
                print(f"  {total_lines:,} lines → ~{total_toks + len(buf):,} tokens")

    if buf:
        chunk_files.append(flush_chunk(buf, chunk_idx))
        total_toks += len(buf)
        buf.clear()

    print(f"Merging {len(chunk_files)} chunks ({total_toks:,} tokens)…")
    merged = np.memmap(TOKEN_CACHE, dtype=np.int32, mode="w+", shape=(total_toks,))
    pos = 0
    for path in chunk_files:
        part = np.load(path)
        merged[pos : pos + len(part)] = part
        pos += len(part)
        os.remove(path)
    merged.flush()
    del merged
    tmp_dir.rmdir()
    print(f"Saved {total_toks:,} tokens → {TOKEN_CACHE}")



def main():
    setup_dirs()
    download_wikipedia()
    download_oscar()
    merge_corpora()
    sp = train_tokenizer()
    tokenize_corpus(sp)
    print("\nData preparation complete ✓")


if __name__ == "__main__":
    main()
