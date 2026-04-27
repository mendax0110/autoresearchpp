"""
Converts the outputs of Karpathy's prepare.py into the formats expected
by autoresearch-cpp:

  data/train.bin         flat int32 token IDs (little-endian)
  data/val.bin           flat int32 token IDs (little-endian)
  data/tokenizer.vocab   one token string per line, index = token ID

Each shard is tokenized once and cached under data/cache/shard_XXXXX.npy.
Interrupted runs resume from the last completed shard.
Shards are streamed into the output files one at a time — peak RAM is
one shard at a time, never the full dataset.

Usage:
    python convert_for_cpp.py \
        --cache-dir ~/.cache/autoresearch \
        --out-dir   data \
        --val-frac  0.005
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def load_tokenizer(cache_dir: Path):
    tok_path = cache_dir / "tokenizer" / "tokenizer.pkl"
    if not tok_path.exists():
        sys.exit(f"Tokenizer not found at {tok_path}. Run prepare.py first.")
    with open(tok_path, "rb") as f:
        tok = pickle.load(f)
    return tok


def extract_vocab(tok, out_dir: Path) -> int:
    vocab_path = out_dir / "tokenizer.vocab"

    id_to_bytes: dict[int, bytes] = {v: k for k, v in tok._mergeable_ranks.items()}
    id_to_special: dict[int, str] = {v: k for k, v in tok._special_tokens.items()}

    vocab_size = tok.max_token_value + 1
    lines: list[str] = []

    for i in range(vocab_size):
        if i in id_to_special:
            lines.append(id_to_special[i])
        elif i in id_to_bytes:
            raw = id_to_bytes[i]
            try:
                lines.append(raw.decode("utf-8"))
            except UnicodeDecodeError:
                lines.append("".join(f"\\x{b:02x}" for b in raw))
        else:
            lines.append(f"<unk_{i}>")

    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Vocab written: {vocab_path}  ({len(lines)} tokens)")
    return len(lines)


def tokenize_shard(pf: Path, tok, eos_id: int) -> np.ndarray:
    table = pq.read_table(pf, columns=["text"])
    texts = table.column("text").to_pylist()

    ids: list[int] = []
    for text in texts:
        if not isinstance(text, str) or not text:
            continue
        ids.extend(tok.encode(text))
        ids.append(eos_id)

    return np.array(ids, dtype=np.int32)


def ensure_shard_cached(pf: Path, tok, eos_id: int, shard_cache_dir: Path) -> Path:
    """Tokenizes pf if not already cached. Returns path to the .npy file."""
    cache_file = shard_cache_dir / (pf.stem + ".npy")
    if not cache_file.exists():
        arr = tokenize_shard(pf, tok, eos_id)
        np.save(cache_file, arr)
    return cache_file


def shard_token_count(cache_file: Path) -> int:
    """Reads only the npy header to get the array length without loading data."""
    with open(cache_file, "rb") as f:
        arr = np.load(cache_file, mmap_mode="r")
        return len(arr)


def stream_to_bins(
        cache_files: list[Path],
        split_token: int,
        train_path: Path,
        val_path: Path,
) -> None:
    """
    Writes train.bin and val.bin by streaming one shard at a time.
    split_token is the global token index where train ends and val begins.
    """
    written = 0
    train_bin = open(train_path, "wb")
    val_bin = open(val_path,   "wb")

    try:
        for cache_file in cache_files:
            arr = np.load(cache_file)
            n = len(arr)

            if written >= split_token:
                # Entire shard goes to val.
                arr.tofile(val_bin)
            elif written + n <= split_token:
                # Entire shard goes to train.
                arr.tofile(train_bin)
            else:
                # Shard straddles the split point.
                cut = split_token - written
                arr[:cut].tofile(train_bin)
                arr[cut:].tofile(val_bin)

            written += n
            del arr
    finally:
        train_bin.close()
        val_bin.close()


def tokenize_shards(cache_dir: Path, tok, val_frac: float, out_dir: Path) -> None:
    data_dir = cache_dir / "data"
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        sys.exit(f"No parquet files found in {data_dir}.")

    shard_cache_dir = out_dir / "cache"
    shard_cache_dir.mkdir(parents=True, exist_ok=True)

    eos_id: int = tok._special_tokens.get("<|endoftext|>", 0)
    print(f"Tokenizing {len(parquet_files)} shard(s)  (eos id={eos_id})...")

    # Pass 1: ensure every shard is cached, one at a time.
    cache_files: list[Path] = []
    for i, pf in enumerate(parquet_files):
        cache_file = shard_cache_dir / (pf.stem + ".npy")
        if cache_file.exists():
            count = shard_token_count(cache_file)
            print(f"  [{i+1}/{len(parquet_files)}] {pf.name}  cached  ({count:,} tokens)")
        else:
            print(f"  [{i+1}/{len(parquet_files)}] {pf.name}  tokenizing...", end="", flush=True)
            arr = tokenize_shard(pf, tok, eos_id)
            np.save(cache_file, arr)
            print(f"  {len(arr):,} tokens  ->  {cache_file.name}")
            del arr
        cache_files.append(cache_file)

    # Pass 2: sum shard sizes from headers only — no data loaded.
    total = sum(shard_token_count(cf) for cf in cache_files)
    split = max(1, int(total * (1.0 - val_frac)))
    print(f"\nTotal tokens: {total:,}  |  train: {split:,}  |  val: {total - split:,}")

    # Pass 3: stream shards into the two output files.
    print("Writing train.bin and val.bin...")
    stream_to_bins(cache_files, split, out_dir / "train.bin", out_dir / "val.bin")

    print(f"train.bin  ->  {out_dir / 'train.bin'}")
    print(f"val.bin    ->  {out_dir / 'val.bin'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert prepare.py outputs for autoresearch-cpp.")
    parser.add_argument("--cache-dir", default="~/.cache/autoresearch", help="Cache directory used by prepare.py (default: ~/.cache/autoresearch)")
    parser.add_argument("--out-dir",   default="data", help="Output directory for .bin and .vocab files (default: data)")
    parser.add_argument("--val-frac",  type=float, default=0.005, help="Fraction of tokens held out for validation (default: 0.005)")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser()
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer(cache_dir)
    extract_vocab(tok, out_dir)
    tokenize_shards(cache_dir, tok, args.val_frac, out_dir)

    print("\nDone. Build and run:")
    print("  cmake -B build -DCMAKE_BUILD_TYPE=Release -DTorch_DIR=<path>")
    print("  cmake --build build -j$(nproc)")
    print("  ./build/autoresearch --device cpu --data-dir data")


if __name__ == "__main__":
    main()