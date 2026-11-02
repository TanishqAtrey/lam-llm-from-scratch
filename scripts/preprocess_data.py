"""
scripts/preprocess_data.py
Tokenizes raw .txt files into a flat .npy array for training.

Memory-safe: streams files in 2000-line batches, writes to disk
incrementally. Never loads more than ~50MB into RAM at once.

Run:
    python scripts/preprocess_data.py \
        --input      data/raw/ \
        --output     data/processed/train.npy \
        --tokenizer  models/tokenizer/tokenizer.model
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from kanha.core.tokenizer import KanhaTokenizer
from kanha.utils.logging import get_logger

log = get_logger("preprocess")

FLUSH_EVERY = 500_000   # write to disk every 500k tokens


def preprocess(args):
    tokenizer = KanhaTokenizer(args.tokenizer)

    # Collect .txt files
    txt_files = []
    if os.path.isdir(args.input):
        for fname in sorted(os.listdir(args.input)):
            if fname.endswith(".txt"):
                txt_files.append(os.path.join(args.input, fname))
    else:
        txt_files = [args.input]

    if not txt_files:
        log.error(f"No .txt files found in {args.input}")
        return

    log.info(f"Found {len(txt_files)} file(s):")
    for f in txt_files:
        log.info(f"  {Path(f).name}  ({os.path.getsize(f)/1e6:.1f} MB)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.bin")
    if tmp_path.exists():
        tmp_path.unlink()

    total_tokens = 0
    buffer       = []

    def flush(f_bin, buf):
        if buf:
            # uint16 is loss-free for 16k vocab (<65535) and saves 50% disk space (6GB vs 12GB)
            np.array(buf, dtype=np.uint16).tofile(f_bin)
        return []

    with open(tmp_path, "wb") as f_bin:
        for fpath in txt_files:
            fname   = Path(fpath).name
            n_lines = sum(1 for _ in open(fpath, "rb"))
            chunk   = []

            log.info(f"Processing {fname} ({n_lines:,} lines)...")
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f_txt:
                pbar = tqdm(f_txt, total=n_lines, desc=f"  {fname}", unit="lines")
                for line in pbar:
                    line = line.strip()
                    if len(line) < args.min_line_len:
                        continue
                    chunk.append(line)
                    if len(chunk) >= args.chunk_lines:
                        ids = tokenizer.encode(" ".join(chunk), add_bos=False, add_eos=False)
                        buffer.extend(ids)
                        total_tokens += len(ids)
                        chunk = []
                        if len(buffer) >= FLUSH_EVERY:
                            buffer = flush(f_bin, buffer)
                            pbar.set_postfix({"tokens": f"{total_tokens/1e6:.1f}M"})
                if chunk:
                    ids = tokenizer.encode(" ".join(chunk), add_bos=False, add_eos=False)
                    buffer.extend(ids)
                    total_tokens += len(ids)

        flush(f_bin, buffer)

    if total_tokens == 0:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError("No tokens were generated. Check dataset/text inputs.")

    # Convert .bin → .npy via chunked memmap (keeps RAM < 100MB, prevents OOM on large files)
    log.info("Converting to memory-mapped .npy...")
    mmap = np.lib.format.open_memmap(
        str(out_path), mode="w+", dtype=np.uint16, shape=(total_tokens,)
    )
    CHUNK_TOKENS = 32 * 1024 * 1024  # 32M tokens = 64MB buffer
    cursor = 0
    with open(tmp_path, "rb") as f_in:
        pbar_conv = tqdm(total=total_tokens, desc="  Writing .npy", unit="tok")
        while cursor < total_tokens:
            n = min(CHUNK_TOKENS, total_tokens - cursor)
            chunk = np.fromfile(f_in, dtype=np.uint16, count=n)
            if len(chunk) == 0:
                break
            mmap[cursor:cursor + len(chunk)] = chunk
            cursor += len(chunk)
            pbar_conv.update(len(chunk))
        pbar_conv.close()
    mmap.flush()
    del mmap
    tmp_path.unlink()

    log.info(f"Done!")
    log.info(f"  Total tokens : {total_tokens:,}")
    log.info(f"  Output       : {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")
    log.info(f"\nNext: python main.py train --data {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        required=True)
    parser.add_argument("--output",       required=True)
    parser.add_argument("--tokenizer",    required=True)
    parser.add_argument("--chunk_lines",  type=int, default=2000)
    parser.add_argument("--min_line_len", type=int, default=20)
    args = parser.parse_args()
    preprocess(args)
