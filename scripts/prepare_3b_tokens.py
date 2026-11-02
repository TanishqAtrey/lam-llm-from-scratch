"""
scripts/prepare_3b_tokens.py
Streams, tokenizes, and saves 3.0 Billion tokens directly into train.npy in uint16.

Peak Disk Optimization:
  Streams directly into a pre-allocated memory-mapped .npy array without any
  intermediate .tmp.bin file. This eliminates the 6.0 GB secondary copy penalty,
  keeping peak disk usage strictly at 6.0 GB (instead of 12.0 GB) and avoiding
  the 20 GB Kaggle disk quota exhaustion.

Peak RAM Optimization:
  Flushes token batches every 500k tokens directly to disk, keeping host RAM
  under 50 MB at all times.

Dataset Recipe (3.0 Billion Tokens):
  1. FineWeb-Edu (sample-10BT)   : 1.5B tokens (50%)
  2. Cosmopedia v2                : 1.0B tokens (33.3%)
  3. Python-Code                  : 0.5B tokens (16.7%)

Run:
  python scripts/prepare_3b_tokens.py \
      --tokenizer models/tokenizer/tokenizer.model \
      --output data/processed/train.npy
"""

import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path even when invoked from nested scripts directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from kanha.core.tokenizer import KanhaTokenizer, train_tokenizer
from kanha.utils.logging import get_logger

log = get_logger("prepare_3b")

FLUSH_EVERY = 500_000   # flush every 500k tokens to keep RAM < 50MB


def prepare_3b_tokens(args):
    tok_path = Path(args.tokenizer)
    if not tok_path.exists():
        log.info(f"Tokenizer not found at {tok_path}.")
        log.info("Bootstrapping tokenizer from FineWeb-Edu sample (~50k texts)...")
        tok_train_file = Path("data/raw/tokenizer_bootstrap.txt")
        tok_train_file.parent.mkdir(parents=True, exist_ok=True)
        ds_sample = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        with open(tok_train_file, "w", encoding="utf-8") as f:
            for i, row in enumerate(ds_sample):
                if i >= 50_000:
                    break
                t = row.get("text", "").strip()
                if t:
                    f.write(t + "\n")
        train_tokenizer(
            input_files=str(tok_train_file),
            model_prefix=str(tok_path.parent / tok_path.stem),
            vocab_size=16000,
        )
        if tok_train_file.exists():
            tok_train_file.unlink()
        log.info(f"✓ Tokenizer trained and saved to {tok_path}")

    tokenizer = KanhaTokenizer(str(tok_path))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Sources and token targets (scaled to 3.0 Billion tokens total)
    scale = args.scale  # default 1.0 = 3.0B tokens
    sources = [
        {
            "name": "FineWeb-Edu",
            "repo": "HuggingFaceFW/fineweb-edu",
            "subset": "sample-10BT",
            "field": "text",
            "target_tokens": int(1_500_000_000 * scale),
        },
        {
            "name": "Cosmopedia-v2",
            "repo": "HuggingFaceTB/smollm-corpus",
            "subset": "cosmopedia-v2",
            "field": "text",
            "target_tokens": int(1_000_000_000 * scale),
        },
        {
            "name": "Python-Code",
            "repo": "jtatman/python-code-dataset-500k",
            "subset": None,
            "field": "text",
            "target_tokens": int(500_000_000 * scale),
        },
    ]

    total_target = sum(s["target_tokens"] for s in sources)
    if total_target == 0:
        raise RuntimeError("No tokens requested. Check scale factor or sources configuration.")

    log.info(f"Target Token Budget : {total_target:,} tokens")
    log.info(f"Output File         : {out_path} (dtype=uint16, direct memmap)")
    log.info(f"Peak Disk Footprint : {(total_target * 2) / 1e9:.2f} GB (zero temporary file overhead)")

    # ── Direct streaming into .npy memmap ──
    # Writing directly to out_path eliminates the 6GB intermediate .tmp.bin file,
    # cutting peak disk consumption from 12GB to 6GB and halving disk I/O time.
    mmap = np.lib.format.open_memmap(
        str(out_path), mode="w+", dtype=np.uint16, shape=(total_target,)
    )
    cursor = 0
    buffer = []

    for src in sources:
        target = src["target_tokens"]
        log.info(f"\nStreaming {src['name']} from {src['repo']} (Target: {target/1e6:.1f}M tokens)...")

        kwargs = {"streaming": True, "split": "train"}
        if src["subset"]:
            kwargs["name"] = src["subset"]

        try:
            ds = load_dataset(src["repo"], **kwargs)
        except Exception as e:
            log.error(f"Failed to load {src['repo']}: {e}")
            continue

        src_tokens = 0
        pbar = tqdm(total=target, desc=f"  {src['name']}", unit="tok")

        for row in ds:
            text = row.get(src["field"], "")
            if not text:
                if "code" in row:
                    text = row["code"]
                elif "content" in row:
                    text = row["content"]
                elif "output" in row or "instruction" in row:
                    inst = row.get("instruction", "").strip()
                    out = row.get("output", "").strip()
                    text = f"{inst}\n\n{out}" if inst and out else (out or inst)
            if not text:
                continue
            text = text.strip()
            if len(text) < 40:
                continue

            remaining = min(target - src_tokens, total_target - cursor - len(buffer))
            if remaining <= 0:
                break

            ids = tokenizer.encode(text, add_bos=True, add_eos=True, max_length=remaining)

            buffer.extend(ids)
            count = len(ids)
            src_tokens += count
            pbar.update(count)

            if len(buffer) >= FLUSH_EVERY:
                mmap[cursor:cursor + len(buffer)] = buffer
                cursor += len(buffer)
                buffer = []
                pbar.set_postfix({"total": f"{cursor/1e6:.1f}M"})

            if src_tokens >= target or cursor >= total_target:
                break

        pbar.close()
        if buffer:
            mmap[cursor:cursor + len(buffer)] = buffer
            cursor += len(buffer)
            buffer = []
        log.info(f"✓ {src['name']} completed: {src_tokens:,} tokens")

    # Guard against 0 tokens generated (e.g. network failure / invalid sources)
    if cursor == 0:
        del mmap
        if out_path.exists():
            out_path.unlink()
        raise RuntimeError("No tokens were generated. Check dataset connectivity/arguments.")

    mmap.flush()
    del mmap

    # Truncate any zero-padded tail so no raw 0s (<pad> tokens) poison pretraining
    if cursor < total_target:
        log.warning(
            f"Streaming terminated early: wrote {cursor:,} tokens "
            f"(target was {total_target:,})."
        )
        log.info("Truncating .npy file header and payload to exact written token count...")
        with open(out_path, "rb+") as f:
            import struct
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                np.lib.format.read_array_header_1_0(f)
                prefix_len = 10
            else:
                np.lib.format.read_array_header_2_0(f)
                prefix_len = 12
            header_len = f.tell()
            H = header_len - prefix_len
            d = {"descr": "<u2", "fortran_order": False, "shape": (cursor,)}
            s = str(d)
            padding = H - len(s) - 1
            if padding >= 0:
                header_str = s + (" " * padding) + "\n"
                f.seek(0)
                if version == (1, 0):
                    f.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", H) + header_str.encode("latin1"))
                else:
                    f.write(b"\x93NUMPY\x02\x00" + struct.pack("<I", H) + header_str.encode("latin1"))
            f.truncate(header_len + cursor * 2)
        log.info(f"✓ File truncated to exact {cursor:,} tokens. Zero-padding eliminated.")

    total_tokens = cursor
    size_gb = out_path.stat().st_size / 1e9
    log.info("\nAll done!")
    log.info(f"  Total tokens : {total_tokens:,}")
    log.info(f"  File size    : {size_gb:.2f} GB (saved in uint16, peak disk strictly {size_gb:.2f} GB)")
    log.info(f"\nReady to train: python main.py train --data {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream 3B tokens directly into train.npy")
    parser.add_argument("--tokenizer", default="models/tokenizer/tokenizer.model")
    parser.add_argument("--output", default="data/processed/train.npy")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor (e.g. 0.01 for 30M token test, 1.0 for 3B)")
    args = parser.parse_args()
    prepare_3b_tokens(args)
