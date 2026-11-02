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
import argparse
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
from kanha.core.tokenizer import KanhaTokenizer
from kanha.utils.logging import get_logger

log = get_logger("prepare_3b")

FLUSH_EVERY = 500_000   # flush every 500k tokens to keep RAM < 50MB


def prepare_3b_tokens(args):
    tokenizer = KanhaTokenizer(args.tokenizer)
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
                continue
            text = text.strip()
            if len(text) < 40:
                continue

            remaining = min(target - src_tokens, total_target - cursor - len(buffer))
            if remaining <= 0:
                break

            ids = tokenizer.encode(text, add_bos=True, add_eos=True)
            if len(ids) > remaining:
                ids = ids[:remaining]

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
