"""
scripts/download_datasets.py
Downloads all 5 datasets needed to train KANHA from scratch.

Datasets:
  Pre-training (Stage 1):
    1. WikiText-103       ~500MB  — clean Wikipedia text
    2. TinyStories        ~2GB    — short stories, great for fluency
    3. OpenWebText        ~1GB    — diverse real web text (100k docs)

  Fine-tuning (Stage 2):
    4. Alpaca 52k         ~45MB   — instruction/response pairs
    5. OpenHermes 2.5     ~100MB  — high-quality GPT-4 generated pairs (20k)

  Alignment (Stage 3):
    6. Anthropic HH-RLHF  ~50MB  — human preference pairs for DPO

Run:
    pip install datasets tqdm
    python scripts/download_datasets.py

All files land in:
    data/raw/         → pre-training .txt files
    data/processed/   → SFT and DPO .jsonl files

Resumable: safe to re-run if interrupted — skips already-completed files.
"""

import os
import json
import time
import sys
from pathlib import Path

# ── Check dependencies ────────────────────────────────────────────────────────
try:
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError:
    print("Missing dependencies. Run:")
    print("  pip install datasets tqdm")
    sys.exit(1)

# ── Directory setup ───────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def already_done(path, min_bytes=1024):
    """Returns True if file exists and is non-trivially large."""
    p = Path(path)
    return p.exists() and p.stat().st_size > min_bytes

def print_size(path):
    size = Path(path).stat().st_size
    if size > 1_000_000_000:
        print(f"  ✓  Size: {size/1e9:.2f} GB")
    elif size > 1_000_000:
        print(f"  ✓  Size: {size/1e6:.1f} MB")
    else:
        print(f"  ✓  Size: {size/1e3:.1f} KB")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET 1: WikiText-103
# ─────────────────────────────────────────────────────────────────────────────
def download_wikitext():
    separator("1/6  WikiText-103  (clean Wikipedia)")
    out = RAW_DIR / "wiki103.txt"

    if already_done(out, min_bytes=10_000_000):
        print(f"  Already downloaded → {out}")
        print_size(out)
        return

    print("  Downloading from Hugging Face...")
    ds = load_dataset(
        "wikitext",
        "wikitext-103-raw-v1",
        split="train",
        trust_remote_code=True,
    )

    written = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in tqdm(ds, desc="  Writing WikiText"):
            text = row["text"].strip()
            if text and len(text) > 50:      # skip section headers and blanks
                f.write(text + "\n\n")
                written += 1

    print(f"  ✓  {written:,} paragraphs written → {out}")
    print_size(out)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 2: TinyStories
# ─────────────────────────────────────────────────────────────────────────────
def download_tinystories():
    separator("2/6  TinyStories  (short stories — great for fluency)")
    out = RAW_DIR / "tinystories.txt"

    if already_done(out, min_bytes=50_000_000):
        print(f"  Already downloaded → {out}")
        print_size(out)
        return

    print("  Downloading from Hugging Face (streaming)...")
    ds = load_dataset(
        "roneneldan/TinyStories",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    MAX_STORIES = 500_000    # ~1GB — enough without being excessive
    written = 0

    with open(out, "w", encoding="utf-8") as f:
        for row in tqdm(ds, desc="  Writing TinyStories", total=MAX_STORIES):
            if written >= MAX_STORIES:
                break
            text = row["text"].strip()
            if text:
                f.write(text + "\n\n")
                written += 1

    print(f"  ✓  {written:,} stories written → {out}")
    print_size(out)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 3: OpenWebText
# ─────────────────────────────────────────────────────────────────────────────
def download_openwebtext():
    separator("3/6  OpenWebText  (diverse web text)")
    out = RAW_DIR / "openwebtext.txt"

    if already_done(out, min_bytes=50_000_000):
        print(f"  Already downloaded → {out}")
        print_size(out)
        return

    print("  Downloading from Hugging Face (streaming, 100k docs)...")
    ds = load_dataset(
        "Skylion007/openwebtext",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    MAX_DOCS = 100_000
    written  = 0

    with open(out, "w", encoding="utf-8") as f:
        for row in tqdm(ds, desc="  Writing OpenWebText", total=MAX_DOCS):
            if written >= MAX_DOCS:
                break
            text = row["text"].strip()
            # Basic quality filter: skip very short docs
            if text and len(text.split()) > 100:
                f.write(text + "\n\n")
                written += 1

    print(f"  ✓  {written:,} documents written → {out}")
    print_size(out)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 4: Alpaca 52k  (SFT)
# ─────────────────────────────────────────────────────────────────────────────
def download_alpaca():
    separator("4/6  Alpaca 52k  (SFT instruction pairs)")
    out = PROCESSED_DIR / "sft_alpaca.jsonl"

    if already_done(out, min_bytes=1_000_000):
        print(f"  Already downloaded → {out}")
        print_size(out)
        return

    print("  Downloading from Hugging Face...")
    ds = load_dataset(
        "tatsu-lab/alpaca",
        split="train",
        trust_remote_code=True,
    )

    written  = 0
    skipped  = 0

    with open(out, "w", encoding="utf-8") as f:
        for row in tqdm(ds, desc="  Writing Alpaca"):
            instruction = row["instruction"].strip()
            inp         = row["input"].strip()
            response    = row["output"].strip()

            # Quality filters
            if not instruction or not response:
                skipped += 1
                continue
            if len(response) < 20:           # too short
                skipped += 1
                continue
            if len(response) > 4000:         # too long for 60M model
                response = response[:4000]

            # Merge instruction + input if input exists
            full_instruction = f"{instruction}\n\n{inp}".strip() if inp else instruction

            obj = {
                "instruction": full_instruction,
                "response":    response,
            }
            f.write(json.dumps(obj) + "\n")
            written += 1

    print(f"  ✓  {written:,} pairs written ({skipped} skipped) → {out}")
    print_size(out)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 5: OpenHermes 2.5  (SFT — higher quality)
# ─────────────────────────────────────────────────────────────────────────────
def download_openhermes():
    separator("5/6  OpenHermes 2.5  (high-quality SFT pairs)")
    out = PROCESSED_DIR / "sft_hermes.jsonl"

    if already_done(out, min_bytes=1_000_000):
        print(f"  Already downloaded → {out}")
        print_size(out)
        return

    print("  Downloading from Hugging Face (streaming, 20k samples)...")
    ds = load_dataset(
        "teknium/OpenHermes-2.5",
        split="train",
        
        trust_remote_code=True,
    )

    MAX_SAMPLES = 20_000
    written     = 0
    skipped     = 0

    with open(out, "w", encoding="utf-8") as f:
        for row in tqdm(ds, desc="  Writing OpenHermes", total=MAX_SAMPLES):
            if written >= MAX_SAMPLES:
                break

            conversations = row.get("conversations", [])

            # Need at least one user + one assistant turn
            if len(conversations) < 2:
                skipped += 1
                continue

            # Find first human and first gpt turn
            human_turn = next(
                (m["value"] for m in conversations if m.get("from") == "human"), None
            )
            gpt_turn = next(
                (m["value"] for m in conversations if m.get("from") == "gpt"), None
            )

            if not human_turn or not gpt_turn:
                skipped += 1
                continue

            if len(gpt_turn) < 30 or len(gpt_turn) > 4000:
                skipped += 1
                continue

            obj = {
                "instruction": human_turn.strip(),
                "response":    gpt_turn.strip(),
            }
            f.write(json.dumps(obj) + "\n")
            written += 1

    print(f"  ✓  {written:,} pairs written ({skipped} skipped) → {out}")
    print_size(out)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 6: Anthropic HH-RLHF  (DPO alignment)
# ─────────────────────────────────────────────────────────────────────────────
def download_hh_rlhf():
    separator("6/6  Anthropic HH-RLHF  (DPO alignment pairs)")
    out = PROCESSED_DIR / "dpo.jsonl"

    if already_done(out, min_bytes=500_000):
        print(f"  Already downloaded → {out}")
        print_size(out)
        return

    print("  Downloading from Hugging Face...")
    ds = load_dataset(
        "Anthropic/hh-rlhf",
        split="train",
        trust_remote_code=True,
    )

    MAX_SAMPLES = 10_000
    written     = 0
    skipped     = 0

    def extract_last_exchange(text: str):
        """
        HH-RLHF format is a full dialogue string like:
            Human: ...
            Assistant: ...
            Human: ...
            Assistant: (final response)
        We extract the last human prompt and final assistant response.
        """
        parts = text.strip().split("\n\nAssistant:")
        if len(parts) < 2:
            return None, None

        response = parts[-1].strip()
        # Get last human turn
        human_parts = parts[-2].split("\n\nHuman:")
        prompt = human_parts[-1].strip()

        if not prompt or not response:
            return None, None
        return prompt, response

    with open(out, "w", encoding="utf-8") as f:
        for row in tqdm(ds, desc="  Writing HH-RLHF"):
            if written >= MAX_SAMPLES:
                break

            chosen_prompt,   chosen_response   = extract_last_exchange(row["chosen"])
            rejected_prompt, rejected_response = extract_last_exchange(row["rejected"])

            # Skip if extraction failed or responses are too similar/short
            if not chosen_response or not rejected_response:
                skipped += 1
                continue
            if len(chosen_response) < 20 or len(rejected_response) < 20:
                skipped += 1
                continue
            if chosen_response == rejected_response:
                skipped += 1
                continue

            obj = {
                "prompt":   chosen_prompt or "",
                "chosen":   chosen_response,
                "rejected": rejected_response,
            }
            f.write(json.dumps(obj) + "\n")
            written += 1

    print(f"  ✓  {written:,} triples written ({skipped} skipped) → {out}")
    print_size(out)


# ─────────────────────────────────────────────────────────────────────────────
# MERGE SFT FILES
# ─────────────────────────────────────────────────────────────────────────────
def merge_sft():
    separator("Merging SFT datasets")
    out     = PROCESSED_DIR / "sft_combined.jsonl"
    sources = [
        PROCESSED_DIR / "sft_alpaca.jsonl",
        PROCESSED_DIR / "sft_hermes.jsonl",
    ]

    total = 0
    with open(out, "w", encoding="utf-8") as fout:
        for src in sources:
            if not src.exists():
                print(f"  Warning: {src} not found, skipping.")
                continue
            count = 0
            with open(src, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if line:
                        fout.write(line + "\n")
                        count += 1
            print(f"  + {src.name}: {count:,} samples")
            total += count

    print(f"  ✓  Combined: {total:,} total SFT samples → {out}")
    print_size(out)


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def print_summary():
    print(f"\n{'='*60}")
    print("  ALL DATASETS READY")
    print(f"{'='*60}\n")

    pre_train_files = [
        RAW_DIR / "wiki103.txt",
        RAW_DIR / "tinystories.txt",
        RAW_DIR / "openwebtext.txt",
    ]
    sft_file = PROCESSED_DIR / "sft_combined.jsonl"
    dpo_file = PROCESSED_DIR / "dpo.jsonl"

    print("  Pre-training data (data/raw/):")
    total_raw = 0
    for f in pre_train_files:
        if f.exists():
            sz = f.stat().st_size
            total_raw += sz
            print(f"    ✓  {f.name:<25} {sz/1e6:>8.1f} MB")
        else:
            print(f"    ✗  {f.name:<25} MISSING")
    print(f"    {'Total':<25} {total_raw/1e6:>8.1f} MB\n")

    print("  Fine-tuning data (data/processed/):")
    for f in [sft_file, dpo_file]:
        if f.exists():
            sz = f.stat().st_size
            lines = sum(1 for _ in open(f))
            print(f"    ✓  {f.name:<30} {lines:>7,} samples   {sz/1e6:.1f} MB")
        else:
            print(f"    ✗  {f.name:<30} MISSING")

    print(f"\n  Next steps:")
    print(f"    1. python scripts/train_tokenizer.py \\")
    print(f"           --input data/raw/ \\")
    print(f"           --output models/tokenizer/tokenizer\n")
    print(f"    2. python scripts/preprocess_data.py \\")
    print(f"           --input data/raw/ \\")
    print(f"           --output data/processed/train.npy \\")
    print(f"           --tokenizer models/tokenizer/tokenizer.model\n")
    print(f"    3. python main.py train --data data/processed/train.npy\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  KANHA Dataset Downloader")
    print("  Downloading all training data...\n")
    print("  Note: First run downloads from Hugging Face (~3-4GB total).")
    print("  Subsequent runs are instant (skips already-downloaded files).\n")

    start = time.time()

    try:
        download_wikitext()
        download_tinystories()
        download_openwebtext()
        download_alpaca()
        download_openhermes()
        download_hh_rlhf()
        merge_sft()
        print_summary()
    except KeyboardInterrupt:
        print("\n\n  Interrupted. Safe to re-run — progress is saved.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n  Error: {e}")
        print("  Safe to re-run — already-downloaded files are skipped.")
        raise

    elapsed = time.time() - start
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    print(f"\n  Total time: {mins}m {secs}s")
    print("  All done! ✓\n")