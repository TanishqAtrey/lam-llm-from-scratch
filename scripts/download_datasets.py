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
# DATASET 1: FineWeb-Edu (High quality educational web text ~1.5B tokens)
# ─────────────────────────────────────────────────────────────────────────────
def download_fineweb_edu(max_docs=400_000):
    separator("1/6  FineWeb-Edu  (educational web text, 1.5B token target)")
    out = RAW_DIR / "fineweb_edu.txt"

    if already_done(out, min_bytes=50_000_000):
        print(f"  Already downloaded → {out}")
        print_size(out)
        return

    print("  Streaming from Hugging Face (HuggingFaceFW/fineweb-edu, sample-10BT)...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    written = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in tqdm(ds, desc="  Writing FineWeb-Edu", total=max_docs):
            if written >= max_docs:
                break
            text = row.get("text", "").strip()
            if text and len(text.split()) > 60:
                f.write(text + "\n\n")
                written += 1

    print(f"  ✓  {written:,} documents written → {out}")
    print_size(out)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 2: Cosmopedia v2 (Synthetic textbooks & articles ~1.0B tokens)
# ─────────────────────────────────────────────────────────────────────────────
def download_cosmopedia(max_docs=300_000):
    separator("2/6  Cosmopedia v2  (synthetic textbooks & course explanations)")
    out = RAW_DIR / "cosmopedia.txt"

    if already_done(out, min_bytes=50_000_000):
        print(f"  Already downloaded → {out}")
        print_size(out)
        return

    print("  Streaming from Hugging Face (HuggingFaceTB/smollm-corpus, cosmopedia-v2)...")
    ds = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        name="cosmopedia-v2",
        split="train",
        streaming=True,
    )

    written = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in tqdm(ds, desc="  Writing Cosmopedia", total=max_docs):
            if written >= max_docs:
                break
            text = row.get("text", "").strip()
            if text and len(text) > 100:
                f.write(text + "\n\n")
                written += 1

    print(f"  ✓  {written:,} textbooks/articles written → {out}")
    print_size(out)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 3: Python-Edu / Code (Clean Python programming logic ~500M tokens)
# ─────────────────────────────────────────────────────────────────────────────
def download_python_edu(max_samples=250_000):
    separator("3/6  Python Code  (code logic & syntax for reasoning / LAM)")
    out = RAW_DIR / "python_code.txt"

    if already_done(out, min_bytes=30_000_000):
        print(f"  Already downloaded → {out}")
        print_size(out)
        return

    print("  Streaming from Hugging Face (jtatman/python-code-dataset-500k)...")
    ds = load_dataset(
        "jtatman/python-code-dataset-500k",
        split="train",
        streaming=True,
    )

    written = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in tqdm(ds, desc="  Writing Python Code", total=max_samples):
            if written >= max_samples:
                break
            text = row.get("text", "").strip()
            if text and len(text) > 80:
                f.write(text + "\n\n")
                written += 1

    print(f"  ✓  {written:,} code samples written → {out}")
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

    print("  Downloading from Hugging Face (20k samples)...")
    ds = load_dataset(
        "teknium/OpenHermes-2.5",
        split="train",
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
        We extract the full dialogue context up to the final Assistant turn as the prompt,
        preserving multi-turn history, and the final turn as the response.
        """
        last_asst_idx = text.rfind("\n\nAssistant:")
        if last_asst_idx == -1:
            return None, None

        prompt = text[:last_asst_idx].strip()
        response = text[last_asst_idx + len("\n\nAssistant:"):].strip()

        # Normalize Human -> User for conversation history consistency with ShortTermMemory
        prompt = prompt.replace("\n\nHuman:", "\n\nUser:").replace("\nHuman:", "\nUser:")
        if prompt.startswith("Human:"):
            prompt = "User:" + prompt[len("Human:"):]
        elif prompt.startswith("\n\nUser:"):
            prompt = prompt[len("\n\nUser:"):].strip()
        elif prompt.startswith("User:"):
            prompt = prompt.strip()

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
        RAW_DIR / "fineweb_edu.txt",
        RAW_DIR / "cosmopedia.txt",
        RAW_DIR / "python_code.txt",
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
    print("  Note: Streams directly from Hugging Face.")
    print("  Subsequent runs are instant (skips already-downloaded files).\n")

    start = time.time()

    try:
        download_fineweb_edu()
        download_cosmopedia()
        download_python_edu()
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