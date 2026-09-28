"""
KANHA SFT Training — Kaggle Notebook
=====================================
Copy-paste this entire script into a Kaggle notebook cell and run it.

Prerequisites (upload these as Kaggle Datasets):
  1. Your project code (kanha-fixed.zip)
  2. Your base model (final_model.pt)
  3. Your tokenizer (tokenizer.model)
  4. Your SFT data (sft_combined.jsonl)
"""

# ══════════════════════════════════════════════════════════════════
# STEP 1: Install dependencies
# ══════════════════════════════════════════════════════════════════
import subprocess
subprocess.run(["pip", "install", "sentencepiece", "pyyaml", "tqdm", "rich"], check=True)

# ══════════════════════════════════════════════════════════════════
# STEP 2: Setup paths — UPDATE THESE to match your Kaggle dataset paths
# ══════════════════════════════════════════════════════════════════
import os
import shutil

# ── CHANGE THESE PATHS to match where you uploaded files on Kaggle ──
# When you upload a dataset named "kanha-files" on Kaggle, files go to:
# /kaggle/input/kanha-files/

CODE_ZIP = "/kaggle/input/kanha-files/kanha-fixed.zip"          # your corrected code
BASE_MODEL = "/kaggle/input/kanha-files/final_model.pt"         # your base pretrained model
TOKENIZER = "/kaggle/input/kanha-files/tokenizer.model"         # your tokenizer
SFT_DATA = "/kaggle/input/kanha-files/sft_combined.jsonl"       # your SFT training data

# Working directory
WORK_DIR = "/kaggle/working/kanha"

# ══════════════════════════════════════════════════════════════════
# STEP 3: Extract code and setup directory structure
# ══════════════════════════════════════════════════════════════════
import zipfile

# Extract code
os.makedirs(WORK_DIR, exist_ok=True)
with zipfile.ZipFile(CODE_ZIP, 'r') as z:
    z.extractall("/kaggle/working/")

# Find the extracted directory
extracted = "/kaggle/working/kanha-source/kanha-source-clean"
if os.path.exists(extracted):
    # Move contents to WORK_DIR
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    shutil.move(extracted, WORK_DIR)

os.chdir(WORK_DIR)
print(f"Working directory: {os.getcwd()}")
print(f"Files: {os.listdir('.')}")

# Setup model/tokenizer/data directories
os.makedirs("models/base", exist_ok=True)
os.makedirs("models/tokenizer", exist_ok=True)
os.makedirs("models/finetuned", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Copy files to expected locations
shutil.copy2(BASE_MODEL, "models/base/final_model.pt")
shutil.copy2(TOKENIZER, "models/tokenizer/tokenizer.model")
shutil.copy2(SFT_DATA, "data/processed/sft_combined.jsonl")

print("\nFiles in place:")
print(f"  Base model:  {os.path.exists('models/base/final_model.pt')}")
print(f"  Tokenizer:   {os.path.exists('models/tokenizer/tokenizer.model')}")
print(f"  SFT data:    {os.path.exists('data/processed/sft_combined.jsonl')}")

# Count SFT samples
with open("data/processed/sft_combined.jsonl") as f:
    n = sum(1 for line in f if line.strip())
print(f"  SFT samples: {n:,}")

# ══════════════════════════════════════════════════════════════════
# STEP 4: Verify everything loads correctly BEFORE training
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*50)
print("  PRE-TRAINING VERIFICATION")
print("="*50)

import sys
sys.path.insert(0, WORK_DIR)

from kanha.core.model import KanhaModel
from kanha.core.tokenizer import KanhaTokenizer
from kanha.prompting.builder import PromptBuilder
from kanha.utils.helpers import get_device

device = get_device()
print(f"\nDevice: {device}")

# Load and verify model
model = KanhaModel.from_pretrained("models/base/final_model.pt")
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# Load and verify tokenizer
tokenizer = KanhaTokenizer("models/tokenizer/tokenizer.model")
print(f"Tokenizer loaded: {tokenizer.vocab_size:,} vocab")

# Verify template
builder = PromptBuilder(include_rules=False)
test_train = builder.build_training_pair("What is Python?", "A programming language.")
test_infer = builder.build(instruction="What is Python?")
assert test_train.startswith(test_infer), "TEMPLATE MISMATCH!"
print(f"Template verified: inference prompt is prefix of training text")

# Verify SFT data loads
import json
with open("data/processed/sft_combined.jsonl") as f:
    sample = json.loads(f.readline())
assert "instruction" in sample and "response" in sample, "Bad SFT data format!"
print(f"SFT data format verified: keys={list(sample.keys())}")

# Test tokenization roundtrip
test_text = builder.build_training_pair(sample["instruction"], sample["response"])
ids = tokenizer.encode(test_text, add_bos=True, add_eos=True)
decoded = tokenizer.decode(ids)
print(f"Tokenization roundtrip OK: {len(ids)} tokens")

print("\nAll checks passed! Ready to train.\n")

# ══════════════════════════════════════════════════════════════════
# STEP 5: TRAIN SFT
# ══════════════════════════════════════════════════════════════════
print("="*50)
print("  STARTING SFT TRAINING")
print("="*50 + "\n")

from kanha.finetune.sft_train import sft_train
from types import SimpleNamespace

args = SimpleNamespace(
    base_model="models/base/final_model.pt",
    data="data/processed/sft_combined.jsonl",
    output="models/finetuned/",
    epochs=2,          # 2 epochs is safe for a 40-60M model
    batch_size=4,      # increase to 8 if GPU memory allows
    lr=3e-5,           # safe learning rate
)

sft_train(args)

print("\n" + "="*50)
print("  SFT TRAINING COMPLETE!")
print("="*50)
print(f"\nModel saved to: models/finetuned/sft_final.pt")
print(f"Also saved: models/finetuned/sft_epoch1.pt, sft_epoch2.pt")

# ══════════════════════════════════════════════════════════════════
# STEP 6: Quick test after SFT
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*50)
print("  QUICK TEST")
print("="*50 + "\n")

from kanha.core.generation import generate

model = KanhaModel.from_pretrained("models/finetuned/sft_final.pt")
builder = PromptBuilder(include_rules=False)

test_questions = [
    "What are the three primary colors?",
    "Give three tips for staying healthy.",
    "What is Python?",
]

for q in test_questions:
    prompt = builder.build(instruction=q)
    response = generate(
        model, tokenizer, prompt,
        max_new_tokens=100,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1,
        device=device,
    )
    print(f"Q: {q}")
    print(f"A: {response[:200]}")
    print()

# ══════════════════════════════════════════════════════════════════
# STEP 7: Download the trained model
# ══════════════════════════════════════════════════════════════════
print("="*50)
print("  DOWNLOAD YOUR TRAINED MODEL")
print("="*50)
print(f"\nFiles to download from /kaggle/working/kanha/models/finetuned/:")
for f in os.listdir("models/finetuned/"):
    size = os.path.getsize(f"models/finetuned/{f}") / 1e6
    print(f"  {f}  ({size:.1f} MB)")
print(f"\nDownload sft_final.pt — that's your trained model.")
print(f"Then run DPO training next.")
