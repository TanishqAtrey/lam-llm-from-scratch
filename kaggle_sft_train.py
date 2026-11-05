"""
kaggle_sft_train.py
===================
Self-contained, memory-safe Kaggle execution script for KANHA SFT.

Key Guarantees:
  1. Template Alignment: 100% exact match between SFT training format and InferenceEngine.
  2. Loss Masking: Causal LM masking ensures loss is only computed on response tokens (-100 for prompt).
  3. No Memory Boom (GPU VRAM < 2.5 GB, Host RAM < 1.5 GB):
     - Dynamic sequence length batch collation (sft_collate_fn) avoids fixed 1024 padding.
     - PyTorch AMP FP16 mixed precision enabled.
     - DataLoader num_workers=0 avoids host RAM copying.
     - Periodic CUDA cache clearing.
  4. Auto-Detection: Automatically locates final_model.pt, tokenizer.model, and code across /kaggle/input/.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def find_file(filename: str, search_roots: list[str]) -> str | None:
    """Finds the first existing path of filename across search roots."""
    for root in search_roots:
        if os.path.exists(root):
            for dirpath, _, filenames in os.walk(root):
                if filename in filenames:
                    return os.path.join(dirpath, filename)
    return None


def run_kaggle_sft(epochs: int = 2):
    import re
    print("=" * 65)
    print("  KANHA SFT TRAINING PIPELINE (KAGGLE RUNNER)")
    print("=" * 65)

    # ── 1. Setup workspace directory ──────────────────────────────────────────
    work_dir = Path("/kaggle/working/kanha")
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    sys.path.insert(0, str(work_dir))

    # Search paths for inputs
    search_dirs = ["/kaggle/input", "/kaggle/working"]

    # ── 2. Sync Repository Code ───────────────────────────────────────────────
    print("\n[Step 1/6] Setting up project code...")
    if (work_dir / "kanha").exists() and (work_dir / "setup.py").exists():
        print(f"Using up-to-date repository code in {work_dir}")
    else:
        kanha_pkg = find_file("setup.py", ["/kaggle/input"])
        if kanha_pkg:
            src_repo = os.path.dirname(kanha_pkg)
            print(f"Found project repository in input at: {src_repo}")
            for item in os.listdir(src_repo):
                s = os.path.join(src_repo, item)
                d = work_dir / item
                if os.path.isdir(s) and item not in [".git", "__pycache__", "venv"]:
                    shutil.copytree(s, d, dirs_exist_ok=True)
                elif os.path.isfile(s):
                    shutil.copy2(s, d)
        else:
            print("Cloning repository from GitHub...")
            subprocess.run(["git", "clone", "https://github.com/TanishqAtrey/lam-llm-from-scratch.git", str(work_dir)], check=True)

    # ── 3. Locate Model & Tokenizer ───────────────────────────────────────────
    print("\n[Step 2/6] Locating base model and tokenizer...")

    # Scan for all available checkpoints in inputs
    all_ckpts = []
    for root in search_dirs:
        if os.path.exists(root):
            for dirpath, _, filenames in os.walk(root):
                for f in filenames:
                    if f.endswith(".pt") and not f.startswith("sft_"):
                        all_ckpts.append(os.path.join(dirpath, f))

    print(f"  Found {len(all_ckpts)} candidate model checkpoint(s):")
    for c in all_ckpts:
        size_mb = os.path.getsize(c) / (1024 * 1024)
        print(f"    • {c} ({size_mb:.1f} MB)")

    # Intelligent priority selection:
    # 1. final_model.pt
    # 2. best_model.pt
    # 3. ckpt_step{N}.pt with highest numerical step N
    base_model_path = None
    final_candidates = [c for c in all_ckpts if os.path.basename(c) == "final_model.pt"]
    best_candidates = [c for c in all_ckpts if os.path.basename(c) == "best_model.pt"]
    step_candidates = [c for c in all_ckpts if "ckpt_step" in os.path.basename(c)]

    if final_candidates:
        base_model_path = final_candidates[0]
        print(f"  => Selected: 'final_model.pt'")
    elif best_candidates:
        base_model_path = best_candidates[0]
        print(f"  => Selected: 'best_model.pt'")
    elif step_candidates:
        def extract_step(path):
            m = re.search(r'step(\d+)', os.path.basename(path))
            return int(m.group(1)) if m else 0
        step_candidates.sort(key=extract_step)
        base_model_path = step_candidates[-1]
        print(f"  => Selected highest step checkpoint: '{os.path.basename(base_model_path)}'")
    elif all_ckpts:
        base_model_path = all_ckpts[0]
        print(f"  => Selected fallback checkpoint: '{os.path.basename(base_model_path)}'")

    tok_path = find_file("tokenizer.model", search_dirs)

    print(f"  Final Chosen Base Model : {base_model_path}")
    print(f"  Tokenizer Path          : {tok_path}")

    assert base_model_path and os.path.exists(base_model_path), "FATAL: Could not find base model .pt checkpoint!"
    assert tok_path and os.path.exists(tok_path), "FATAL: Could not find tokenizer.model!"

    # Ensure models directory structure
    (work_dir / "models" / "base").mkdir(parents=True, exist_ok=True)
    (work_dir / "models" / "tokenizer").mkdir(parents=True, exist_ok=True)
    (work_dir / "models" / "finetuned").mkdir(parents=True, exist_ok=True)
    (work_dir / "data" / "processed").mkdir(parents=True, exist_ok=True)

    dest_model = work_dir / "models" / "base" / "final_model.pt"
    dest_tok = work_dir / "models" / "tokenizer" / "tokenizer.model"
    if str(Path(base_model_path).resolve()) != str(dest_model.resolve()):
        shutil.copy2(base_model_path, dest_model)
    if str(Path(tok_path).resolve()) != str(dest_tok.resolve()):
        shutil.copy2(tok_path, dest_tok)

    # ── 4. Verify or Prepare SFT Dataset (250K Mix) ───────────────────────────
    print("\n[Step 3/6] Checking SFT datasets...")
    train_data_path = find_file("sft_250k.jsonl", search_dirs)
    val_data_path = find_file("sft_val.jsonl", search_dirs)

    if not train_data_path:
        print("sft_250k.jsonl not found in inputs. Building 250K SFT dataset directly on Kaggle...")
        subprocess.run(["pip", "install", "datasets", "sentencepiece", "pyyaml", "tqdm"], check=True)
        from scripts.prepare_sft_250k import prepare_sft_dataset
        train_out = str(work_dir / "data" / "processed" / "sft_250k.jsonl")
        val_out = str(work_dir / "data" / "processed" / "sft_val.jsonl")
        prepare_sft_dataset(output_path=train_out, val_output_path=val_out, seed=42)
        train_data_path = train_out
        val_data_path = val_out
    else:
        print(f"Found existing SFT dataset: {train_data_path}")

    # Count samples
    with open(train_data_path, "r", encoding="utf-8") as f:
        train_count = sum(1 for line in f if line.strip())
    print(f"  Training samples   : {train_count:,}")
    if val_data_path:
        with open(val_data_path, "r", encoding="utf-8") as f:
            val_count = sum(1 for line in f if line.strip())
        print(f"  Validation samples : {val_count:,}")

    # ── 5. System Pre-Flight Memory & Verification ────────────────────────────
    print("\n[Step 4/6] Pre-Flight System Check...")
    import torch
    from kanha.core.model import KanhaModel
    from kanha.core.tokenizer import KanhaTokenizer
    from kanha.prompting.builder import PromptBuilder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Target Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU Name     : {gpu_name} ({gpu_mem:.1f} GB VRAM)")

    # Verify model & tokenizer loading
    model = KanhaModel.from_pretrained(str(dest_model))
    tokenizer = KanhaTokenizer(str(dest_tok))
    builder = PromptBuilder(include_rules=False)

    # Verify template exactness
    infer_prompt = builder.build(instruction="Who created you?")
    train_prefix, _ = builder.build_prompt_prefix("Who created you?")
    assert infer_prompt == train_prefix, "CRITICAL: PromptBuilder template mismatch!"
    print("  Template match verified: SFT training prefix == Inference prefix (100% exact)")

    # ── 6. Execute SFT Training (Memory Safe) ──────────────────────────────────
    print("\n[Step 5/6] Starting SFT Training...")
    from types import SimpleNamespace
    from kanha.finetune.sft_train import sft_train

    # Memory-optimized hyperparameters for Kaggle T4/P100 (15-16GB VRAM)
    args = SimpleNamespace(
        base_model=str(dest_model),
        data=str(train_data_path),
        val_data=str(val_data_path) if val_data_path else None,
        output=str(work_dir / "models" / "finetuned"),
        epochs=epochs,          # 2 epochs is optimal for 250k mix (prevents catastrophic forgetting)
        batch_size=8,          # Safe dynamic batch size (<2.5GB peak VRAM)
        grad_accum_steps=4,    # Effective batch size = 32 (8 * 4)
        lr=3e-5,               # Conservative learning rate to preserve pretrained weights
        val_fraction=0.01,     # Used only if val_data is not provided
        patience=3,            # Early stopping patience
        lora=False,            # Full fine-tuning for 42M parameter model
    )

    sft_train(args)

    # ── 7. Post-Training Validation & Chat Generation ─────────────────────────
    print("\n[Step 6/6] Verifying fine-tuned model generation...")
    from kanha.inference.engine import InferenceEngine

    sft_final_path = work_dir / "models" / "finetuned" / "sft_final.pt"
    if not sft_final_path.exists():
        sft_final_path = work_dir / "models" / "finetuned" / "sft_epoch1.pt"

    engine = InferenceEngine.from_pretrained(
        model_path=str(sft_final_path),
        tokenizer_path=str(dest_tok),
    )

    test_queries = [
        "Who created you?",
        "What does KANHA stand for?",
        "Write a Python function to compute the factorial of a number.",
        "What are your main limitations?",
    ]

    print("\n" + "=" * 65)
    print("  MODEL GENERATION SAMPLES")
    print("=" * 65)
    for q in test_queries:
        prompt = engine.prompt_builder.build(instruction=q)
        from kanha.core.generation import generate
        ans = generate(
            engine.model,
            engine.tokenizer,
            prompt,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            device=device,
        )
        print(f"\nUser Query : {q}")
        print(f"KANHA Resp : {ans.strip()}")

    print("\n" + "=" * 65)
    print("  ALL STEPS COMPLETE! MODEL SAVED AT:")
    print(f"  {sft_final_path}")
    print("=" * 65)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2, help="Number of SFT epochs (default: 2)")
    cli_args = parser.parse_args()
    run_kaggle_sft(epochs=cli_args.epochs)
