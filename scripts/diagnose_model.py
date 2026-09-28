"""
scripts/diagnose_model.py
Diagnostic tool for debugging garbage output after SFT/DPO.

Run this to check:
  1. Checkpoint integrity (correct keys, matching architecture)
  2. Prompt template formatting (shows what the model actually sees)
  3. Token distribution (detects collapsed vocabulary)
  4. Weight statistics (detects catastrophic forgetting / NaN)

Usage:
    python scripts/diagnose_model.py \
        --model models/finetuned/sft_final.pt \
        --tokenizer models/tokenizer/tokenizer.model
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np


def check_checkpoint(model_path):
    """Checks checkpoint structure and key names."""
    print(f"\n{'='*60}")
    print(f"  1. CHECKPOINT INTEGRITY: {model_path}")
    print(f"{'='*60}")

    if not os.path.exists(model_path):
        print(f"  ERROR: File not found: {model_path}")
        return None

    checkpoint = torch.load(model_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        print(f"  ERROR: Checkpoint is not a dict (type: {type(checkpoint)})")
        print(f"  This means the model was saved with torch.save(model.state_dict(), ...)")
        print(f"  instead of model.save_pretrained(). This WILL cause loading issues.")
        return None

    print(f"  Top-level keys: {list(checkpoint.keys())}")

    # Check for model_config
    if "model_config" in checkpoint:
        config = checkpoint["model_config"]
        print(f"  model_config: {config}")
    else:
        print(f"  WARNING: No 'model_config' key found!")
        print(f"  The checkpoint may have been saved by a different script.")
        print(f"  Expected: {{'model_config': {{...}}, 'model_state': {{...}}}}")

    # Check for model_state
    if "model_state" in checkpoint:
        state = checkpoint["model_state"]
        print(f"  model_state keys ({len(state)}):")
        for key in sorted(state.keys()):
            shape = state[key].shape
            dtype = state[key].dtype
            has_nan = torch.isnan(state[key]).any().item()
            has_inf = torch.isinf(state[key]).any().item()
            mean = state[key].float().mean().item()
            std = state[key].float().std().item()
            flag = ""
            if has_nan:
                flag += " [NaN!]"
            if has_inf:
                flag += " [Inf!]"
            if std < 1e-8:
                flag += " [DEAD - zero variance]"
            if std > 10:
                flag += " [EXPLODED - high variance]"
            print(f"    {key:<50} {str(shape):<25} mean={mean:+.6f} std={std:.6f}{flag}")
    elif "state_dict" in checkpoint:
        print(f"  WARNING: Found 'state_dict' instead of 'model_state'")
        print(f"  The loading code expects 'model_state'. This mismatch causes garbage output!")
        state = checkpoint["state_dict"]
        print(f"  Keys: {list(state.keys())[:5]}...")
    else:
        print(f"  WARNING: Neither 'model_state' nor 'state_dict' found!")
        print(f"  Available keys: {list(checkpoint.keys())}")
        # Maybe it's a raw state dict
        if any(k.startswith("token_emb") or k.startswith("layers") for k in checkpoint.keys()):
            print(f"  It looks like a raw state_dict saved without wrapper.")
            state = checkpoint
        else:
            print(f"  ERROR: Cannot identify model weights in checkpoint.")
            return None

    # Check for LoRA wrapper artifacts
    lora_keys = [k for k in state.keys() if k.startswith("model.")]
    if lora_keys:
        print(f"\n  WARNING: Found {len(lora_keys)} keys with 'model.' prefix!")
        print(f"  This means the checkpoint was saved from a LoRA-wrapped model")
        print(f"  without merging. KanhaModel.from_pretrained() expects keys")
        print(f"  WITHOUT the 'model.' prefix. This WILL cause garbage output!")
        print(f"  Examples: {lora_keys[:3]}")

    return checkpoint


def check_prompt_template():
    """Shows the exact prompt that the model sees during inference."""
    print(f"\n{'='*60}")
    print(f"  2. PROMPT TEMPLATE CHECK")
    print(f"{'='*60}")

    from kanha.prompting.builder import PromptBuilder

    builder = PromptBuilder(include_rules=False)
    prompt = builder.build(instruction="What are the three primary colors?")

    print(f"  What the model sees during inference:")
    print(f"  {'─'*50}")
    print(f"  {repr(prompt)}")
    print(f"  {'─'*50}")
    print()

    # Training format
    training_text = builder.build_training_pair(
        "What are the three primary colors?",
        "The three primary colors are red, blue, and yellow."
    )
    print(f"  What the model sees during SFT training:")
    print(f"  {'─'*50}")
    print(f"  {repr(training_text)}")
    print(f"  {'─'*50}")

    # Check consistency
    # The inference prompt should be a prefix of the training text
    inference_prefix = prompt.split("### Response:\n")[-1]  # empty after Response:
    if "### Instruction:" in prompt and "### Response:" in prompt:
        print(f"\n  PASS: Template has correct ### Instruction: and ### Response: markers")
    else:
        print(f"\n  FAIL: Template missing markers!")


def check_token_distribution(model_path, tokenizer_path=None):
    """Generates tokens and checks for collapsed distribution."""
    print(f"\n{'='*60}")
    print(f"  3. TOKEN DISTRIBUTION CHECK")
    print(f"{'='*60}")

    try:
        from kanha.core.model import KanhaModel
        from kanha.core.tokenizer import KanhaTokenizer
        from kanha.core.generation import generate

        model = KanhaModel.from_pretrained(model_path)
        tokenizer = KanhaTokenizer(tokenizer_path)

        # Generate with the CORRECT template
        from kanha.prompting.builder import PromptBuilder
        builder = PromptBuilder(include_rules=False)
        prompt = builder.build(instruction="Hello, how are you?")

        print(f"  Generating with proper template prompt...")
        print(f"  Prompt: {repr(prompt[:100])}...")

        response = generate(
            model, tokenizer, prompt,
            max_new_tokens=50,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
        )

        print(f"  Response: {repr(response[:200])}")

        # Check for repetitive characters
        if len(set(response)) < 5 and len(response) > 10:
            print(f"  WARNING: Response has very low character diversity ({len(set(response))} unique chars)")
            print(f"  This indicates the model's output distribution has collapsed.")
            print(f"  Likely causes:")
            print(f"    - SFT learning rate was too high (catastrophic forgetting)")
            print(f"    - Checkpoint was saved/loaded incorrectly")
            print(f"    - Prompt template mismatch between training and inference")
        else:
            print(f"  PASS: Response has diverse characters")

    except Exception as e:
        print(f"  Could not run generation test: {e}")


def check_weight_tying(model_path):
    """Checks that embedding and lm_head weights are properly tied."""
    print(f"\n{'='*60}")
    print(f"  4. WEIGHT TYING CHECK")
    print(f"{'='*60}")

    checkpoint = torch.load(model_path, map_location="cpu")
    state = checkpoint.get("model_state", checkpoint)

    emb_key = "token_emb.weight"
    lm_key = "lm_head.weight"

    if emb_key in state and lm_key in state:
        emb = state[emb_key]
        lm = state[lm_key]
        if torch.equal(emb, lm):
            print(f"  PASS: Embedding and LM head weights are identical (tied)")
        else:
            diff = (emb - lm).abs().mean().item()
            print(f"  WARNING: Weights differ by {diff:.6f}")
            print(f"  If intentional (untied after training), this is OK.")
            print(f"  If unintentional, weight tying may have been broken during saving.")
    else:
        print(f"  Could not find both '{emb_key}' and '{lm_key}' in checkpoint")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KANHA Model Diagnostic Tool")
    parser.add_argument("--model", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--tokenizer", default=None, help="Path to tokenizer.model")
    args = parser.parse_args()

    check_checkpoint(args.model)
    check_prompt_template()
    check_weight_tying(args.model)
    check_token_distribution(args.model, args.tokenizer)

    print(f"\n{'='*60}")
    print(f"  DIAGNOSIS COMPLETE")
    print(f"{'='*60}")
    print(f"\n  If you see garbage output, the most likely fixes are:")
    print(f"  1. Re-run SFT with the corrected training script (kanha/finetune/sft_train.py)")
    print(f"     - Uses matching prompt template")
    print(f"     - Masks instruction tokens in loss")
    print(f"     - Uses low learning rate (3e-5)")
    print(f"  2. Re-run DPO on top of the corrected SFT model")
    print(f"  3. Check that the tokenizer.model file matches what was used during pre-training")
    print()
