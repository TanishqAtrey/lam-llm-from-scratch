"""
scripts/test_base_model.py
Quick test to check if your base pretrained model is good enough for SFT.

Tests the model with RAW text prompts (no instruction template) —
this is what the base model was actually trained on.

Run:
    python scripts/test_base_model.py --model models/base/final_model.pt
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from kanha.core.model import KanhaModel
from kanha.core.tokenizer import KanhaTokenizer
from kanha.core.generation import generate
from kanha.utils.helpers import get_device


def test_base_model(model_path, tokenizer_path=None):
    print(f"\n{'='*60}")
    print(f"  BASE MODEL QUALITY CHECK")
    print(f"  Model: {model_path}")
    print(f"{'='*60}\n")

    model = KanhaModel.from_pretrained(model_path)
    tokenizer = KanhaTokenizer(tokenizer_path)
    device = get_device()

    # Raw text prompts — the kind of text the base model was trained on
    # NO instruction template — just continue the text
    prompts = [
        "Once upon a time, there was a",
        "The capital of France is",
        "Python is a programming language that",
        "In the morning, the sun rises and",
        "The three primary colors are",
    ]

    print("Testing with RAW text prompts (no template):")
    print("The base model should continue these naturally.\n")

    for prompt in prompts:
        print(f"  Prompt: \"{prompt}\"")

        # Generate WITHOUT the instruction template
        # Just raw text continuation — what the base model was trained for
        response = generate(
            model, tokenizer, prompt,
            max_new_tokens=60,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            device=device,
        )

        print(f"  Output: \"{response[:200]}\"")

        # Quality checks
        unique_chars = len(set(response))
        has_spaces = " " in response
        is_repetitive = any(response.count(response[i:i+10]) > 3
                           for i in range(min(5, len(response)))
                           if len(response[i:i+10]) == 10)

        issues = []
        if unique_chars < 10 and len(response) > 20:
            issues.append("very low character diversity")
        if not has_spaces and len(response) > 10:
            issues.append("no spaces (tokenizer or model issue)")
        if is_repetitive:
            issues.append("heavy repetition")

        if issues:
            print(f"  Issues: {', '.join(issues)}")
        else:
            print(f"  Quality: OK")
        print()

    print(f"{'='*60}")
    print(f"  INTERPRETATION:")
    print(f"{'='*60}")
    print(f"""
  IF outputs are semi-coherent English with spaces:
    → Base model is GOOD. Proceed with SFT:
      python main.py finetune --base_model {model_path} \\
          --data data/processed/sft_combined.jsonl --epochs 2

  IF outputs have no spaces / pure repetition / garbage:
    → Base model needs MORE pretraining. Options:
      a) Train for more steps (increase max_steps in config.yaml)
      b) Check your tokenizer (run: python scripts/test_tokenizer.py)
      c) Lower learning rate or increase warmup_steps

  IF outputs are mostly symbols/numbers:
    → Tokenizer mismatch. Make sure you're using the SAME
      tokenizer.model file that was used during pretraining.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    args = parser.parse_args()
    test_base_model(args.model, args.tokenizer)
