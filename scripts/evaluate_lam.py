#!/usr/bin/env python
"""
scripts/evaluate_lam.py
Standalone script to evaluate a trained LAM agent.

Run:
    python scripts/evaluate_lam.py \\
        --model models/lam/lam_final.pt \\
        --test_data data/lam/test.jsonl \\
        --output results/lam_eval.json
"""

import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kanha.core.model import KanhaModel
from kanha.core.tokenizer import KanhaTokenizer
from kanha.lam.evaluator import LAMEvaluator
from kanha.utils.helpers import ensure_dir


def main():
    parser = argparse.ArgumentParser(description="Evaluate LAM agent")
    parser.add_argument("--model", required=True,
                        help="Path to trained LAM model (.pt)")
    parser.add_argument("--test_data", required=True,
                        help="Path to test JSONL file")
    parser.add_argument("--output", default="results/lam_eval.json",
                        help="Where to save evaluation report")
    args = parser.parse_args()

    print("Loading model...")
    model = KanhaModel.from_pretrained(args.model)
    tokenizer = KanhaTokenizer()

    print(f"Loading test data from {args.test_data}...")
    evaluator = LAMEvaluator(model, tokenizer, args.test_data)

    print("Running evaluation...")
    report = evaluator.evaluate()

    # Print summary
    print("\n" + "=" * 60)
    print(report.summary())
    print("=" * 60)

    # Save detailed results
    ensure_dir(os.path.dirname(args.output))
    with open(args.output, "w") as f:
        json.dump(dataclasses.asdict(report), f, indent=2, default=str)
    print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
