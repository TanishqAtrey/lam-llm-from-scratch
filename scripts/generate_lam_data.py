#!/usr/bin/env python
"""
scripts/generate_lam_data.py
Standalone script to generate LAM training data.

Run:
    python scripts/generate_lam_data.py \\
        --output data/lam/trajectories.jsonl \\
        --count 100000 \\
        --seed 42
"""

import argparse
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kanha.lam.data.generator import TrajectoryGenerator, split_data
from kanha.utils.helpers import ensure_dir


def main():
    parser = argparse.ArgumentParser(description="Generate LAM training data")
    parser.add_argument("--output", default="data/lam/trajectories.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--count", type=int, default=100000,
                        help="Number of trajectories to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--split", action="store_true",
                        help="Also split into train/val/test")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()

    ensure_dir(os.path.dirname(args.output))

    print(f"Generating {args.count:,} trajectories...")
    gen = TrajectoryGenerator(
        num_trajectories=args.count,
        seed=args.seed,
    )
    stats = gen.generate(args.output)

    print(f"\nDone! Generated {stats['total']:,} trajectories to {args.output}")
    for tier, count in sorted(stats.get("per_tier", {}).items()):
        print(f"  {tier}: {count:,}")

    if args.split:
        base = os.path.splitext(args.output)[0]
        train_path = f"{base}_train.jsonl"
        val_path = f"{base}_val.jsonl"
        test_path = f"{base}_test.jsonl"

        print(f"\nSplitting into train/val/test...")
        split_data(
            args.output, train_path, val_path, test_path,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        for label, path in [("Train", train_path), ("Val", val_path), ("Test", test_path)]:
            with open(path) as f:
                n = sum(1 for line in f if line.strip())
            print(f"  {label}: {n:,} ({path})")

    print("\nNext steps:")
    print(f"  python main.py lam-train \\")
    print(f"      --base_model models/base/final_model.pt \\")
    print(f"      --data {args.output} \\")
    print(f"      --output models/lam/")


if __name__ == "__main__":
    main()
