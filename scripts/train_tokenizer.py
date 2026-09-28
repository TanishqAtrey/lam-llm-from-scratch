"""
scripts/train_tokenizer.py
Trains a SentencePiece BPE tokenizer on raw text files.

SentencePiece samples up to --sample_size sentences from your data
so it never loads the full 2.5GB into RAM — safe on 8GB machines.

Run:
    python scripts/train_tokenizer.py \
        --input  data/raw/ \
        --output models/tokenizer/tokenizer \
        --vocab  16000
"""

import os
import argparse
from pathlib import Path
from kanha.core.tokenizer import train_tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KANHA tokenizer")
    parser.add_argument("--input",       required=True,                help="Dir of .txt files or single file")
    parser.add_argument("--output",      default="models/tokenizer/tokenizer", help="Output model prefix")
    parser.add_argument("--vocab",       type=int, default=16000,      help="Vocabulary size")
    parser.add_argument("--sample_size", type=int, default=5_000_000,  help="Max sentences to sample (memory cap)")
    parser.add_argument("--threads",     type=int, default=8,          help="CPU threads")
    args = parser.parse_args()

    # Collect .txt files
    if os.path.isdir(args.input):
        files = sorted([
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.endswith(".txt")
        ])
        if not files:
            print(f"No .txt files found in {args.input}")
            exit(1)
        input_arg = files
    else:
        input_arg = args.input

    print(f"\nTraining tokenizer")
    print(f"  Input    : {[Path(f).name for f in input_arg] if isinstance(input_arg, list) else input_arg}")
    print(f"  Vocab    : {args.vocab:,}")
    print(f"  Samples  : {args.sample_size:,}")
    print(f"  Threads  : {args.threads}")
    print(f"  Output   : {args.output}.model\n")

    train_tokenizer(
        input_files=input_arg,
        model_prefix=args.output,
        vocab_size=args.vocab,
        sample_size=args.sample_size,
        num_threads=args.threads,
    )

    print(f"\nDone. Tokenizer saved to {args.output}.model")
    print(f"\nNext step:")
    print(f"  python scripts/preprocess_data.py \\")
    print(f"      --input data/raw/ \\")
    print(f"      --output data/processed/train.npy \\")
    print(f"      --tokenizer {args.output}.model")
