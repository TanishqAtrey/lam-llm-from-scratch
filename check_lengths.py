import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Check token lengths of instruction/response JSONL dataset.")
    parser.add_argument("--data", default="data/processed/sft_combined.jsonl",
                        help="Path to JSONL dataset (default: data/processed/sft_combined.jsonl)")
    parser.add_argument("--tokenizer", default=None,
                        help="Path to tokenizer model prefix/file (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: dataset file not found at '{args.data}'", file=sys.stderr)
        sys.exit(1)

    try:
        from kanha.core.tokenizer import KanhaTokenizer
    except ImportError:
        print("Error: sentencepiece is required. Run: pip install sentencepiece", file=sys.stderr)
        sys.exit(1)

    from kanha.prompting.builder import PromptBuilder
    builder = PromptBuilder(include_rules=False)
    tok = KanhaTokenizer(model_path=args.tokenizer) if args.tokenizer else KanhaTokenizer()
    lengths = []

    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            instr = s.get("instruction", s.get("prompt", ""))
            resp = s.get("response", s.get("completion", ""))
            p_prefix, _ = builder.build_prompt_prefix(instr)
            p_ids = tok.encode(p_prefix, add_bos=True, add_eos=False)
            r_ids = tok.encode(resp, add_bos=False, add_eos=True)
            lengths.append(len(p_ids) + len(r_ids))

    total = len(lengths)
    if total == 0:
        print(f"Dataset '{args.data}' is empty.")
        return

    lengths.sort()

    print(f"Dataset: {args.data}")
    print(f"Total samples: {total:,}")
    print(f"Min:  {lengths[0]}")
    print(f"p50:  {lengths[total // 2]}")
    print(f"p75:  {lengths[int(total * 0.75)]}")
    print(f"p90:  {lengths[int(total * 0.90)]}")
    print(f"p95:  {lengths[int(total * 0.95)]}")
    print(f"p99:  {lengths[int(total * 0.99)]}")
    print(f"Max:  {lengths[-1]}")

    under_256 = sum(1 for l in lengths if l <= 256)
    under_384 = sum(1 for l in lengths if l <= 384)
    under_512 = sum(1 for l in lengths if l <= 512)

    print(f"\n<= 256 tokens: {under_256:,} ({100*under_256/total:.1f}%)")
    print(f"<= 384 tokens: {under_384:,} ({100*under_384/total:.1f}%)")
    print(f"<= 512 tokens: {under_512:,} ({100*under_512/total:.1f}%)")


if __name__ == "__main__":
    main()
