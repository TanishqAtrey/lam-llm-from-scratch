import json
from kanha.core.tokenizer import KanhaTokenizer

tok = KanhaTokenizer()
lengths = []

with open("data/processed/sft_combined.jsonl", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        s = json.loads(line)
        instr_len = len(tok.encode(s["instruction"]))
        resp_len  = len(tok.encode(s["response"]))
        lengths.append(instr_len + resp_len)

lengths.sort()
total = len(lengths)

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
