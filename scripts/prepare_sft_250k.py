"""
scripts/prepare_sft_250k.py
Curates and builds the canonical 250K SFT dataset mix for KANHA:

Dataset Mix (250,000 Total):
  1. Smol-SmolTalk (HuggingFaceTB/smol-smoltalk)            : 229,000 (91.6%)
  2. Python Code Instructions (iamtarun/python_code_instructions_18k_alpaca) : 18,000  (7.2%)
  3. Everyday Conversations (HuggingFaceTB/everyday-conversations-llama3.1-2k):  2,000  (0.8%)
  4. Custom Identity Data (KANHA assistant / creator Tanishq Atrey)          :  1,000  (0.4%)

Validation Set:
  - 1,000 held-out samples from Smol-SmolTalk test split -> data/processed/sft_val.jsonl

Filtering:
  - Context window limit: <= 1024 tokens (KanhaTokenizer)
  - Drop empty messages and whitespace-only turns
  - Deduplicate on first user prompt
  - Source stratification preserved across Smol-SmolTalk subsets

Run:
  python scripts/prepare_sft_250k.py [--output data/processed/sft_250k.jsonl] [--val_output data/processed/sft_val.jsonl]
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kanha.core.tokenizer import KanhaTokenizer
from kanha.prompting.builder import PromptBuilder
from kanha.utils.logging import get_logger

log = get_logger("prepare_sft_250k")


# ─── Identity Data Generator ──────────────────────────────────────────────────

def generate_identity_dataset(tokenizer: KanhaTokenizer, target_count: int = 1000, seed: int = 42) -> list[dict]:
    """
    Generates 1,000 diverse, high-quality identity pairs for KANHA.
    Covering:
      - Name & Acronym: KANHA (Knowledge-Augmented Neural Heuristic Assistant)
      - Creator: Tanishq Atrey
      - Architecture: ~42M parameter decoder-only transformer (8 layers, 8 heads, 512 dim, 2048 ff_dim, 1024 context)
      - Training: Pretrained on ~3.0B tokens, followed by SFT and DPO
      - Capabilities: Python programming, technical explanations, summarization, tool calling, RAG
      - Limitations: Compact model size, 1024 context window, honesty when uncertain
      - Distinction from other models: Explicitly denies being ChatGPT, Claude, LLaMA, or made by OpenAI/Google/Meta
      - Multi-turn dialogues with greetings, clarifications, and follow-ups
    """
    rng = random.Random(seed)
    samples = []
    seen_prompts = set()

    # 1. Creator & Origins
    creators_qa = [
        (
            ["Who created you?", "Who is your creator?", "Who made you?", "Who built KANHA?",
             "Who programmed you?", "Who developed this AI?", "Who is the author of KANHA?",
             "Who trained you?", "Whose creation are you?", "Who is behind KANHA?"],
            ["I was created and trained from scratch by Tanishq Atrey.",
             "My creator is Tanishq Atrey, who designed my architecture and trained me from scratch.",
             "I was developed by Tanishq Atrey as a custom decoder-only transformer model.",
             "Tanishq Atrey built and trained me from scratch, from the SentencePiece tokenizer and pretraining to instruction tuning.",
             "I am KANHA, an AI assistant created by Tanishq Atrey."]
        ),
        (
            ["Were you made by OpenAI?", "Are you ChatGPT?", "Are you created by Google?", "Are you a Meta LLaMA model?",
             "Did Anthropic make you?", "Are you an OpenAI model?"],
            ["No, I was not created by OpenAI or any large tech corporation. I am KANHA, designed and trained from scratch by Tanishq Atrey.",
             "No, I am not ChatGPT. My name is KANHA, and I was created independently by Tanishq Atrey.",
             "No, I am KANHA. I am an independent open-weights transformer trained from scratch by Tanishq Atrey.",
             "No, I am neither ChatGPT nor LLaMA. I am KANHA, created and trained by Tanishq Atrey."]
        ),
    ]

    # 2. Name & Meaning
    name_qa = [
        (
            ["What is your name?", "Who are you?", "Identify yourself.", "May I know your name?",
             "Introduce yourself.", "What should I call you?", "What is your identity?"],
            ["I am KANHA, an AI assistant created to help you with programming, answering questions, and general problem solving.",
             "My name is KANHA (Knowledge-Augmented Neural Heuristic Assistant). How can I assist you today?",
             "I am KANHA, a lightweight and efficient language model created by Tanishq Atrey.",
             "I am KANHA, a 42M-parameter decoder-only transformer trained from scratch by Tanishq Atrey."]
        ),
        (
            ["What does KANHA stand for?", "What is the full form of KANHA?", "Is KANHA an acronym?",
             "Why are you named KANHA?", "Explain your name."],
            ["KANHA stands for Knowledge-Augmented Neural Heuristic Assistant.",
             "My name KANHA is an acronym for Knowledge-Augmented Neural Heuristic Assistant, reflecting my design for tool integration and retrieval.",
             "KANHA stands for Knowledge-Augmented Neural Heuristic Assistant, developed by Tanishq Atrey."]
        ),
    ]

    # 3. Technical Architecture & Specs
    architecture_qa = [
        (
            ["What is your architecture?", "How many parameters do you have?", "Tell me your model specifications.",
             "What are your technical specs?", "Can you explain your architecture?", "How big is your model?"],
            ["I am a ~42M parameter decoder-only transformer featuring 8 layers, 8 attention heads, an embedding dimension of 512, an MLP dimension of 2048, and a context window of 1024 tokens.",
             "Technically, I have 41,755,136 parameters across 8 transformer decoder layers, with 8 attention heads, RoPE position embeddings, and a vocabulary of 16,000 SentencePiece BPE tokens.",
             "My architecture is an 8-layer decoder-only causal transformer totaling ~42 million parameters with a 1024-token context length, created by Tanishq Atrey.",
             "I am built with 8 transformer layers, 8 multi-head attention heads, 512 hidden dimension, 2048 feed-forward dimension, and a 16,000-token BPE tokenizer, running with a 1024 context window."]
        ),
        (
            ["How were you trained?", "What was your training process?", "How did Tanishq Atrey train you?",
             "What datasets were you trained on?", "Can you explain your training stages?"],
            ["I was trained in three stages: first, unsupervised pretraining from scratch on ~3.0 billion tokens; second, Supervised Fine-Tuning (SFT) on curated instruction and dialogue datasets; and third, Direct Preference Optimization (DPO) for alignment.",
             "My training followed a full modern LLM pipeline: tokenizer training (16k BPE), causal pretraining on 3.0B tokens, supervised instruction tuning (SFT), and preference alignment (DPO), all conducted by Tanishq Atrey."]
        ),
        (
            ["What tokenizer do you use?", "What is your vocabulary size?"],
            ["I use a custom SentencePiece Byte-Pair Encoding (BPE) tokenizer with a vocabulary size of 16,000 tokens.",
             "My tokenizer is a 16,000-vocabulary SentencePiece BPE model with standard special tokens for padding, unknown tokens, start-of-sequence, and end-of-sequence."]
        ),
    ]

    # 4. Capabilities & Limitations
    capabilities_qa = [
        (
            ["What are your capabilities?", "What can you do?", "How can you help me?",
             "What tasks are you good at?", "What are your strengths?"],
            ["I can help you with Python programming, debugging code, explaining technical and mathematical concepts, answering questions, drafting text, and utilizing tools and RAG retrieval.",
             "My primary strengths are Python coding, concise reasoning, instruction following, and efficient execution within my 1024-token context window.",
             "I can assist with writing clean code, solving algorithmic problems, answering knowledge queries, and executing tool actions when connected to my inference engine."]
        ),
        (
            ["What are your limitations?", "What can you not do?", "What are your constraints?",
             "Where do you struggle?", "Are there things you cannot do?"],
            ["As a compact 42M-parameter model, my context window is 1024 tokens. I do not have live internet access without tools, and I strive to be honest and concise rather than fabricating answers.",
             "My constraints include a 1024-token context limit and a compact parameter scale. I am optimized for local efficiency, but for extremely deep specialized fields, I rely on retrieval (RAG) and tool execution.",
             "I cannot browse the live web without external tools, and I do not retain memory across sessions beyond my conversation context. When I don't know something, I will say so honestly."]
        ),
    ]

    # 5. Multi-Turn Identity Dialogues
    dialogue_templates = [
        [
            ("Hello! Who are you?", "Hello! I am KANHA, an AI assistant created by Tanishq Atrey. How can I help you today?"),
            ("What does your name mean?", "KANHA stands for Knowledge-Augmented Neural Heuristic Assistant."),
            ("Who built you?", "I was designed and trained from scratch by Tanishq Atrey."),
        ],
        [
            ("Hi there", "Hello! How can I assist you today?"),
            ("Are you ChatGPT?", "No, I am KANHA, an independent AI assistant developed from scratch by Tanishq Atrey."),
            ("What can you do?", "I can assist you with writing Python code, explaining technical topics, answering questions, and reasoning through problems."),
        ],
        [
            ("Hey KANHA", "Greetings! How can I help you today?"),
            ("Tell me about your architecture.", "I am an 8-layer, 8-head decoder-only transformer with ~42M parameters, 512 hidden dimension, and a 1024 context window."),
            ("Who is your creator?", "My creator is Tanishq Atrey."),
        ],
        [
            ("Can you help me write Python code?", "Yes! I can write, explain, and debug Python code for you. What would you like to build?"),
            ("Who trained you to write code?", "I was trained from scratch on diverse code and instruction data by Tanishq Atrey."),
        ],
        [
            ("What are your hardware requirements?", "Because I have ~42M parameters, I run very efficiently on consumer CPUs, Apple Silicon, and standard GPUs with under 2GB of memory."),
            ("Who developed this model?", "I was developed and trained by Tanishq Atrey."),
        ],
    ]

    greetings = [
        "", "Hello! ", "Hi KANHA, ", "Hey there, ", "Good day, ", "Excuse me, ",
        "Can you tell me, ", "I want to know: ", "Quick question: ", "Could you please explain: ",
        "Hey, ", "Hi, ", "Tell me, ", "I am curious: ", "Just wondering: ", "May I ask, ",
        "Pardon me, ", "Please answer: ", "Greetings! ", "Hello KANHA, ", "Hey assistant, ",
        "Hi! ", "Yo, ", "Dear KANHA, ", "I have a question: "
    ]
    suffixes = [
        "", " Thanks!", " Thank you.", " Please let me know.", " Be concise.",
        " Explain clearly.", " I appreciate your help.", " Briefly.", " Thanks in advance."
    ]

    single_categories = creators_qa + name_qa + architecture_qa + capabilities_qa
    candidates = []

    # Generate single-turn pool
    for g in greetings:
        for s in suffixes:
            for q_list, a_list in single_categories:
                for q in q_list:
                    full_q = f"{g}{q}{s}".strip()
                    norm_q = full_q.lower().strip()
                    if norm_q not in seen_prompts:
                        seen_prompts.add(norm_q)
                        candidates.append({"instruction": full_q, "response": rng.choice(a_list)})

    rng.shuffle(candidates)
    samples.extend(candidates[: int(target_count * 0.85)])

    # Generate multi-turn dialogues for the remaining ~15%
    for dt in dialogue_templates:
        for num_turns in range(2, len(dt) + 1):
            sub_turns = dt[:num_turns]
            hist_parts = [f"User: {u}\nAssistant: {a}" for u, a in sub_turns[:-1]]
            last_u, last_a = sub_turns[-1]
            inst = f"{chr(10).join(hist_parts)}\nUser: {last_u}"
            norm_p = inst.lower().strip()
            if norm_p not in seen_prompts:
                seen_prompts.add(norm_p)
                samples.append({"instruction": inst, "response": last_a})

    # If any remaining needed, draw from candidates
    if len(samples) < target_count:
        for c in candidates[int(target_count * 0.85):]:
            samples.append(c)
            if len(samples) >= target_count:
                break

    rng.shuffle(samples)
    final_samples = samples[:target_count]
    log.info(f"Generated {len(final_samples)} identity samples.")
    return final_samples


# ─── Dataset Processors ────────────────────────────────────────────────────────

def format_messages_to_pair(messages: list[dict]) -> tuple[str, str]:
    """
    Converts a standard chat messages list [{'role': ..., 'content': ...}] into (instruction, response).
    Single-turn: instruction is user turn, response is assistant turn.
    Multi-turn: instruction is 'User: ...\nAssistant: ...\nUser: ...', response is final assistant turn.
    System message (if present at index 0) is prepended to the first user turn.
    """
    if not messages:
        return "", ""

    system_prefix = ""
    clean_messages = []
    for m in messages:
        role = m.get("role", "").lower().strip()
        content = m.get("content", "").strip()
        if not content:
            continue
        if role == "system" and not clean_messages:
            system_prefix = content
        elif role in ("user", "human"):
            if system_prefix and not clean_messages:
                content = f"{system_prefix}\n\n{content}"
                system_prefix = ""
            clean_messages.append(("User", content))
        elif role in ("assistant", "ai", "bot"):
            clean_messages.append(("Assistant", content))

    if not clean_messages:
        return "", ""

    # Ensure conversation ends with an Assistant response
    if clean_messages[-1][0] != "Assistant":
        clean_messages.pop()

    if not clean_messages:
        return "", ""

    # Find the last assistant turn
    final_resp = clean_messages[-1][1]
    prior_turns = clean_messages[:-1]

    if not prior_turns:
        return "", ""

    # If only 1 user turn + 1 assistant turn: return pure instruction text without "User:" prefix
    if len(prior_turns) == 1 and prior_turns[0][0] == "User":
        return prior_turns[0][1], final_resp

    # Multi-turn dialogue: format prior turns as 'User: ...\nAssistant: ...'
    dialogue_lines = [f"{role}: {text}" for role, text in prior_turns]
    instruction = "\n".join(dialogue_lines)
    return instruction, final_resp


def is_valid_sample(instruction: str, response: str, tokenizer: KanhaTokenizer, max_len: int = 1024) -> bool:
    """Checks non-empty content and ensures total tokens fit within max_len (1024)."""
    if not instruction or not response:
        return False
    inst_clean = instruction.strip()
    resp_clean = response.strip()
    if not inst_clean or not resp_clean:
        return False

    builder = PromptBuilder(include_rules=False)
    prefix, _ = builder.build_prompt_prefix(inst_clean)
    p_ids = tokenizer.encode(prefix, add_bos=True, add_eos=False)
    r_ids = tokenizer.encode(resp_clean, add_bos=False, add_eos=True)
    total_len = len(p_ids) + len(r_ids)
    return 1 < total_len <= max_len


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def prepare_sft_dataset(
    output_path: str = "data/processed/sft_250k.jsonl",
    val_output_path: str = "data/processed/sft_val.jsonl",
    seed: int = 42,
):
    from datasets import load_dataset

    log.info("=" * 65)
    log.info("  STARTING KANHA 250K SFT DATASET PREPARATION")
    log.info("=" * 65)

    tokenizer = KanhaTokenizer()
    rng = random.Random(seed)
    seen_prompts = set()

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Validation Set (1,000 held out from Smol-SmolTalk test split)
    # ──────────────────────────────────────────────────────────────────────────
    log.info("Step 1: Preparing 1,000 validation samples from Smol-SmolTalk test split...")
    ds_test = load_dataset("HuggingFaceTB/smol-smoltalk", split="test")
    val_samples = []
    test_indices = list(range(len(ds_test)))
    rng.shuffle(test_indices)

    for idx in test_indices:
        row = ds_test[idx]
        inst, resp = format_messages_to_pair(row.get("messages", []))
        if is_valid_sample(inst, resp, tokenizer, max_len=1024):
            val_samples.append({"instruction": inst, "response": resp, "source": "smol-smoltalk-val"})
            seen_prompts.add(inst.lower().strip())
            if len(val_samples) >= 1000:
                break

    log.info(f"Held out {len(val_samples):,} validation samples.")
    os.makedirs(os.path.dirname(val_output_path) or ".", exist_ok=True)
    with open(val_output_path, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s) + "\n")
    log.info(f"Saved validation dataset to {val_output_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Python Code Instructions (18,000 samples)
    # ──────────────────────────────────────────────────────────────────────────
    log.info("Step 2: Processing iamtarun/python_code_instructions_18k_alpaca...")
    ds_code = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
    code_samples = []
    code_indices = list(range(len(ds_code)))
    rng.shuffle(code_indices)

    for idx in code_indices:
        row = ds_code[idx]
        inst = row.get("instruction", "").strip()
        inp = str(row.get("input", "")).strip() if row.get("input") else ""
        if inp:
            inst = f"{inst}\n\nInput:\n{inp}"
        resp = row.get("output", "").strip()

        norm_p = inst.lower().strip()
        if norm_p not in seen_prompts and is_valid_sample(inst, resp, tokenizer, max_len=1024):
            seen_prompts.add(norm_p)
            code_samples.append({"instruction": inst, "response": resp, "source": "python_code_18k"})
            if len(code_samples) >= 18000:
                break

    log.info(f"Collected {len(code_samples):,} Python Code Instruction samples.")

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Everyday Conversations (2,000 samples, 1 copy per conditional check)
    # ──────────────────────────────────────────────────────────────────────────
    log.info("Step 3: Processing HuggingFaceTB/everyday-conversations-llama3.1-2k...")
    ds_everyday = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", split="train_sft")
    everyday_samples = []
    everyday_indices = list(range(len(ds_everyday)))
    rng.shuffle(everyday_indices)

    for idx in everyday_indices:
        row = ds_everyday[idx]
        inst, resp = format_messages_to_pair(row.get("messages", []))
        norm_p = inst.lower().strip()
        if norm_p not in seen_prompts and is_valid_sample(inst, resp, tokenizer, max_len=1024):
            seen_prompts.add(norm_p)
            everyday_samples.append({"instruction": inst, "response": resp, "source": "everyday-conversations"})
            if len(everyday_samples) >= 2000:
                break

    log.info(f"Collected {len(everyday_samples):,} Everyday Conversations samples.")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Identity Dataset (1,000 samples)
    # ──────────────────────────────────────────────────────────────────────────
    log.info("Step 4: Generating 1,000 Identity samples...")
    identity_raw = generate_identity_dataset(tokenizer, target_count=1000, seed=seed)
    identity_samples = []
    for s in identity_raw:
        inst = s["instruction"]
        resp = s["response"]
        norm_p = inst.lower().strip()
        if norm_p not in seen_prompts and is_valid_sample(inst, resp, tokenizer, max_len=1024):
            seen_prompts.add(norm_p)
            identity_samples.append({"instruction": inst, "response": resp, "source": "kanha-identity"})

    # Ensure exactly 1,000 identity samples
    if len(identity_samples) < 1000:
        extra_needed = 1000 - len(identity_samples)
        for s in identity_raw:
            if extra_needed <= 0:
                break
            identity_samples.append({"instruction": s["instruction"], "response": s["response"], "source": "kanha-identity"})
            extra_needed -= 1

    log.info(f"Collected {len(identity_samples):,} Identity samples.")

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Smol-SmolTalk Core Chat (229,000 samples with preserved source proportions)
    # ──────────────────────────────────────────────────────────────────────────
    target_smol_count = 250000 - len(code_samples) - len(everyday_samples) - len(identity_samples)
    log.info(f"Step 5: Sampling {target_smol_count:,} stratified samples from HuggingFaceTB/smol-smoltalk train split...")

    ds_smol = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
    log.info(f"Total Smol-SmolTalk train rows: {len(ds_smol):,}")

    # Group valid indices by source
    by_source = defaultdict(list)
    smol_indices = list(range(len(ds_smol)))
    rng.shuffle(smol_indices)

    log.info("Scanning and filtering Smol-SmolTalk candidates...")
    total_collected = 0
    pool_target = target_smol_count + 35000

    for idx in smol_indices:
        row = ds_smol[idx]
        src = row.get("source", "other")
        inst, resp = format_messages_to_pair(row.get("messages", []))
        norm_p = inst.lower().strip()
        if norm_p not in seen_prompts and is_valid_sample(inst, resp, tokenizer, max_len=1024):
            seen_prompts.add(norm_p)
            by_source[src].append({"instruction": inst, "response": resp, "source": f"smoltalk-{src}"})
            total_collected += 1
            if total_collected >= pool_target:
                break

    log.info(f"Collected {total_collected:,} valid candidates across {len(by_source)} sources.")

    # Stratified selection to maintain original source proportions
    smol_selected = []
    remaining_targets = {}
    total_pool = total_collected

    for src, items in by_source.items():
        proportion = len(items) / total_pool
        quota = int(target_smol_count * proportion)
        quota = min(quota, len(items))
        smol_selected.extend(items[:quota])
        remaining_targets[src] = items[quota:]

    # Fill any remaining deficit due to integer rounding
    deficit = target_smol_count - len(smol_selected)
    if deficit > 0:
        log.info(f"Topping up deficit of {deficit:,} samples...")
        for src, leftover in remaining_targets.items():
            take = min(deficit, len(leftover))
            smol_selected.extend(leftover[:take])
            deficit -= take
            if deficit <= 0:
                break

    # If any slight surplus due to rounding, clamp to exact target
    smol_selected = smol_selected[:target_smol_count]
    log.info(f"Selected exactly {len(smol_selected):,} Smol-SmolTalk samples.")

    # ──────────────────────────────────────────────────────────────────────────
    # 6. Combine, Final Validation, and Shuffle
    # ──────────────────────────────────────────────────────────────────────────
    log.info("Step 6: Combining and shuffling final 250K dataset...")
    full_mix = code_samples + everyday_samples + identity_samples + smol_selected
    rng.shuffle(full_mix)

    log.info(f"Total dataset size: {len(full_mix):,} examples")
    source_counts = Counter(x["source"] for x in full_mix)
    for src, cnt in source_counts.most_common(12):
        log.info(f"  {src}: {cnt:,} ({cnt/len(full_mix)*100:.2f}%)")

    # Save to disk
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in full_mix:
            f.write(json.dumps(ex) + "\n")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log.info(f"Successfully saved {len(full_mix):,} samples to {output_path} ({file_size_mb:.1f} MB)")
    log.info("=" * 65)
    log.info("  250K SFT DATASET PREPARATION COMPLETE!")
    log.info("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare KANHA 250K SFT dataset mix")
    parser.add_argument("--output", default="data/processed/sft_250k.jsonl", help="Output train JSONL path")
    parser.add_argument("--val_output", default="data/processed/sft_val.jsonl", help="Output val JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and shuffling")
    args = parser.parse_args()

    prepare_sft_dataset(
        output_path=args.output,
        val_output_path=args.val_output,
        seed=args.seed,
    )
