"""
kanha/finetune/sft_train.py
Supervised Fine-Tuning (SFT) for KANHA.

CRITICAL DETAILS that prevent garbage output:
  1. Uses the EXACT same prompt template as inference (PromptBuilder)
  2. Masks instruction tokens in loss — model only learns to generate responses
  3. Uses a LOW learning rate (1e-5 to 5e-5) to avoid catastrophic forgetting
  4. Saves checkpoint in the same format as KanhaModel.save_pretrained()

Common mistakes that cause garbage after SFT:
  - Using a different prompt template during training vs inference
  - Not masking instruction tokens (forces model to memorize template verbatim)
  - Learning rate too high (> 1e-4 for full fine-tuning destroys pretrained weights)
  - Training for too many epochs without early stopping

Run:
    python main.py finetune \
        --base_model models/base/final_model.pt \
        --data data/processed/sft_combined.jsonl \
        --output models/finetuned/ \
        --epochs 3 \
        --lr 3e-5
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from kanha.core.model import KanhaModel
from kanha.core.tokenizer import KanhaTokenizer
from kanha.prompting.builder import PromptBuilder
from kanha.utils.config import cfg
from kanha.utils.helpers import get_device, ensure_dir
from kanha.utils.logging import get_logger

log = get_logger("sft")


# ── Dataset ───────────────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    """
    Loads JSONL with {"instruction": ..., "response": ...} format.

    CRITICAL: Uses PromptBuilder.build_training_pair() to format data
    with the SAME template that inference uses. This prevents the #1
    cause of garbage output (template mismatch).

    Also computes a loss mask that excludes instruction tokens —
    the model should only be trained to GENERATE the response, not
    memorize the instruction template.
    """

    def __init__(self, data_path: str, tokenizer: KanhaTokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.prompt_builder = PromptBuilder(include_rules=False)
        self.samples = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("instruction") and obj.get("response"):
                    self.samples.append(obj)

        log.info(f"Loaded {len(self.samples):,} SFT samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        instruction = sample["instruction"]
        response = sample["response"]

        # ── Build full training sequence using the CANONICAL template ──
        # This is "### Instruction:\n{instruction}\n\n### Response:\n{response}"
        full_text = self.prompt_builder.build_training_pair(instruction, response)

        # ── Tokenize ──
        full_ids = self.tokenizer.encode(
            full_text, add_bos=True, add_eos=True, max_length=self.max_len
        )

        # ── Compute instruction length for loss masking ──
        # Tokenize just the instruction part (everything before the response)
        instruction_prefix = f"### Instruction:\n{instruction}\n\n### Response:\n"
        instruction_ids = self.tokenizer.encode(
            instruction_prefix, add_bos=True, add_eos=False
        )
        mask_len = len(instruction_ids)

        # ── Create shifted input/target pairs ──
        # Standard causal LM: input = tokens[:-1], target = tokens[1:]
        input_ids = full_ids[:-1]
        labels = full_ids[1:]

        # ── Mask instruction tokens in labels ──
        # Set to -100 so cross_entropy ignores them
        for i in range(min(mask_len - 1, len(labels))):
            labels[i] = -100

        # ── Pad to max_len - 1 (since we removed one token by shifting) ──
        seq_len = self.max_len - 1
        pad_len = seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.PAD_ID] * pad_len
            labels = labels + [-100] * pad_len
        else:
            input_ids = input_ids[:seq_len]
            labels = labels[:seq_len]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ── Training loop ─────────────────────────────────────────────────────────────

def sft_train(args):
    """
    Main SFT training function. Called from main.py finetune command.
    """
    device = get_device()

    # ── Load base model ──
    log.info(f"Loading base model from {args.base_model}")
    model = KanhaModel.from_pretrained(args.base_model)
    model.train()
    model.to(device)

    # ── Load tokenizer ──
    tokenizer = KanhaTokenizer()

    # ── Dataset + DataLoader ──
    dataset = SFTDataset(args.data, tokenizer, max_len=cfg.model.max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    # ── Optimizer ──
    # CRITICAL: Use a LOW learning rate for SFT to avoid catastrophic forgetting.
    # For a 40-60M model with full fine-tuning:
    #   - 1e-5 to 5e-5 is safe
    #   - 1e-4 is risky (may destroy pretrained weights)
    #   - 3e-4 or higher WILL cause garbage output
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.95),
    )

    # ── Warmup + cosine schedule ──
    total_steps = len(dataloader) * args.epochs
    warmup_steps = min(100, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.1 + 0.9 * (1 + __import__("math").cos(3.14159 * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training ──
    ensure_dir(args.output)
    global_step = 0
    best_loss = float("inf")

    log.info(f"SFT Training | epochs={args.epochs} | lr={args.lr} | "
             f"batch_size={args.batch_size} | samples={len(dataset):,}")
    log.info(f"Total steps: {total_steps:,} | Warmup: {warmup_steps}")

    for epoch in range(args.epochs):
        epoch_loss = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            logits, loss, _ = model(input_ids, targets=labels)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        avg_loss = epoch_loss / max(n_batches, 1)
        log.info(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

        # Save checkpoint each epoch
        ckpt_path = os.path.join(args.output, f"sft_epoch{epoch+1}.pt")
        model.save_pretrained(ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.output, "sft_final.pt")
            model.save_pretrained(best_path)
            log.info(f"Best model saved to {best_path} (loss={best_loss:.4f})")

    log.info("SFT training complete!")
    log.info(f"Best model: {args.output}/sft_final.pt")
