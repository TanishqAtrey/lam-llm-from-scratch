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
import math
import torch
import torch.utils.data
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

        # Safety guard: ensure at least one supervised token remains to prevent NaN loss
        if not any(l != -100 for l in labels) and len(labels) > 0:
            labels[-1] = full_ids[-1]

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


# ── Validation helper ─────────────────────────────────────────────────────────

@torch.no_grad()
def _eval_val_loss(model, val_loader, device):
    """Computes average loss over the entire validation set."""
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        if not (labels != -100).any():
            continue
        _, loss, _ = model(input_ids, targets=labels)
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


# ── Training loop ─────────────────────────────────────────────────────────────

def sft_train(args):
    """
    Main SFT training function. Called from main.py finetune command.

    Includes validation loss tracking and early stopping:
      --val_fraction : held-out fraction (default 0.1)
      --patience     : epochs without val improvement before stopping (default 3)
    """
    device = get_device()

    val_fraction = getattr(args, "val_fraction", None) or getattr(cfg.training, "val_fraction", 0.1)
    patience = getattr(args, "patience", None) or getattr(cfg.training, "sft_patience", 3)

    # ── Load base model ──
    log.info(f"Loading base model from {args.base_model}")
    model = KanhaModel.from_pretrained(args.base_model)

    if getattr(args, "lora", False):
        from kanha.core.lora import LoRAModel
        log.info("Wrapping model in LoRAModel adapter")
        model = LoRAModel(
            model,
            rank=getattr(cfg.lora, "rank", 8),
            alpha=getattr(cfg.lora, "alpha", 16.0),
            dropout=getattr(cfg.lora, "dropout", 0.05),
            target_modules=getattr(cfg.lora, "target_modules", ["q_proj", "v_proj"]),
        )
        model.print_trainable_params()

    model.train()
    model.to(device)

    # ── Load tokenizer ──
    tokenizer = KanhaTokenizer()

    # ── Dataset with train/val split ──
    full_dataset = SFTDataset(args.data, tokenizer, max_len=cfg.model.max_seq_len)
    val_size = max(1, int(len(full_dataset) * val_fraction))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, drop_last=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, drop_last=False, num_workers=0,
    )

    log.info(f"Split: {train_size:,} train / {val_size:,} val samples")

    # ── Optimizer ──
    # CRITICAL: Use a LOW learning rate for SFT to avoid catastrophic forgetting.
    # For a 40-60M model with full fine-tuning:
    #   - 1e-5 to 5e-5 is safe
    #   - 1e-4 is risky (may destroy pretrained weights)
    #   - 3e-4 or higher WILL cause garbage output
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.95),
    )

    # ── Warmup + cosine schedule ──
    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(100, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(1.0, max(0.0, progress))
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training ──
    ensure_dir(args.output)
    best_val_loss = float("inf")
    patience_counter = 0

    log.info(f"SFT Training | epochs={args.epochs} | lr={args.lr} | "
             f"batch_size={args.batch_size} | train={train_size:,} | val={val_size:,}")
    log.info(f"Total steps: {total_steps:,} | Warmup: {warmup_steps} | Patience: {patience}")

    for epoch in range(args.epochs):
        epoch_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            if not (labels != -100).any():
                continue

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

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        avg_train = epoch_loss / max(n_batches, 1)

        # ── Validation ──
        val_loss = _eval_val_loss(model, val_loader, device)
        log.info(
            f"Epoch {epoch+1} | Train Loss: {avg_train:.4f} | "
            f"Val Loss: {val_loss:.4f} | Best Val: {best_val_loss:.4f}"
        )

        # Save checkpoint each epoch
        ckpt_path = os.path.join(args.output, f"sft_epoch{epoch+1}.pt")
        model.save_pretrained(ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = os.path.join(args.output, "sft_final.pt")
            model.save_pretrained(best_path)
            log.info(f"Best model saved to {best_path} (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            log.info(f"No val improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                log.info(
                    f"Early stopping at epoch {epoch+1} — val loss has not "
                    f"improved for {patience} epochs. Best: {best_val_loss:.4f}"
                )
                break

    log.info("SFT training complete!")
    log.info(f"Best model: {args.output}/sft_final.pt (val_loss={best_val_loss:.4f})")
