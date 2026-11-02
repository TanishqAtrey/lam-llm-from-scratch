"""
kanha/finetune/lam_train.py
SFT training for the Language Action Model (LAM).

Trains the model on expanded trajectory steps: the prompt is everything
up to '### Action\\n' and the response is the action text itself.

Uses the same training infrastructure as sft_train.py (optimizer, scheduler,
gradient clipping, checkpointing) but with a LAM-specific dataset that masks
the trajectory prompt and trains only on action tokens.

Run:
    python main.py lam-train \\
        --base_model models/base/final_model.pt \\
        --data data/lam/trajectories.jsonl \\
        --output models/lam/ \\
        --epochs 3 \\
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
from kanha.prompting.builder import build_lam_step_prompt
from kanha.utils.config import cfg
from kanha.utils.helpers import get_device, ensure_dir
from kanha.utils.logging import get_logger

log = get_logger("lam_sft")


# ── Dataset ───────────────────────────────────────────────────────────────────

class LAMSFTDataset(Dataset):
    """
    Loads trajectory JSONL, expands each trajectory into per-step
    (prompt, response) pairs, and tokenises them with prompt masking.

    Each JSONL line has:
        {"goal": "...", "steps": [{"action": "...", "observation": "..."}, ...], ...}

    For a trajectory with N steps we create N training examples:
        step 0: prompt = "### Goal\\n{goal}\\n### Action\\n"
                 response = steps[0]["action"]
        step i: prompt = goal + steps[:i] history + "### Action\\n"
                 response = steps[i]["action"]
    """

    def __init__(self, data_path: str, tokenizer: KanhaTokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples: list[tuple[str, str]] = []  # (prompt, response)

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                traj = json.loads(line)
                goal = traj["goal"]
                steps = traj.get("steps", [])
                if not steps:
                    continue
                # Expand into per-step examples
                for i, step in enumerate(steps):
                    history = steps[:i]
                    prompt = build_lam_step_prompt(goal, history)
                    response = step["action"]
                    self.examples.append((prompt, response))

        log.info(f"Loaded {len(self.examples):,} LAM training steps from {data_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        prompt, response = self.examples[idx]

        # Build full text: prompt + response
        full_text = prompt + response

        # Tokenise
        full_ids = self.tokenizer.encode(
            full_text, add_bos=True, add_eos=True, max_length=self.max_len
        )

        # Compute prompt length for loss masking
        prompt_ids = self.tokenizer.encode(
            prompt, add_bos=True, add_eos=False
        )
        mask_len = len(prompt_ids)

        # Shifted input / target pairs (standard causal LM)
        input_ids = full_ids[:-1]
        labels = full_ids[1:]

        # Mask prompt tokens in labels
        for i in range(min(mask_len - 1, len(labels))):
            labels[i] = -100

        # Safety guard: ensure at least one supervised token remains to prevent NaN loss
        if not any(l != -100 for l in labels) and len(labels) > 0:
            labels[-1] = full_ids[-1]

        # Pad / truncate to max_len - 1
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
def _eval_lam_val_loss(model, val_loader, device):
    """Computes average validation loss over the LAM val set."""
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

def lam_sft_train(args):
    """
    SFT training on trajectory step-expansion data.

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
    max_seq_len = getattr(cfg.model, "max_seq_len", 512)
    full_dataset = LAMSFTDataset(args.data, tokenizer, max_len=max_seq_len)

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

    log.info(f"Split: {train_size:,} train / {val_size:,} val steps")

    # ── Optimizer (same safe defaults as sft_train) ──
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

    log.info(
        f"LAM SFT Training | epochs={args.epochs} | lr={args.lr} | "
        f"batch_size={args.batch_size} | train={train_size:,} | val={val_size:,}"
    )
    log.info(f"Total optimiser steps: {total_steps:,} | Warmup: {warmup_steps} | Patience: {patience}")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"LAM Epoch {epoch+1}/{args.epochs}")
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
        val_loss = _eval_lam_val_loss(model, val_loader, device)
        log.info(
            f"LAM Epoch {epoch+1} | Train Loss: {avg_train:.4f} | "
            f"Val Loss: {val_loss:.4f} | Best Val: {best_val_loss:.4f}"
        )

        # Save checkpoint each epoch
        ckpt_path = os.path.join(args.output, f"lam_epoch{epoch+1}.pt")
        model.save_pretrained(ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = os.path.join(args.output, "lam_final.pt")
            model.save_pretrained(best_path)
            log.info(f"Best LAM model saved to {best_path} (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            log.info(f"No val improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                log.info(
                    f"Early stopping at epoch {epoch+1} — val loss has not "
                    f"improved for {patience} epochs. Best: {best_val_loss:.4f}"
                )
                break

    log.info("LAM SFT training complete!")
    log.info(f"Best model: {args.output}/lam_final.pt (val_loss={best_val_loss:.4f})")
