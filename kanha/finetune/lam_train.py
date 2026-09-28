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


# ── Training loop ─────────────────────────────────────────────────────────────

def lam_sft_train(args):
    """
    SFT training on trajectory step-expansion data.

    Nearly identical to sft_train() but uses LAMSFTDataset and
    LAM prompt markers for loss masking.
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
    max_seq_len = getattr(cfg.model, "max_seq_len", 512)
    dataset = LAMSFTDataset(args.data, tokenizer, max_len=max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    # ── Optimizer (same safe defaults as sft_train) ──
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
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training ──
    ensure_dir(args.output)
    best_loss = float("inf")

    log.info(
        f"LAM SFT Training | epochs={args.epochs} | lr={args.lr} | "
        f"batch_size={args.batch_size} | steps={len(dataset):,}"
    )
    log.info(f"Total optimiser steps: {total_steps:,} | Warmup: {warmup_steps}")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"LAM Epoch {epoch+1}/{args.epochs}")
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

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        avg_loss = epoch_loss / max(n_batches, 1)
        log.info(f"LAM Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

        # Save checkpoint each epoch
        ckpt_path = os.path.join(args.output, f"lam_epoch{epoch+1}.pt")
        model.save_pretrained(ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.output, "lam_final.pt")
            model.save_pretrained(best_path)
            log.info(f"Best LAM model saved to {best_path} (loss={best_loss:.4f})")

    log.info("LAM SFT training complete!")
    log.info(f"Best model: {args.output}/lam_final.pt")
