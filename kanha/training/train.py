"""
kanha/training/train.py
Pre-training from scratch on tokenized text data.

Expects a .npy file of token ids (created by scripts/preprocess_data.py).
Trains a causal language model with next-token prediction.

Run:
    python main.py train --data data/processed/train.npy
"""

import os
import math
import numpy as np
import torch
from tqdm import tqdm

from kanha.core.model import KanhaModel
from kanha.utils.config import cfg
from kanha.utils.helpers import get_device, ensure_dir, count_parameters
from kanha.utils.logging import get_logger

log = get_logger("train")


class TokenDataset:
    """
    Memory-mapped dataset over a flat .npy token array.
    Returns fixed-length sequences for causal LM training.
    """

    def __init__(self, data_path: str, seq_len: int):
        self.data = np.load(data_path, mmap_mode="r")
        self.seq_len = seq_len
        self.n_tokens = len(self.data)
        self.n_sequences = self.n_tokens // (seq_len + 1)
        log.info(f"Dataset: {self.n_tokens:,} tokens → {self.n_sequences:,} sequences")

    def __len__(self):
        return self.n_sequences

    def get_batch(self, batch_size: int, device: torch.device):
        """Returns a random batch of (input_ids, targets)."""
        indices = np.random.randint(0, self.n_tokens - self.seq_len - 1, size=batch_size)
        x = np.stack([self.data[i:i + self.seq_len] for i in indices])
        y = np.stack([self.data[i + 1:i + self.seq_len + 1] for i in indices])
        return (
            torch.tensor(x, dtype=torch.long, device=device),
            torch.tensor(y, dtype=torch.long, device=device),
        )


def train(args):
    """
    Main pre-training loop. Called from main.py train command.
    """
    device = get_device()
    tc = cfg.training
    mc = cfg.model

    # ── Dataset ──
    dataset = TokenDataset(args.data, seq_len=mc.max_seq_len)

    # ── Model ──
    if args.resume:
        log.info(f"Resuming from {args.resume}")
        model = KanhaModel.from_pretrained(args.resume)
    else:
        log.info("Initializing new model from config")
        model = KanhaModel()

    model.train()
    model.to(device)
    log.info(f"Parameters: {count_parameters(model):,}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc.learning_rate,
        weight_decay=tc.weight_decay,
        betas=(0.9, 0.95),
    )

    # ── Cosine schedule with warmup ──
    def lr_lambda(step):
        if step < tc.warmup_steps:
            return step / max(tc.warmup_steps, 1)
        progress = (step - tc.warmup_steps) / max(tc.max_steps - tc.warmup_steps, 1)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training loop ──
    save_dir = cfg.paths.base_model
    ensure_dir(save_dir)

    running_loss = 0.0
    pbar = tqdm(range(1, tc.max_steps + 1), desc="Pre-training")

    for step in pbar:
        # Gradient accumulation
        optimizer.zero_grad()
        accum_loss = 0.0

        for micro_step in range(tc.grad_accum_steps):
            input_ids, targets = dataset.get_batch(tc.batch_size, device)
            _, loss, _ = model(input_ids, targets=targets)
            loss = loss / tc.grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        optimizer.step()
        scheduler.step()

        running_loss += accum_loss

        if step % tc.log_every == 0:
            avg = running_loss / tc.log_every
            lr_now = scheduler.get_last_lr()[0]
            pbar.set_postfix({"loss": f"{avg:.4f}", "lr": f"{lr_now:.2e}"})
            running_loss = 0.0

        if step % tc.save_every == 0:
            ckpt_path = os.path.join(save_dir, f"ckpt_step{step}.pt")
            model.save_pretrained(ckpt_path)
            log.info(f"Checkpoint saved: {ckpt_path}")

    # Final save
    final_path = os.path.join(save_dir, "final_model.pt")
    model.save_pretrained(final_path)
    log.info(f"Training complete! Model saved to {final_path}")
