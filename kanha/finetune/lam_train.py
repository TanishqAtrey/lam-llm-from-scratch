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
    Loads trajectory JSONL (or list of trajectory dicts), expands each trajectory
    into per-step (prompt, response) pairs, and tokenises them with prompt masking.
    """

    def __init__(self, data: Union[str, list[dict]], tokenizer: KanhaTokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples: list[tuple[str, str]] = []  # (prompt, response)

        trajectories = []
        if isinstance(data, str):
            with open(data, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        trajectories.append(json.loads(line))
        else:
            trajectories = data

        for traj in trajectories:
            goal = traj.get("goal", "")
            steps = traj.get("steps", [])
            if not steps or not goal:
                continue
            for i, step in enumerate(steps):
                history = steps[:i]
                prompt = build_lam_step_prompt(goal, history)
                response = step.get("action", "")
                if response:
                    self.examples.append((prompt, response))

        src_desc = f"{len(trajectories)} trajectories" if isinstance(data, list) else data
        log.info(f"Loaded {len(self.examples):,} LAM training steps from {src_desc}")

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

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def lam_collate_fn(batch, pad_id=0):
    """Dynamically pads each batch to its maximum sequence length instead of fixed 1023."""
    max_len = max(len(x["input_ids"]) for x in batch)
    b_input_ids = []
    b_labels = []
    for x in batch:
        inp = x["input_ids"]
        lbl = x["labels"]
        pad_len = max_len - len(inp)
        if pad_len > 0:
            inp = torch.cat([inp, torch.full((pad_len,), pad_id, dtype=torch.long)])
            lbl = torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])
        b_input_ids.append(inp)
        b_labels.append(lbl)
    return {
        "input_ids": torch.stack(b_input_ids),
        "labels": torch.stack(b_labels),
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
    val_data = getattr(args, "val_data", None)
    if not val_data:
        candidate_val = os.path.join(os.path.dirname(args.data), "val.jsonl")
        if os.path.exists(candidate_val) and candidate_val != args.data:
            val_data = candidate_val

    if val_data and os.path.exists(val_data):
        log.info(f"Using separate trajectory validation file: {val_data}")
        train_dataset = LAMSFTDataset(args.data, tokenizer, max_len=max_seq_len)
        val_dataset = LAMSFTDataset(val_data, tokenizer, max_len=max_seq_len)
    else:
        # Load full trajectories and split at trajectory level to prevent step leakage
        import random as _rnd
        with open(args.data, "r", encoding="utf-8") as f:
            all_trajs = [json.loads(line) for line in f if line.strip()]

        val_count = max(1, int(len(all_trajs) * val_fraction))
        train_count = len(all_trajs) - val_count

        rng = _rnd.Random(42)
        rng.shuffle(all_trajs)
        train_trajs = all_trajs[:train_count]
        val_trajs = all_trajs[train_count:]

        train_dataset = LAMSFTDataset(train_trajs, tokenizer, max_len=max_seq_len)
        val_dataset = LAMSFTDataset(val_trajs, tokenizer, max_len=max_seq_len)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, drop_last=True, num_workers=0,
        collate_fn=lam_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, drop_last=False, num_workers=0,
        collate_fn=lam_collate_fn,
    )

    log.info(f"Split: {len(train_dataset):,} train / {len(val_dataset):,} val steps")

    # ── Optimizer (same safe defaults as sft_train) ──
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.95),
    )

    grad_accum_steps = getattr(args, "grad_accum_steps", None) or getattr(cfg.training, "grad_accum_steps", 4)
    grad_accum_steps = max(1, int(grad_accum_steps))

    # ── Warmup + cosine schedule ──
    total_steps = math.ceil(len(train_loader) / grad_accum_steps) * args.epochs
    warmup_steps = min(100, max(1, total_steps // 10))

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
        f"batch_size={args.batch_size} | grad_accum={grad_accum_steps} | "
        f"train={train_size:,} | val={val_size:,}"
    )
    log.info(f"Total optimiser steps: {total_steps:,} | Warmup: {warmup_steps} | Patience: {patience}")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"LAM Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            if not (labels != -100).any():
                continue

            # Forward
            logits, loss, _ = model(input_ids, targets=labels)

            # Backward with gradient accumulation
            loss_scaled = loss / grad_accum_steps
            loss_scaled.backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
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
