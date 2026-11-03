"""
kanha/training/train.py
Pre-training from scratch on tokenized text data.

Expects a .npy file of token ids (created by scripts/prepare_3b_tokens.py).
Trains a causal language model with next-token prediction.

Features:
  - Validation loss tracking (eval every N steps on held-out 10% split)
  - Early stopping with configurable patience
  - Saves best checkpoint by validation loss (not just latest)

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

    Supports train/val splitting: pass split='train' or split='val'
    with a val_fraction (default 0.1) to carve out a held-out set.
    """

    def __init__(self, data_path: str, seq_len: int,
                 split: str = "all", val_fraction: float = 0.1):
        full_data = np.load(data_path, mmap_mode="r")
        self.seq_len = seq_len

        n_total = len(full_data)
        val_size = int(n_total * val_fraction)

        if split == "val":
            self.data = full_data[:val_size]
        elif split == "train":
            self.data = full_data[val_size:]
        else:
            self.data = full_data

        self.n_tokens = len(self.data)
        self.n_sequences = self.n_tokens // (seq_len + 1)
        log.info(f"Dataset [{split}]: {self.n_tokens:,} tokens → {self.n_sequences:,} sequences")

    def __len__(self):
        return self.n_sequences

    def get_batch(self, batch_size: int, device: torch.device):
        """Returns a random batch of (input_ids, targets)."""
        max_start = self.n_tokens - self.seq_len - 1
        if max_start <= 0:
            raise ValueError(f"Dataset has too few tokens ({self.n_tokens}) for sequence length {self.seq_len}")
        indices = np.random.randint(0, max_start, size=batch_size)
        x = np.stack([self.data[i:i + self.seq_len] for i in indices])
        y = np.stack([self.data[i + 1:i + self.seq_len + 1] for i in indices])
        return (
            torch.tensor(x, dtype=torch.long, device=device),
            torch.tensor(y, dtype=torch.long, device=device),
        )


@torch.no_grad()
def _evaluate_val_loss(model, val_dataset, device, batch_size, n_eval_batches=20, use_amp=False):
    """
    Computes average validation loss over n_eval_batches random batches.
    Model is temporarily set to eval mode and restored after.
    """
    model.eval()
    total_loss = 0.0
    for _ in range(n_eval_batches):
        input_ids, targets = val_dataset.get_batch(batch_size, device)
        with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", dtype=torch.float16 if use_amp else torch.float32, enabled=use_amp):
            _, loss, _ = model(input_ids, targets=targets)
        total_loss += loss.item()
    model.train()
    return total_loss / n_eval_batches


def _save_pretrain_checkpoint(path: str, model: KanhaModel, optimizer, scheduler, step: int, best_val_loss: float, scaler=None):
    cfg_dict = {
        "vocab_size":  model.vocab_size,
        "dim":         model.dim,
        "n_layers":    model.n_layers,
        "n_heads":     model.n_heads,
        "ff_dim":      model.ff_dim,
        "max_seq_len": model.max_seq_len,
        "dropout":     model.dropout_p,
    }
    if hasattr(model, "n_kv_heads") and model.n_kv_heads is not None:
        cfg_dict["n_kv_heads"] = model.n_kv_heads

    ckpt_data = {
        "step": step,
        "model_config": cfg_dict,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }
    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
        ckpt_data["scaler_state"] = scaler.state_dict()
    if torch.cuda.is_available():
        ckpt_data["cuda_rng_state"] = torch.cuda.get_rng_state_all()

    torch.save(ckpt_data, path)
    log.info(f"Checkpoint saved: {path} (with optimizer, scheduler & RNG states)")


def train(args):
    """
    Main pre-training loop. Called from main.py train command.

    Includes validation loss tracking and early stopping:
      --val_fraction  : fraction of data for validation (default 0.1)
      --patience      : early-stop if val loss doesn't improve for N evals (default 5)
      --eval_every    : evaluate val loss every N steps (default = save_every)
    """
    device = get_device()
    tc = cfg.training
    mc = cfg.model

    val_fraction = getattr(args, "val_fraction", None)
    if val_fraction is None:
        val_fraction = getattr(tc, "val_fraction", 0.1)
    patience = getattr(args, "patience", None) or getattr(tc, "patience", 5)
    eval_every = getattr(args, "eval_every", None) or getattr(tc, "eval_every", None) or getattr(tc, "save_every", 5000)

    # ── Datasets (train / val split) ──
    train_dataset = TokenDataset(args.data, seq_len=mc.max_seq_len,
                                 split="train", val_fraction=val_fraction)
    full_data = np.load(args.data, mmap_mode="r")
    val_size = int(len(full_data) * val_fraction) if val_fraction else 0
    if val_fraction > 0 and val_size > mc.max_seq_len + 1:
        val_dataset = TokenDataset(args.data, seq_len=mc.max_seq_len,
                                   split="val", val_fraction=val_fraction)
    else:
        val_dataset = None
        if val_fraction > 0:
            log.warning(
                f"Validation split requested ({val_fraction}) but val_size ({val_size}) "
                f"<= seq_len + 1 ({mc.max_seq_len + 1}). Disabling validation."
            )

    # ── Model ──
    start_step = 0
    raw_ckpt = None
    if args.resume:
        log.info(f"Resuming from {args.resume}")
        raw_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        start_step = raw_ckpt.get("step", 0)
        if start_step == 0:
            import re
            match = re.search(r"step(\d+)", os.path.basename(args.resume))
            if match:
                start_step = int(match.group(1))
        model = KanhaModel.from_pretrained(args.resume)
        if start_step > 0:
            log.info(f"Resuming training from step {start_step}")
    else:
        log.info("Initializing new model from config")
        model = KanhaModel()

    model.train()
    model.to(device)
    log.info(f"Parameters: {count_parameters(model):,}")

    # ── Optimizer ──
    # Separate 2D matrix weights (decayed) from 1D biases and RMSNorm scales (non-decayed)
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": tc.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=tc.learning_rate,
        betas=(0.9, 0.95),
    )

    # Ensure 'initial_lr' is defined on all param groups (required by PyTorch when resuming with last_epoch >= 0)
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group.get("lr", tc.learning_rate))

    # ── Cosine schedule with warmup ──
    def lr_lambda(step):
        if step < tc.warmup_steps:
            return step / max(tc.warmup_steps, 1)
        progress = (step - tc.warmup_steps) / max(tc.max_steps - tc.warmup_steps, 1)
        progress = min(1.0, max(0.0, progress))
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=start_step - 1 if start_step > 0 else -1
    )

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        log.info("AMP FP16 Mixed Precision enabled (T4 Tensor Cores active)")

    best_val_loss = float("inf")
    # Restore optimizer and scheduler states if present in checkpoint
    if raw_ckpt is not None:
        if "optimizer_state" in raw_ckpt:
            try:
                optimizer.load_state_dict(raw_ckpt["optimizer_state"])
                for group in optimizer.param_groups:
                    group.setdefault("initial_lr", group.get("lr", tc.learning_rate))
                log.info("Restored AdamW optimizer state (momentum & variance buffers)")
            except Exception as e:
                log.warning(f"Could not restore optimizer state: {e}")
        if "scheduler_state" in raw_ckpt:
            try:
                scheduler.load_state_dict(raw_ckpt["scheduler_state"])
                log.info("Restored learning rate scheduler state")
            except Exception as e:
                log.warning(f"Could not restore scheduler state: {e}")
        if "scaler_state" in raw_ckpt and scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
            try:
                scaler.load_state_dict(raw_ckpt["scaler_state"])
                log.info("Restored AMP GradScaler state")
            except Exception as e:
                log.warning(f"Could not restore GradScaler state: {e}")
        if "best_val_loss" in raw_ckpt:
            best_val_loss = raw_ckpt["best_val_loss"]
        if "torch_rng_state" in raw_ckpt:
            try:
                torch.set_rng_state(raw_ckpt["torch_rng_state"])
                log.info("Restored PyTorch RNG state")
            except Exception as e:
                log.warning(f"Could not restore PyTorch RNG state: {e}")
        if "numpy_rng_state" in raw_ckpt:
            try:
                np.random.set_state(raw_ckpt["numpy_rng_state"])
                log.info("Restored NumPy RNG state")
            except Exception as e:
                log.warning(f"Could not restore NumPy RNG state: {e}")
        if "cuda_rng_state" in raw_ckpt and raw_ckpt["cuda_rng_state"] is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(raw_ckpt["cuda_rng_state"])
                log.info("Restored CUDA RNG state")
            except Exception as e:
                log.warning(f"Could not restore CUDA RNG state: {e}")

    # ── Training loop ──
    save_dir = cfg.paths.base_model
    ensure_dir(save_dir)

    running_loss = 0.0
    patience_counter = 0
    pbar = tqdm(
        range(start_step + 1, tc.max_steps + 1),
        initial=start_step,
        total=tc.max_steps,
        desc="Pre-training",
    )

    for step in pbar:
        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(tc.grad_accum_steps):
            input_ids, targets = train_dataset.get_batch(tc.batch_size, device)
            with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", dtype=torch.float16 if use_amp else torch.float32, enabled=use_amp):
                _, loss, _ = model(input_ids, targets=targets)
                loss = loss / tc.grad_accum_steps
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += accum_loss

        if step % tc.log_every == 0:
            avg = running_loss / tc.log_every
            lr_now = scheduler.get_last_lr()[0]
            pbar.set_postfix({"loss": f"{avg:.4f}", "lr": f"{lr_now:.2e}"})
            running_loss = 0.0

        if step % tc.save_every == 0:
            ckpt_path = os.path.join(save_dir, f"ckpt_step{step}.pt")
            _save_pretrain_checkpoint(ckpt_path, model, optimizer, scheduler, step, best_val_loss, scaler=scaler)

        # ── Validation + Early stopping ──
        if val_dataset is not None and eval_every and step % eval_every == 0:
            val_loss = _evaluate_val_loss(
                model, val_dataset, device, tc.batch_size, use_amp=use_amp
            )
            log.info(f"Step {step} | Val Loss: {val_loss:.4f} | Best: {best_val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_path = os.path.join(save_dir, "best_model.pt")
                _save_pretrain_checkpoint(best_path, model, optimizer, scheduler, step, best_val_loss, scaler=scaler)
                log.info(f"New best model saved (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                log.info(f"No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    log.info(
                        f"Early stopping at step {step} — val loss has not "
                        f"improved for {patience} evals. Best: {best_val_loss:.4f}"
                    )
                    break

    # Final save
    final_path = os.path.join(save_dir, "final_model.pt")
    final_step = step if "step" in locals() else tc.max_steps
    _save_pretrain_checkpoint(final_path, model, optimizer, scheduler, final_step, best_val_loss, scaler=scaler)
    log.info(f"Training complete! Model saved to {final_path}")
    log.info(f"Best validation loss: {best_val_loss:.4f} (saved to {save_dir}/best_model.pt)")
