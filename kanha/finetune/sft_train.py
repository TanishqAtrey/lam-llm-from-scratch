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

    def __init__(self, data_path: str, tokenizer: KanhaTokenizer, max_len: int = 1024):
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
                inst = obj.get("instruction") or obj.get("prompt", "")
                if obj.get("input") and str(obj["input"]).strip():
                    inst = f"{inst}\n\n{str(obj['input']).strip()}".strip() if inst else str(obj["input"]).strip()
                resp = obj.get("response") or obj.get("output") or obj.get("completion", "")
                if inst and resp and resp.strip():
                    p_prefix, _ = self.prompt_builder.build_prompt_prefix(inst)
                    r_text = resp.strip()
                    p_ids = self.tokenizer.encode(p_prefix, add_bos=True, add_eos=False)
                    r_ids = self.tokenizer.encode(r_text, add_bos=False, add_eos=True)
                    tot_len = len(p_ids) + len(r_ids)
                    if tot_len <= self.max_len and len(self.tokenizer.encode(r_text, add_bos=False, add_eos=False)) > 0:
                        self.samples.append({"instruction": inst, "response": r_text})

        log.info(f"Loaded {len(self.samples):,} valid SFT samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        instruction = sample["instruction"]
        response = sample["response"]

        # ── Compute canonical instruction prefix and response text directly ──
        # Avoids fragile string-searching (e.g. rfind) which breaks if response contains delimiters
        instruction_prefix, _ = self.prompt_builder.build_prompt_prefix(instruction)
        resp_text = response

        # ── Tokenize prompt and response separately to prevent BPE boundary-piece merging ──
        instruction_ids = self.tokenizer.encode(
            instruction_prefix, add_bos=True, add_eos=False
        )
        response_ids = self.tokenizer.encode(
            resp_text, add_bos=False, add_eos=True
        )
        full_ids = instruction_ids + response_ids
        mask_len = len(instruction_ids)

        if len(full_ids) > self.max_len:
            full_ids = full_ids[:self.max_len - 1] + [self.tokenizer.EOS_ID]

        # ── Create shifted input/target pairs ──
        # Standard causal LM: input = tokens[:-1], target = tokens[1:]
        input_ids = full_ids[:-1]
        labels = full_ids[1:]

        # ── Mask instruction tokens in labels ──
        # Set to -100 so cross_entropy ignores them
        for i in range(min(mask_len - 1, len(labels))):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def sft_collate_fn(batch, pad_id=0):
    """Dynamically pads each batch to its maximum sequence length instead of fixed 1023."""
    # Filter out samples with no valid response tokens
    valid_batch = [x for x in batch if (x["labels"] != -100).any()]
    if valid_batch:
        batch = valid_batch

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
def _eval_val_loss(model, val_loader, device):
    """Computes average loss over the entire validation set."""
    model.eval()
    total_loss = 0.0
    n = 0
    use_amp = (device.type == "cuda")
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        if not (labels != -100).any():
            continue
        with torch.amp.autocast("cuda", enabled=use_amp):
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

    val_fraction = getattr(args, "val_fraction", None)
    if val_fraction is None:
        val_fraction = getattr(cfg.training, "val_fraction", 0.1)
    val_fraction = float(val_fraction)

    patience = getattr(args, "patience", None)
    if patience is None:
        patience = getattr(cfg.training, "sft_patience", 3)

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
        if getattr(args, "lr", None) is None:
            default_lora_lr = 1e-4
            log.info(f"LoRA enabled: setting default SFT learning rate to {default_lora_lr}")
            args.lr = default_lora_lr

    if getattr(args, "lr", None) is None:
        args.lr = 3e-5

    model.train()
    model.to(device)

    # ── Load tokenizer ──
    tokenizer = KanhaTokenizer()

    # ── Dataset with train/val split ──
    model_max_len = getattr(model, "max_seq_len", getattr(getattr(model, "model", None), "max_seq_len", cfg.model.max_seq_len))
    val_data_path = getattr(args, "val_data", None)
    if val_data_path and os.path.exists(val_data_path):
        log.info(f"Loading separate validation dataset from {val_data_path}")
        train_dataset = SFTDataset(args.data, tokenizer, max_len=model_max_len)
        val_dataset = SFTDataset(val_data_path, tokenizer, max_len=model_max_len)
        train_size = len(train_dataset)
        val_size = len(val_dataset)
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size,
            shuffle=False, drop_last=False, num_workers=0,
            collate_fn=sft_collate_fn,
        )
    else:
        full_dataset = SFTDataset(args.data, tokenizer, max_len=model_max_len)
        val_size = max(1, int(len(full_dataset) * val_fraction)) if len(full_dataset) > 10 else 0
        train_size = len(full_dataset) - val_size
        if val_size > 0 and train_size > 0:
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size,
                shuffle=False, drop_last=False, num_workers=0,
                collate_fn=sft_collate_fn,
            )
        else:
            train_dataset = full_dataset
            train_size = len(full_dataset)
            val_dataset = None
            val_size = 0
            val_loader = None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, drop_last=False, num_workers=0,
        collate_fn=sft_collate_fn,
    )

    log.info(f"Split: {train_size:,} train / {val_size:,} val samples")

    # ── Optimizer (separate weight decay for LoRA / biases / norms) ──
    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "lora_" in name or "bias" in name or "norm" in name:
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": cfg.training.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    grad_accum_steps = getattr(args, "grad_accum_steps", None) or getattr(cfg.training, "grad_accum_steps", 4)
    grad_accum_steps = max(1, int(grad_accum_steps))

    # ── Warmup + cosine schedule (starts at step 1 instead of lr=0) ──
    total_steps = math.ceil(len(train_loader) / grad_accum_steps) * args.epochs
    warmup_steps = min(100, max(1, total_steps // 10))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(1.0, max(0.0, progress))
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training ──
    ensure_dir(args.output)
    best_val_loss = float("inf")
    patience_counter = 0

    log.info(f"SFT Training | epochs={args.epochs} | lr={args.lr} | "
             f"batch_size={args.batch_size} | grad_accum={grad_accum_steps} | "
             f"train={train_size:,} | val={val_size:,}")
    log.info(f"Total optim steps: {total_steps:,} | Warmup: {warmup_steps} | Patience: {patience}")

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        log.info("AMP FP16 mixed precision enabled for SFT")

    for epoch in range(args.epochs):
        epoch_loss = 0
        n_batches = 0
        optimizer.zero_grad()
        accum_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            if not (labels != -100).any():
                is_last_batch = (batch_idx + 1 == len(train_loader))
                if is_last_batch and accum_count > 0:
                    rem = accum_count % grad_accum_steps
                    if rem != 0:
                        scale_factor = grad_accum_steps / rem
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad.data.mul_(scale_factor)
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    accum_count = 0
                continue

            # Forward with AMP
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, loss, _ = model(input_ids, targets=labels)
                loss_scaled = loss / grad_accum_steps

            # Backward with gradient accumulation
            scaler.scale(loss_scaled).backward()
            accum_count += 1

            is_accum_boundary = (accum_count % grad_accum_steps == 0)
            is_last_batch = (batch_idx + 1 == len(train_loader))

            if is_accum_boundary or (is_last_batch and accum_count > 0):
                # Scale gradient if remainder micro-batch is smaller than grad_accum_steps
                rem = accum_count % grad_accum_steps
                if is_last_batch and rem != 0:
                    scale_factor = grad_accum_steps / rem
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.data.mul_(scale_factor)

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                accum_count = 0

            epoch_loss += loss.item()
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        avg_train = epoch_loss / max(n_batches, 1)

        # ── Validation ──
        val_loss = _eval_val_loss(model, val_loader, device) if val_loader else avg_train
        log.info(
            f"Epoch {epoch+1} | Train Loss: {avg_train:.4f} | "
            f"Val Loss: {val_loss:.4f} | Best Val: {best_val_loss:.4f}"
        )

        # Free GPU memory cache between epochs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
