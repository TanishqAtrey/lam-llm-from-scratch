"""
kanha/finetune/dpo_train.py
Direct Preference Optimization (DPO) for KANHA.

DPO aligns the model with human preferences without needing a
separate reward model. Given (prompt, chosen, rejected) triples,
it directly optimizes the policy to prefer chosen over rejected.

Data format (JSONL):
    {"prompt": "...", "chosen": "...", "rejected": "..."}

CRITICAL: Uses the same prompt template as SFT training.

Run:
    python main.py dpo \
        --base_model models/finetuned/sft_final.pt \
        --data data/processed/dpo.jsonl \
        --output models/finetuned/ \
        --epochs 1 \
        --lr 1e-6
"""

import os
import json
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from kanha.core.model import KanhaModel
from kanha.core.tokenizer import KanhaTokenizer
from kanha.prompting.builder import PromptBuilder
from kanha.utils.config import cfg
from kanha.utils.helpers import get_device, ensure_dir
from kanha.utils.logging import get_logger

log = get_logger("dpo")


def disable_dropout(model: torch.nn.Module) -> None:
    """Disables all dropout layers in model to prevent policy/ref divergence noise in DPO."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if hasattr(module, "dropout"):
            if isinstance(module.dropout, torch.nn.Dropout):
                module.dropout.p = 0.0
            elif isinstance(module.dropout, (float, int)):
                module.dropout = 0.0
        if hasattr(module, "lora_dropout"):
            if isinstance(module.lora_dropout, torch.nn.Dropout):
                module.lora_dropout.p = 0.0
            elif isinstance(module.lora_dropout, (float, int)):
                module.lora_dropout = 0.0


# ── DPO Loss ──────────────────────────────────────────────────────────────────

def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes DPO loss.

    L_DPO = -log sigmoid(beta * (log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x)))

    Args:
        policy_chosen_logps   : log probs of chosen under current policy
        policy_rejected_logps : log probs of rejected under current policy
        ref_chosen_logps      : log probs of chosen under reference model
        ref_rejected_logps    : log probs of rejected under reference model
        beta                  : temperature parameter

    Returns:
        (loss, chosen_rewards, rejected_rewards)
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss, chosen_rewards, rejected_rewards


def compute_log_probs(model, input_ids, labels, device, length_norm: bool = False):
    """
    Computes per-sequence log probabilities for the response tokens only.

    IMPORTANT: input_ids and labels are ALREADY shifted by the dataset
    (input_ids = full[:-1], labels = full[1:]), so NO additional shifting
    is needed here. logits[i] should predict labels[i] directly.

    Args:
        model       : KanhaModel
        input_ids   : (B, S) input token ids (already shifted)
        labels      : (B, S) target labels (already shifted, -100 for masked)
        device      : torch device
        length_norm : whether to divide by sequence length (reduces verbosity bias)

    Returns:
        (B,) tensor of summed (or length-normalized) log probabilities per sequence
    """
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    logits, _, _ = model(input_ids)

    # No shifting needed — dataset already provides shifted pairs
    # logits[i] predicts the token at labels[i]
    log_probs = F.log_softmax(logits, dim=-1)

    # Handle -100 labels (can't use as gather indices)
    safe_labels = labels.clone()
    safe_labels[safe_labels == -100] = 0  # replace with any valid index

    # Gather the log prob of the target token at each position
    per_token_logps = torch.gather(
        log_probs, 2, safe_labels.unsqueeze(2)
    ).squeeze(2)

    # Mask: only count response tokens (where label != -100)
    mask = (labels != -100).float()
    per_token_logps = per_token_logps * mask

    summed_logps = per_token_logps.sum(dim=-1)
    if length_norm:
        token_counts = mask.sum(dim=-1).clamp(min=1.0)
        return summed_logps / token_counts
    return summed_logps


# ── Dataset ───────────────────────────────────────────────────────────────────

class DPODataset(Dataset):
    """
    Loads JSONL with {"prompt": ..., "chosen": ..., "rejected": ...}
    and tokenizes both chosen and rejected completions.
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
                if obj.get("prompt") and obj.get("chosen") and obj.get("rejected"):
                    p = obj["prompt"]
                    c_ids, c_lbls = self._tokenize_pair(p, obj["chosen"])
                    r_ids, r_lbls = self._tokenize_pair(p, obj["rejected"])
                    # Guard: verify at least 1 valid supervised token exists in both chosen and rejected
                    if (c_lbls != -100).any() and (r_lbls != -100).any():
                        self.samples.append(obj)

        log.info(f"Loaded {len(self.samples):,} valid DPO samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def _tokenize_pair(self, prompt: str, response: str):
        """Tokenizes a (prompt, response) pair with boundary-safe instruction masking and dialogue support."""
        clean_prompt = prompt
        if clean_prompt.startswith("### Instruction:\n"):
            clean_prompt = clean_prompt[len("### Instruction:\n"):].strip()
        if "### Response:\n" in clean_prompt:
            clean_prompt = clean_prompt[:clean_prompt.rfind("### Response:\n")].strip()

        # Format with canonical template (handles multi-turn conversation history automatically)
        full_text = self.prompt_builder.build_training_pair(clean_prompt, response)

        # Compute instruction prefix and response text for boundary-safe tokenization
        if "### Response:\n" in full_text:
            split_idx = full_text.rfind("### Response:\n") + len("### Response:\n")
            instruction_prefix = full_text[:split_idx]
            resp_text = full_text[split_idx:]
        else:
            instruction_prefix = f"### Instruction:\n{clean_prompt}\n\n### Response:\n"
            resp_text = response

        # Tokenize prompt and response separately to guarantee exact boundary alignment
        instruction_ids = self.tokenizer.encode(
            instruction_prefix, add_bos=True, add_eos=False
        )
        resp_ids = self.tokenizer.encode(
            resp_text, add_bos=False, add_eos=True
        )
        full_ids = instruction_ids + resp_ids
        mask_len = len(instruction_ids)

        if len(full_ids) > self.max_len:
            full_ids = full_ids[:self.max_len]

        # Shifted input/labels
        input_ids = full_ids[:-1]
        labels = full_ids[1:]

        # Mask instruction tokens
        for i in range(min(mask_len - 1, len(labels))):
            labels[i] = -100

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = sample["prompt"]

        chosen_ids, chosen_labels = self._tokenize_pair(prompt, sample["chosen"])
        rejected_ids, rejected_labels = self._tokenize_pair(prompt, sample["rejected"])

        return {
            "chosen_input_ids": chosen_ids,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_ids,
            "rejected_labels": rejected_labels,
        }


def dpo_collate_fn(batch, pad_id=0):
    """Dynamically pads chosen and rejected pairs to the maximum length in the batch."""
    max_len = max(
        max(len(x["chosen_input_ids"]) for x in batch),
        max(len(x["rejected_input_ids"]) for x in batch),
    )

    b_c_ids, b_c_lbls, b_r_ids, b_r_lbls = [], [], [], []
    for x in batch:
        c_id, c_lbl = x["chosen_input_ids"], x["chosen_labels"]
        r_id, r_lbl = x["rejected_input_ids"], x["rejected_labels"]
        c_pad = max_len - len(c_id)
        if c_pad > 0:
            c_id = torch.cat([c_id, torch.full((c_pad,), pad_id, dtype=torch.long)])
            c_lbl = torch.cat([c_lbl, torch.full((c_pad,), -100, dtype=torch.long)])
        r_pad = max_len - len(r_id)
        if r_pad > 0:
            r_id = torch.cat([r_id, torch.full((r_pad,), pad_id, dtype=torch.long)])
            r_lbl = torch.cat([r_lbl, torch.full((r_pad,), -100, dtype=torch.long)])
        b_c_ids.append(c_id)
        b_c_lbls.append(c_lbl)
        b_r_ids.append(r_id)
        b_r_lbls.append(r_lbl)

    return {
        "chosen_input_ids": torch.stack(b_c_ids),
        "chosen_labels": torch.stack(b_c_lbls),
        "rejected_input_ids": torch.stack(b_r_ids),
        "rejected_labels": torch.stack(b_r_lbls),
    }


# ── Training ──────────────────────────────────────────────────────────────────

def dpo_train(args):
    """
    Main DPO training function.
    """
    device = get_device()

    # Load policy model (start from SFT checkpoint)
    log.info(f"Loading policy model from {args.base_model}")
    policy = KanhaModel.from_pretrained(args.base_model)

    is_lora = getattr(args, "lora", False)
    if is_lora:
        from kanha.core.lora import LoRAModel
        log.info("Wrapping policy model in LoRAModel adapter")
        policy = LoRAModel(
            policy,
            rank=getattr(cfg.lora, "rank", 8),
            alpha=getattr(cfg.lora, "alpha", 16.0),
            dropout=getattr(cfg.lora, "dropout", 0.05),
            target_modules=getattr(cfg.lora, "target_modules", ["q_proj", "v_proj"]),
        )
        policy.print_trainable_params()
        if getattr(args, "lr", 5e-6) <= 5e-6:
            default_lora_lr = 5e-5
            log.info(
                f"LoRA enabled: adjusting DPO learning rate from {args.lr} to {default_lora_lr} "
                f"(LoRA adapters require higher LR than full fine-tuning)"
            )
            args.lr = default_lora_lr

    policy.train()
    disable_dropout(policy)
    policy.to(device)

    # Reference model: when LoRA is used, we disable adapters dynamically during reference pass (saves ~50% VRAM)
    ref = None
    if not is_lora:
        log.info("Creating frozen reference model...")
        ref = KanhaModel.from_pretrained(args.base_model)
        ref.eval()
        disable_dropout(ref)
        ref.to(device)
        for p in ref.parameters():
            p.requires_grad = False
    else:
        log.info("LoRA enabled: Reusing base model weights with disabled adapter as reference (0 duplicate VRAM).")

    # Tokenizer
    tokenizer = KanhaTokenizer()

    # Dataset — use LAMDPODataset for lam-dpo, regular DPODataset for dpo
    lam_mode = getattr(args, "lam_mode", False)
    if not lam_mode and os.path.exists(args.data):
        try:
            with open(args.data, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        first_obj = json.loads(line)
                        if "goal" in first_obj or "### Goal" in first_obj.get("prompt", ""):
                            lam_mode = True
                        break
        except Exception:
            pass

    if lam_mode:
        from kanha.lam.trajectory import LAMDPODataset
        dataset = LAMDPODataset(args.data, tokenizer, max_len=cfg.model.max_seq_len)
    else:
        dataset = DPODataset(args.data, tokenizer, max_len=cfg.model.max_seq_len)

    # Train / Val split for held-out reward accuracy tracking
    val_fraction = getattr(args, "val_fraction", 0.1)
    val_size = max(1, int(len(dataset) * val_fraction)) if len(dataset) > 10 else 0
    train_size = len(dataset) - val_size

    if val_size > 0:
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        train_dataset, val_dataset = dataset, None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0,
        collate_fn=dpo_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0,
        collate_fn=dpo_collate_fn,
    ) if val_dataset else None

    # Optimizer — separate weight decay for LoRA / biases / norms
    dpo_cfg = getattr(cfg, 'dpo', None)
    dpo_weight_decay = getattr(dpo_cfg, 'weight_decay', 0.0) if dpo_cfg else 0.0
    dpo_grad_clip = getattr(dpo_cfg, 'grad_clip', 1.0) if dpo_cfg else 1.0

    decay_params = []
    nodecay_params = []
    for name, param in policy.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "lora_" in name or "bias" in name or "norm" in name:
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": dpo_weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95))

    # DPO hyperparameters
    beta = getattr(args, "beta", getattr(dpo_cfg, 'beta', 0.1) if dpo_cfg else 0.1)
    length_norm = getattr(args, "length_norm", False) or (getattr(dpo_cfg, "length_norm", False) if dpo_cfg else False)
    effective_beta = beta

    grad_accum_steps = getattr(args, "grad_accum_steps", None) or (
        getattr(dpo_cfg, "grad_accum_steps", 4) if dpo_cfg else 4
    )
    grad_accum_steps = max(1, int(grad_accum_steps))

    total_steps = math.ceil(len(train_loader) / grad_accum_steps) * args.epochs
    warmup_steps = min(50, max(1, total_steps // 10))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(1.0, max(0.0, progress))
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ensure_dir(args.output)
    best_val_score = float("-inf")
    best_val_acc = -1.0
    best_val_margin = float("-inf")

    log.info(f"DPO Training | epochs={args.epochs} | lr={args.lr} | beta={beta} | "
             f"grad_accum={grad_accum_steps} | length_norm={length_norm} | train={train_size} | val={val_size}")

    for epoch in range(args.epochs):
        epoch_loss = 0
        n_batches = 0
        optimizer.zero_grad()
        accum_count = 0

        pbar = tqdm(train_loader, desc=f"DPO Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            # Guard against batches containing truncated sequences with no supervision tokens
            valid_mask = (batch["chosen_labels"] != -100).any(dim=-1) & (batch["rejected_labels"] != -100).any(dim=-1)
            if not valid_mask.any():
                is_last_batch = (batch_idx + 1 == len(train_loader))
                if is_last_batch and accum_count > 0:
                    rem = accum_count % grad_accum_steps
                    if rem != 0:
                        scale_factor = grad_accum_steps / rem
                        for p in policy.parameters():
                            if p.grad is not None:
                                p.grad.data.mul_(scale_factor)
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), dpo_grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    accum_count = 0
                continue
            if not valid_mask.all():
                batch = {k: v[valid_mask] for k, v in batch.items()}

            # Compute log probs for chosen/rejected under policy
            policy_chosen_logps = compute_log_probs(
                policy, batch["chosen_input_ids"], batch["chosen_labels"], device, length_norm=length_norm
            )
            policy_rejected_logps = compute_log_probs(
                policy, batch["rejected_input_ids"], batch["rejected_labels"], device, length_norm=length_norm
            )

            # Compute log probs under reference model (eval mode, no grad)
            with torch.no_grad():
                policy.eval()
                if ref is not None:
                    ref_chosen_logps = compute_log_probs(
                        ref, batch["chosen_input_ids"], batch["chosen_labels"], device, length_norm=length_norm
                    )
                    ref_rejected_logps = compute_log_probs(
                        ref, batch["rejected_input_ids"], batch["rejected_labels"], device, length_norm=length_norm
                    )
                else:
                    # Dynamically disable LoRA adapter to evaluate reference on policy base weights without duplicate VRAM
                    with policy.disable_adapter():
                        ref_chosen_logps = compute_log_probs(
                            policy, batch["chosen_input_ids"], batch["chosen_labels"], device, length_norm=length_norm
                        )
                        ref_rejected_logps = compute_log_probs(
                            policy, batch["rejected_input_ids"], batch["rejected_labels"], device, length_norm=length_norm
                        )
                policy.train()
                disable_dropout(policy)

            loss, chosen_rewards, rejected_rewards = dpo_loss(
                policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=effective_beta
            )
            reward_margin = (chosen_rewards - rejected_rewards).mean().item()
            reward_acc = (chosen_rewards > rejected_rewards).float().mean().item()

            # Backward with gradient accumulation
            loss_scaled = loss / grad_accum_steps
            loss_scaled.backward()
            accum_count += 1

            is_accum_boundary = (accum_count % grad_accum_steps == 0)
            is_last_batch = (batch_idx + 1 == len(train_loader))

            if is_accum_boundary or (is_last_batch and accum_count > 0):
                rem = accum_count % grad_accum_steps
                if is_last_batch and rem != 0:
                    scale_factor = grad_accum_steps / rem
                    for p in policy.parameters():
                        if p.grad is not None:
                            p.grad.data.mul_(scale_factor)

                torch.nn.utils.clip_grad_norm_(policy.parameters(), dpo_grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum_count = 0

            epoch_loss += loss.item()
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "margin": f"{reward_margin:.3f}",
                "acc": f"{reward_acc:.1%}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        avg_loss = epoch_loss / max(n_batches, 1)
        log.info(f"DPO Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

        # Held-out validation evaluation
        if val_loader:
            policy.eval()
            val_margins, val_accs = [], []
            with torch.no_grad():
                for vbatch in val_loader:
                    v_mask = (vbatch["chosen_labels"] != -100).any(dim=-1) & (vbatch["rejected_labels"] != -100).any(dim=-1)
                    if not v_mask.any():
                        continue
                    if not v_mask.all():
                        vbatch = {k: v[v_mask] for k, v in vbatch.items()}
                    p_c = compute_log_probs(policy, vbatch["chosen_input_ids"], vbatch["chosen_labels"], device, length_norm=length_norm)
                    p_r = compute_log_probs(policy, vbatch["rejected_input_ids"], vbatch["rejected_labels"], device, length_norm=length_norm)
                    if ref is not None:
                        r_c = compute_log_probs(ref, vbatch["chosen_input_ids"], vbatch["chosen_labels"], device, length_norm=length_norm)
                        r_r = compute_log_probs(ref, vbatch["rejected_input_ids"], vbatch["rejected_labels"], device, length_norm=length_norm)
                    else:
                        with policy.disable_adapter():
                            r_c = compute_log_probs(policy, vbatch["chosen_input_ids"], vbatch["chosen_labels"], device, length_norm=length_norm)
                            r_r = compute_log_probs(policy, vbatch["rejected_input_ids"], vbatch["rejected_labels"], device, length_norm=length_norm)
                    c_rew = effective_beta * (p_c - r_c)
                    r_rew = effective_beta * (p_r - r_r)
                    val_margins.append((c_rew - r_rew).mean().item())
                    val_accs.append((c_rew > r_rew).float().mean().item())

            avg_val_acc = sum(val_accs) / max(len(val_accs), 1)
            avg_val_margin = sum(val_margins) / max(len(val_margins), 1)
            composite_score = avg_val_acc * 100.0 + avg_val_margin
            log.info(
                f"DPO Epoch {epoch+1} Val | Reward Acc: {avg_val_acc:.1%} | "
                f"Margin: {avg_val_margin:.3f} | Score: {composite_score:.3f}"
            )

            if composite_score > best_val_score or (avg_val_acc > best_val_acc):
                best_val_score = composite_score
                best_val_acc = avg_val_acc
                best_val_margin = avg_val_margin
                best_ckpt = os.path.join(args.output, "dpo_best.pt")
                policy.save_pretrained(best_ckpt)
                log.info(
                    f"New best DPO model saved to {best_ckpt} (Val Acc: {best_val_acc:.1%}, "
                    f"Margin: {best_val_margin:.3f}, Score: {best_val_score:.3f})"
                )

            policy.train()
            disable_dropout(policy)

        epoch_ckpt = os.path.join(args.output, f"dpo_epoch{epoch+1}.pt")
        policy.save_pretrained(epoch_ckpt)

    # Save final
    save_path = os.path.join(args.output, "dpo_final.pt")
    policy.save_pretrained(save_path)
    log.info(f"DPO model saved to {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DPO alignment training")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="models/finetuned/")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps (default 4)")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--length_norm", action="store_true", help="Normalize sequence log probs by sequence length to reduce length bias")
    parser.add_argument("--lam_mode", action="store_true", help="Enable LAM DPO mode for agent trajectories")
    parser.add_argument("--lora", action="store_true", help="Enable LoRA parameter-efficient training")
    cli_args = parser.parse_args()
    dpo_train(cli_args)
