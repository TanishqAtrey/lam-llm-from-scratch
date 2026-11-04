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


# ── DPO Loss ──────────────────────────────────────────────────────────────────

def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Computes DPO loss.

    L_DPO = -log sigmoid(beta * (log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x)))

    Args:
        policy_chosen_logps   : log probs of chosen under current policy
        policy_rejected_logps : log probs of rejected under current policy
        ref_chosen_logps      : log probs of chosen under reference model
        ref_rejected_logps    : log probs of rejected under reference model
        beta                  : temperature parameter (higher = stronger preference signal / steeper policy update; lower = stronger regularization to reference model)
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss


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
                    self.samples.append(obj)

        log.info(f"Loaded {len(self.samples):,} DPO samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def _tokenize_pair(self, prompt: str, response: str):
        """Tokenizes a (prompt, response) pair with instruction masking."""
        # Strip "Human: " prefix if present (common in HH-RLHF format)
        clean_prompt = prompt
        if clean_prompt.lower().startswith("human:"):
            clean_prompt = clean_prompt[6:].strip()

        # Format with canonical template
        full_text = self.prompt_builder.build_training_pair(clean_prompt, response)

        # Tokenize
        full_ids = self.tokenizer.encode(
            full_text, add_bos=True, add_eos=True, max_length=self.max_len
        )

        # Compute instruction length for masking
        instruction_prefix = f"### Instruction:\n{clean_prompt}\n\n### Response:\n"
        instruction_ids = self.tokenizer.encode(
            instruction_prefix, add_bos=True, add_eos=False
        )
        mask_len = len(instruction_ids)

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

    if getattr(args, "lora", False):
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

    policy.train()
    disable_dropout(policy)
    policy.to(device)

    # Create reference model (frozen copy of policy)
    log.info("Creating frozen reference model...")
    ref = KanhaModel.from_pretrained(args.base_model)
    ref.eval()
    disable_dropout(ref)
    ref.to(device)
    for p in ref.parameters():
        p.requires_grad = False

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
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0,
        collate_fn=dpo_collate_fn,
    )

    # Optimizer — very low LR for DPO
    dpo_cfg = getattr(cfg, 'dpo', None)
    dpo_weight_decay = getattr(dpo_cfg, 'weight_decay', 0.0) if dpo_cfg else 0.0
    dpo_grad_clip = getattr(dpo_cfg, 'grad_clip', 1.0) if dpo_cfg else 1.0

    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=dpo_weight_decay,
        betas=(0.9, 0.95),
    )

    # DPO hyperparameters
    beta = getattr(args, "beta", getattr(dpo_cfg, 'beta', 0.1) if dpo_cfg else 0.1)
    length_norm = getattr(args, "length_norm", False) or (getattr(dpo_cfg, "length_norm", False) if dpo_cfg else False)

    total_steps = len(dataloader) * args.epochs
    warmup_steps = min(50, max(1, total_steps // 10))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(1.0, max(0.0, progress))
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ensure_dir(args.output)

    log.info(f"DPO Training | epochs={args.epochs} | lr={args.lr} | beta={beta} | length_norm={length_norm}")

    for epoch in range(args.epochs):
        epoch_loss = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"DPO Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            # Compute log probs for chosen/rejected under policy
            policy_chosen_logps = compute_log_probs(
                policy, batch["chosen_input_ids"], batch["chosen_labels"], device, length_norm=length_norm
            )
            policy_rejected_logps = compute_log_probs(
                policy, batch["rejected_input_ids"], batch["rejected_labels"], device, length_norm=length_norm
            )

            # Compute log probs under reference model (no grad)
            with torch.no_grad():
                ref_chosen_logps = compute_log_probs(
                    ref, batch["chosen_input_ids"], batch["chosen_labels"], device, length_norm=length_norm
                )
                ref_rejected_logps = compute_log_probs(
                    ref, batch["rejected_input_ids"], batch["rejected_labels"], device, length_norm=length_norm
                )

            chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
            rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
            reward_margin = (chosen_rewards - rejected_rewards).mean().item()
            reward_acc = (chosen_rewards > rejected_rewards).float().mean().item()

            loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), dpo_grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

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
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--length_norm", action="store_true", help="Normalize sequence log probs by sequence length to reduce length bias")
    parser.add_argument("--lam_mode", action="store_true", help="Enable LAM DPO mode for agent trajectories")
    cli_args = parser.parse_args()
    dpo_train(cli_args)
