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
        beta                  : temperature parameter (lower = stronger preference)
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss


def compute_log_probs(model, input_ids, labels, device):
    """
    Computes per-sequence log probabilities for the response tokens only.

    IMPORTANT: input_ids and labels are ALREADY shifted by the dataset
    (input_ids = full[:-1], labels = full[1:]), so NO additional shifting
    is needed here. logits[i] should predict labels[i] directly.

    Args:
        model     : KanhaModel
        input_ids : (B, S) input token ids (already shifted)
        labels    : (B, S) target labels (already shifted, -100 for masked)
        device    : torch device

    Returns:
        (B,) tensor of summed log probabilities per sequence
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

    # Sum log probs per sequence
    return per_token_logps.sum(dim=-1)


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

        # Pad
        seq_len = self.max_len - 1
        pad_len = seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.PAD_ID] * pad_len
            labels = labels + [-100] * pad_len
        else:
            input_ids = input_ids[:seq_len]
            labels = labels[:seq_len]

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


# ── Training ──────────────────────────────────────────────────────────────────

def dpo_train(args):
    """
    Main DPO training function.
    """
    device = get_device()

    # Load policy model (start from SFT checkpoint)
    log.info(f"Loading policy model from {args.base_model}")
    policy = KanhaModel.from_pretrained(args.base_model)
    policy.train()
    policy.to(device)

    # Create reference model (frozen copy of policy)
    log.info("Creating frozen reference model...")
    ref = KanhaModel.from_pretrained(args.base_model)
    ref.eval()
    ref.to(device)
    for p in ref.parameters():
        p.requires_grad = False

    # Tokenizer
    tokenizer = KanhaTokenizer()

    # Dataset
    dataset = DPODataset(args.data, tokenizer, max_len=cfg.model.max_seq_len)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0
    )

    # Optimizer — very low LR for DPO (1e-6 to 5e-6)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    # DPO hyperparameters
    beta = getattr(args, "beta", 0.1)

    ensure_dir(args.output)
    global_step = 0

    log.info(f"DPO Training | epochs={args.epochs} | lr={args.lr} | beta={beta}")

    for epoch in range(args.epochs):
        epoch_loss = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"DPO Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            # Compute log probs for chosen/rejected under policy
            policy_chosen_logps = compute_log_probs(
                policy, batch["chosen_input_ids"], batch["chosen_labels"], device
            )
            policy_rejected_logps = compute_log_probs(
                policy, batch["rejected_input_ids"], batch["rejected_labels"], device
            )

            # Compute log probs under reference model (no grad)
            with torch.no_grad():
                ref_chosen_logps = compute_log_probs(
                    ref, batch["chosen_input_ids"], batch["chosen_labels"], device
                )
                ref_rejected_logps = compute_log_probs(
                    ref, batch["rejected_input_ids"], batch["rejected_labels"], device
                )

            # DPO loss
            loss = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                beta=beta,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / max(n_batches, 1)
        log.info(f"DPO Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

    # Save
    save_path = os.path.join(args.output, "dpo_final.pt")
    policy.save_pretrained(save_path)
    log.info(f"DPO model saved to {save_path}")
