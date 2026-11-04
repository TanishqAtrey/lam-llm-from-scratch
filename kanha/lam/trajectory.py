import json
import torch
from torch.utils.data import Dataset
from kanha.core.tokenizer import KanhaTokenizer
from kanha.utils.logging import get_logger

logger = get_logger(__name__)

GOAL_MARKER = '### Goal\n'
ACTION_MARKER = '### Action\n'
OBSERVATION_MARKER = '### Observation\n'

def format_trajectory(goal: str, steps: list[dict]) -> str:
    parts = [f"{GOAL_MARKER}{goal}\n"]
    for step in steps:
        parts.append(f"{ACTION_MARKER}{step['action']}\n")
        parts.append(f"{OBSERVATION_MARKER}{step['observation']}\n")
    return "".join(parts)

def format_step_prompt(goal: str, history: list[dict], include_trailing_action_marker=True) -> str:
    parts = [f"{GOAL_MARKER}{goal}\n"]
    for step in history:
        parts.append(f"{ACTION_MARKER}{step['action']}\n")
        parts.append(f"{OBSERVATION_MARKER}{step['observation']}\n")
    if include_trailing_action_marker:
        parts.append(ACTION_MARKER)
    return "".join(parts)



class LAMDPODataset(Dataset):
    def __init__(self, data_path: str, tokenizer: KanhaTokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                ex = json.loads(line)
                if 'chosen' in ex and 'rejected' in ex:
                    self.examples.append(ex)
                    
        logger.info(f"Loaded {len(self.examples)} DPO examples from {data_path}")

    def __len__(self):
        return len(self.examples)
        
    def _process_one(self, prompt, response):
        full_text = prompt + response
        prompt_tokens = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        mask_len = len(prompt_tokens)
        
        full_ids = self.tokenizer.encode(
            full_text, add_bos=True, add_eos=True
        )
        if len(full_ids) > self.max_len:
            full_ids = full_ids[:self.max_len]
        
        # Standard shifted input/target pairs
        input_ids = full_ids[:-1]
        labels = full_ids[1:]
        
        # Mask prompt tokens
        for i in range(min(mask_len - 1, len(labels))):
            labels[i] = -100
            
        return input_ids, labels


    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = ex.get('prompt') or format_step_prompt(ex.get('goal', ''), ex.get('history', []))
        chosen = ex['chosen']
        rejected = ex['rejected']
        
        c_tokens, c_labels = self._process_one(prompt, chosen)
        r_tokens, r_labels = self._process_one(prompt, rejected)
        
        return {
            'chosen_input_ids': torch.tensor(c_tokens, dtype=torch.long),
            'chosen_labels': torch.tensor(c_labels, dtype=torch.long),
            'rejected_input_ids': torch.tensor(r_tokens, dtype=torch.long),
            'rejected_labels': torch.tensor(r_labels, dtype=torch.long)
        }


def lam_dpo_collate_fn(batch, pad_id=0):
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

