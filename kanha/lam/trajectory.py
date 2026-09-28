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

def expand_to_steps(trajectory: dict) -> list[dict]:
    steps = trajectory.get('steps', [])
    goal = trajectory.get('goal', '')
    
    expanded = []
    for i in range(len(steps)):
        prompt = format_step_prompt(goal, steps[:i])
        response = steps[i]['action'] + '\n'
        expanded.append({
            'prompt': prompt,
            'response': response
        })
    return expanded

class LAMDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: KanhaTokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                traj = json.loads(line)
                self.examples.extend(expand_to_steps(traj))
                
        logger.info(f"Loaded {len(self.examples)} examples from {data_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = ex['prompt']
        response = ex['response']
        
        full_text = prompt + response
        
        # Tokenize the prompt alone to find length for masking
        prompt_tokens = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        mask_len = len(prompt_tokens)
        
        # Tokenize full text
        full_ids = self.tokenizer.encode(
            full_text, add_bos=True, add_eos=True, max_length=self.max_len
        )
        
        # Standard shifted input/target pairs (same as SFTDataset)
        input_ids = full_ids[:-1]
        labels = full_ids[1:]
        
        # Mask prompt tokens in labels (set to -100)
        for i in range(min(mask_len - 1, len(labels))):
            labels[i] = -100
            
        # Pad/truncate to max_len - 1
        seq_len = self.max_len - 1
        pad_len = seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.PAD_ID] * pad_len
            labels = labels + [-100] * pad_len
        else:
            input_ids = input_ids[:seq_len]
            labels = labels[:seq_len]
            
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


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
            full_text, add_bos=True, add_eos=True, max_length=self.max_len
        )
        
        # Standard shifted input/target pairs
        input_ids = full_ids[:-1]
        labels = full_ids[1:]
        
        # Mask prompt tokens
        for i in range(min(mask_len - 1, len(labels))):
            labels[i] = -100
            
        # Pad/truncate to max_len - 1
        seq_len = self.max_len - 1
        pad_len = seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.PAD_ID] * pad_len
            labels = labels + [-100] * pad_len
        else:
            input_ids = input_ids[:seq_len]
            labels = labels[:seq_len]
            
        return input_ids, labels


    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = ex['prompt']
        chosen = ex['chosen'] + '\n'
        rejected = ex['rejected'] + '\n'
        
        c_tokens, c_labels = self._process_one(prompt, chosen)
        r_tokens, r_labels = self._process_one(prompt, rejected)
        
        return {
            'chosen_input_ids': torch.tensor(c_tokens, dtype=torch.long),
            'chosen_labels': torch.tensor(c_labels, dtype=torch.long),
            'rejected_input_ids': torch.tensor(r_tokens, dtype=torch.long),
            'rejected_labels': torch.tensor(r_labels, dtype=torch.long)
        }
