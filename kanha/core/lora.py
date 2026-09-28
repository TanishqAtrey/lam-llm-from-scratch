"""
kanha/core/lora.py
LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
"""

import math
import torch
import torch.nn as nn
from typing import List, Optional
from kanha.utils.logging import get_logger

log = get_logger(__name__)


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = None, dropout: float = 0.0):
        super().__init__()
        self.linear  = linear
        self.rank    = rank
        self.alpha   = alpha or float(rank)
        self.scaling = self.alpha / self.rank

        d_in  = linear.in_features
        d_out = linear.out_features
        device = linear.weight.device

        for param in self.linear.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(rank, d_in, device=device))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank, device=device))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)

        if x.dtype in (torch.float16, torch.bfloat16):
            lora_x   = self.lora_dropout(x.float())
            lora_out = (lora_x @ self.lora_A.T) @ self.lora_B.T
            return base_out + (lora_out * self.scaling).to(x.dtype)
        else:
            lora_x   = self.lora_dropout(x)
            lora_out = (lora_x @ self.lora_A.T) @ self.lora_B.T
            return base_out + lora_out * self.scaling

    def merge(self) -> nn.Linear:
        merged = nn.Linear(self.linear.in_features, self.linear.out_features,
                           bias=self.linear.bias is not None)
        merged.weight.data = self.linear.weight.data + (self.scaling * self.lora_B @ self.lora_A)
        if self.linear.bias is not None:
            merged.bias.data = self.linear.bias.data.clone()
        return merged


class LoRAModel(nn.Module):
    def __init__(self, model: nn.Module, rank: int = 8, alpha: float = None,
                 lora_dropout: float = 0.05, target_modules: Optional[List[str]] = None):
        super().__init__()
        self.model          = model
        self.rank           = rank
        self.alpha          = alpha or float(rank)
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        for param in self.model.parameters():
            param.requires_grad = False
        self._inject_lora(lora_dropout)

    def _inject_lora(self, dropout: float):
        replaced = 0
        for name, module in self.model.named_modules():
            parent_name, child_name = self._split_name(name)
            if child_name not in self.target_modules:
                continue
            if not isinstance(module, nn.Linear):
                continue
            parent = self._get_module(self.model, parent_name)
            setattr(parent, child_name,
                    LoRALinear(module, rank=self.rank, alpha=self.alpha, dropout=dropout))
            replaced += 1
        log.info(f"LoRA injected into {replaced} linear layers (rank={self.rank}, target={self.target_modules})")

    @staticmethod
    def _split_name(name):
        parts = name.rsplit(".", 1)
        return (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])

    @staticmethod
    def _get_module(model, name):
        if not name:
            return model
        for part in name.split("."):
            model = getattr(model, part)
        return model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def print_trainable_params(self):
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(f"LoRA params: {trained:,} trainable / {total:,} total ({100.*trained/total:.2f}%)")

    def merge_and_unload(self):
        for name, module in self.model.named_modules():
            parent_name, child_name = self._split_name(name)
            if not isinstance(module, LoRALinear):
                continue
            parent = self._get_module(self.model, parent_name)
            setattr(parent, child_name, module.merge())
        log.info("LoRA weights merged into base model.")
        return self.model

    def save_lora(self, path: str):
        lora_state = {k: v for k, v in self.state_dict().items()
                      if "lora_A" in k or "lora_B" in k}
        torch.save({"lora_state": lora_state, "rank": self.rank,
                    "alpha": self.alpha, "target_modules": self.target_modules}, path)
        log.info(f"LoRA weights saved to {path}")

    def load_lora(self, path: str):
        checkpoint = torch.load(path, map_location="cpu")
        self.load_state_dict(checkpoint["lora_state"], strict=False)
        log.info(f"LoRA loaded from {path}")