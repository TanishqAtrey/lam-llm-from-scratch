"""
kanha/core/lora.py
LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
"""

import math
from contextlib import contextmanager
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
        self.adapter_enabled = True
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)
        if not getattr(self, "adapter_enabled", True) or self.scaling == 0.0:
            return base_out
        lora_x = self.lora_dropout(x).to(self.lora_A.dtype)
        lora_out = (lora_x @ self.lora_A.T) @ self.lora_B.T
        return base_out + (lora_out * self.scaling).to(x.dtype)

    def merge(self) -> nn.Linear:
        merged = nn.Linear(self.linear.in_features, self.linear.out_features,
                           bias=self.linear.bias is not None,
                           device=self.linear.weight.device,
                           dtype=self.linear.weight.dtype)
        delta = (self.scaling * self.lora_B @ self.lora_A).to(
            dtype=self.linear.weight.dtype, device=self.linear.weight.device
        )
        merged.weight.data = self.linear.weight.data + delta
        if self.linear.bias is not None:
            merged.bias.data = self.linear.bias.data.clone()
        return merged


class LoRAModel(nn.Module):
    def __init__(self, model: nn.Module, rank: int = None, alpha: float = None,
                 lora_dropout: float = None, target_modules: Optional[List[str]] = None,
                 dropout: float = None):
        super().__init__()
        self.model = model

        # Read defaults from config, fall back to sensible values
        try:
            from kanha.utils.config import cfg
            lora_cfg = getattr(cfg, 'lora', None)
        except Exception:
            lora_cfg = None

        if dropout is not None and lora_dropout is None:
            lora_dropout = dropout

        self.rank = rank if rank is not None else (getattr(lora_cfg, 'rank', 8) if lora_cfg else 8)
        self.alpha = alpha if alpha is not None else (getattr(lora_cfg, 'alpha', float(self.rank)) if lora_cfg else float(self.rank))
        _dropout = lora_dropout if lora_dropout is not None else (getattr(lora_cfg, 'dropout', 0.05) if lora_cfg else 0.05)

        if target_modules is not None:
            self.target_modules = target_modules
        elif lora_cfg and hasattr(lora_cfg, 'target_modules'):
            self.target_modules = list(lora_cfg.target_modules) if not isinstance(lora_cfg.target_modules, list) else lora_cfg.target_modules
        else:
            self.target_modules = ["q_proj", "v_proj"]

        for param in self.model.parameters():
            param.requires_grad = False
        self._inject_lora(_dropout)

    def _inject_lora(self, dropout: float):
        to_replace = []
        for name, module in self.model.named_modules():
            parent_name, child_name = self._split_name(name)
            if child_name not in self.target_modules:
                continue
            if not isinstance(module, nn.Linear):
                continue
            to_replace.append((parent_name, child_name, module))

        for parent_name, child_name, module in to_replace:
            parent = self._get_module(self.model, parent_name)
            setattr(parent, child_name,
                    LoRALinear(module, rank=self.rank, alpha=self.alpha, dropout=dropout))
        log.info(f"LoRA injected into {len(to_replace)} linear layers (rank={self.rank}, target={self.target_modules})")

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

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if "model" in self.__dict__ or ("_modules" in self.__dict__ and "model" in self._modules):
                return getattr(self.model, name)
            raise

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @contextmanager
    def disable_adapter(self):
        """Context manager that temporarily disables LoRA adapters (evaluating the base model)."""
        prev_states = {}
        for name, m in self.named_modules():
            if isinstance(m, LoRALinear):
                prev_states[name] = getattr(m, "adapter_enabled", True)
                m.adapter_enabled = False
        try:
            yield
        finally:
            for name, m in self.named_modules():
                if isinstance(m, LoRALinear):
                    m.adapter_enabled = prev_states.get(name, True)

    def set_adapter_enabled(self, enabled: bool = True):
        """Globally enables or disables all LoRA adapters."""
        for m in self.modules():
            if isinstance(m, LoRALinear):
                m.adapter_enabled = enabled

    def print_trainable_params(self):
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(f"LoRA params: {trained:,} trainable / {total:,} total ({100.*trained/total:.2f}%)")

    def merge_and_unload(self):
        to_merge = []
        for name, module in self.model.named_modules():
            parent_name, child_name = self._split_name(name)
            if not isinstance(module, LoRALinear):
                continue
            to_merge.append((parent_name, child_name, module))

        for parent_name, child_name, module in to_merge:
            parent = self._get_module(self.model, parent_name)
            setattr(parent, child_name, module.merge())
        log.info("LoRA weights merged into base model.")
        return self.model

    def save_pretrained(self, path: str, step: int = None):
        """
        Saves the model checkpoint with LoRA weights merged into the base weights.
        Non-destructive: does NOT mutate or unload the in-memory LoRA layers,
        so training can safely continue for subsequent epochs/steps.
        The resulting checkpoint is 100% compatible with KanhaModel.from_pretrained().
        """
        clean_state = {}
        for k, v in self.model.state_dict().items():
            if ".linear." in k:
                base_k = k.replace(".linear.", ".")
                clean_state[base_k] = v.clone()
            elif "lora_A" in k or "lora_B" in k:
                continue
            else:
                clean_state[k] = v.clone()

        for name, module in self.model.named_modules():
            if isinstance(module, LoRALinear):
                weight_key = f"{name}.weight"
                delta = (module.scaling * module.lora_B @ module.lora_A).to(
                    dtype=module.linear.weight.dtype, device=module.linear.weight.device
                )
                clean_state[weight_key] = module.linear.weight.data + delta

        cfg_dict = {
            "vocab_size":  self.model.vocab_size,
            "dim":         self.model.dim,
            "n_layers":    self.model.n_layers,
            "n_heads":     self.model.n_heads,
            "ff_dim":      self.model.ff_dim,
            "max_seq_len": self.model.max_seq_len,
            "dropout":     self.model.dropout_p,
        }
        if hasattr(self.model, "n_kv_heads") and self.model.n_kv_heads is not None:
            cfg_dict["n_kv_heads"] = self.model.n_kv_heads

        data = {
            "model_config": cfg_dict,
            "model_state": clean_state,
        }
        if step is not None:
            data["step"] = step
        torch.save(data, path)
        log.info(f"Model saved to {path} (compatible state_dict with LoRA applied, non-destructive)")

    def save_lora(self, path: str):
        lora_state = {k: v for k, v in self.state_dict().items()
                      if "lora_A" in k or "lora_B" in k}
        torch.save({"lora_state": lora_state, "rank": self.rank,
                    "alpha": self.alpha, "target_modules": self.target_modules}, path)
        log.info(f"LoRA weights saved to {path}")

    def load_lora(self, path: str):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        lora_state = checkpoint["lora_state"]
        incompatible = self.load_state_dict(lora_state, strict=False)
        if incompatible.unexpected_keys:
            log.warning(f"Unexpected keys while loading LoRA from {path}: {incompatible.unexpected_keys}")
        missing_lora = [k for k in incompatible.missing_keys if "lora_A" in k or "lora_B" in k]
        if missing_lora:
            log.warning(f"Missing LoRA adapter keys when loading from {path}: {missing_lora}")
        log.info(f"LoRA loaded from {path}")