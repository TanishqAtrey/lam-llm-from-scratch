"""
kanha/core/model.py
KANHA Transformer — optimised for Apple Silicon MPS.

Key optimisations:
  1. Causal mask pre-registered as a buffer (never re-allocated)
  2. use_cache=False during training — no KV tuple bookkeeping
  3. use_cache=True during inference — standard KV cache
"""

import math
import torch
import torch.nn as nn
from typing import Optional, List, Tuple

from kanha.core.layers import TransformerBlock, RMSNorm
from kanha.core.layers_gqa import TransformerBlockGQA
from kanha.utils.config import cfg
from kanha.utils.helpers import get_device, count_parameters
from kanha.utils.logging import get_logger

log = get_logger(__name__)


class KanhaModel(nn.Module):

    def __init__(
        self,
        vocab_size: int  = None,
        dim: int         = None,
        n_layers: int    = None,
        n_heads: int     = None,
        ff_dim: int      = None,
        max_seq_len: int = None,
        dropout: float   = None,
        n_kv_heads: int  = None,
    ):
        super().__init__()

        mc = cfg.model
        self.vocab_size  = vocab_size  if vocab_size  is not None else mc.vocab_size
        self.dim         = dim         if dim         is not None else mc.dim
        self.n_layers    = n_layers    if n_layers    is not None else mc.n_layers
        self.n_heads     = n_heads     if n_heads     is not None else mc.n_heads
        self.ff_dim      = ff_dim      if ff_dim      is not None else mc.ff_dim
        self.max_seq_len = max_seq_len if max_seq_len is not None else mc.max_seq_len
        self.dropout_p   = dropout     if dropout     is not None else mc.dropout
        self.n_kv_heads  = n_kv_heads  if n_kv_heads  is not None else getattr(mc, "n_kv_heads", None)

        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.drop      = nn.Dropout(self.dropout_p)

        if self.n_kv_heads is not None and self.n_kv_heads < self.n_heads:
            self.layers = nn.ModuleList([
                TransformerBlockGQA(self.dim, self.n_heads, self.n_kv_heads, self.ff_dim, self.dropout_p, max_seq_len=self.max_seq_len)
                for _ in range(self.n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                TransformerBlock(self.dim, self.n_heads, self.ff_dim, self.dropout_p, max_seq_len=self.max_seq_len)
                for _ in range(self.n_layers)
            ])

        self.norm    = RMSNorm(self.dim)
       
        # Weight tying
        self.lm_head = nn.Linear(self.dim, self.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

        # Residual projection scaling (GPT-2 / Megatron-LM style)
        # Scales residual branch weights by 1/sqrt(2 * n_layers) to stabilize variance growth
        res_scale = 1.0 / math.sqrt(2.0 * self.n_layers)
        for layer in self.layers:
            if hasattr(layer, "attn") and hasattr(layer.attn, "out_proj"):
                layer.attn.out_proj.weight.data.mul_(res_scale)
            if hasattr(layer, "ffn") and hasattr(layer.ffn, "w2"):
                layer.ffn.w2.weight.data.mul_(res_scale)

        log.info(f"KanhaModel | params: {count_parameters(self):,}")

    def _init_weights(self, module):
        # GPT-2 / LLaMA-style init: std = 1/sqrt(dim) for better gradient flow
        # at scale. For dim=512, std ≈ 0.044; for dim=768, std ≈ 0.036.
        std = 1.0 / (self.dim ** 0.5)
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple]] = None,
        offset: Optional[int] = None,
    ):
        """
        Args:
            input_ids : (B, S)
            targets   : (B, S) for training loss
            kv_caches : per-layer KV caches — only pass during inference
            offset    : cumulative sequence position for RoPE during autoregressive inference

        Returns:
            logits    : (B, S, vocab_size)
            loss      : scalar or None
            new_caches: list of (k, v) tuples — None entries during training
        """
        x = self.drop(self.token_emb(input_ids))

        # use_cache only when kv_caches explicitly provided (inference path)
        use_cache = kv_caches is not None
        new_caches = []

        for i, layer in enumerate(self.layers):
            cache_i = kv_caches[i] if use_cache else None
            x, new_cache = layer(x, use_cache=use_cache, kv_cache=cache_i, offset=offset)
            new_caches.append(new_cache)   # None during training — no allocation

        x      = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            if (targets != -100).any():
                loss = nn.functional.cross_entropy(
                    logits.view(-1, self.vocab_size),
                    targets.view(-1),
                    ignore_index=-100,
                )
            else:
                loss = logits.sum() * 0.0

        return logits, loss, new_caches

    @classmethod
    def from_pretrained(cls, path: str, device: Optional[torch.device] = None) -> "KanhaModel":
        device = device or get_device()
        import os
        if os.path.isdir(path):
            candidates = [
                os.path.join(path, "final_model.pt"),
                os.path.join(path, "sft_final.pt"),
                os.path.join(path, "dpo_best.pt"),
                os.path.join(path, "dpo_final.pt"),
                os.path.join(path, "lam_final.pt"),
                os.path.join(path, "best_model.pt"),
                os.path.join(path, "model.pt"),
            ]
            chosen = None
            for cand in candidates:
                if os.path.isfile(cand):
                    chosen = cand
                    break
            if chosen is None:
                pt_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(".pt")])
                if pt_files:
                    chosen = pt_files[-1]
                else:
                    raise FileNotFoundError(f"No .pt checkpoint found in directory: {path}")
            path = chosen
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model      = cls(**checkpoint.get("model_config", {}))
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
        log.info(f"Loaded model from {path}")
        return model

    def save_pretrained(self, path: str, step: int = None):
        cfg_dict = {
            "vocab_size":  self.vocab_size,
            "dim":         self.dim,
            "n_layers":    self.n_layers,
            "n_heads":     self.n_heads,
            "ff_dim":      self.ff_dim,
            "max_seq_len": self.max_seq_len,
            "dropout":     self.dropout_p,
        }
        if self.n_kv_heads is not None:
            cfg_dict["n_kv_heads"] = self.n_kv_heads
        data = {
            "model_config": cfg_dict,
            "model_state": self.state_dict(),
        }
        if step is not None:
            data["step"] = step
        torch.save(data, path)
        log.info(f"Model saved to {path}")
