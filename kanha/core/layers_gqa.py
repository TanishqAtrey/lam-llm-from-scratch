"""
kanha/core/layers_gqa.py
Grouped Query Attention (GQA) — used by Mistral, LLaMA-2-70B.

Instead of n_heads Q/K/V triplets (MHA), GQA uses:
  - n_heads query heads (full resolution queries)
  - n_kv_heads key/value heads (shared, fewer heads)
  - each KV head is shared by (n_heads // n_kv_heads) query heads

Benefits:
  - 3–5× faster inference (smaller KV cache)
  - ½–¼ KV memory footprint
  - Same quality as MHA when n_kv_heads >= 4

Example: n_heads=8, n_kv_heads=2 → 4 queries share each KV pair
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from kanha.core.layers import RotaryEmbedding, apply_rope, RMSNorm, FeedForward


class GroupedQueryAttention(nn.Module):
    """
    GQA: Multi-query attention with grouped key/value heads.

    Args:
        dim        : model hidden dimension
        n_heads    : number of query heads
        n_kv_heads : number of key/value heads (must divide n_heads evenly)
        dropout    : attention dropout
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int = None,
        dropout: float = 0.0,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"

        self.n_heads     = n_heads
        self.n_kv_heads  = n_kv_heads or max(1, n_heads // 4)   # default: n_heads/4
        self.n_rep       = self.n_heads // self.n_kv_heads       # repetitions per KV group
        self.head_dim    = dim // n_heads
        self.scale       = self.head_dim ** -0.5
        self.max_seq_len = max_seq_len

        assert n_heads % self.n_kv_heads == 0, \
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"

        # Q has n_heads, K/V have n_kv_heads
        self.q_proj  = nn.Linear(dim, n_heads    * self.head_dim, bias=False)
        self.k_proj  = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj  = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj= nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.rope    = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expands KV heads to match the number of query heads.
        (B, n_kv_heads, S, head_dim) → (B, n_heads, S, head_dim)
        """
        if self.n_rep == 1:
            return x
        B, n_kv, S, d = x.shape
        return (
            x[:, :, None, :, :]
             .expand(B, n_kv, self.n_rep, S, d)
             .reshape(B, n_kv * self.n_rep, S, d)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        offset: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K — pass offset for correct KV-cache positional encoding
        actual_offset = offset if offset is not None else (kv_cache[0].shape[2] if kv_cache is not None else 0)
        cos, sin = self.rope(S, offset=actual_offset)
        q, k = apply_rope(q, k, cos, sin)

        # KV cache concatenation (compact KV heads)
        new_kv_cache = None
        if use_cache:
            if kv_cache is not None:
                k_cache, v_cache = kv_cache
                k = torch.cat([k_cache, k], dim=2)
                v = torch.cat([v_cache, v], dim=2)
            if k.shape[2] > self.max_seq_len:
                k = k[:, :, -self.max_seq_len:, :]
                v = v[:, :, -self.max_seq_len:, :]
            new_kv_cache = (k, v)

        # Expand KV to match query heads
        k_rep = self._repeat_kv(k)   # (B, n_heads, S_full, head_dim)
        v_rep = self._repeat_kv(v)

        # ── SDPA: single fused kernel ──
        attn_mask = mask
        if attn_mask is None:
            if kv_cache is None:
                is_causal = S > 1
            else:
                is_causal = False
                if S > 1:
                    row_idx = torch.arange(S, device=x.device).unsqueeze(1)
                    col_idx = torch.arange(k_rep.shape[2], device=x.device).unsqueeze(0)
                    past_len = k_rep.shape[2] - S
                    attn_mask = col_idx <= (row_idx + past_len)
        else:
            is_causal = False

        dropout_p = self.dropout.p if self.training and x.device.type != 'mps' else 0.0

        out = F.scaled_dot_product_attention(
            q, k_rep, v_rep,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        out = self.out_proj(out)
        return out, new_kv_cache


class TransformerBlockGQA(nn.Module):
    """
    Transformer block using GQA instead of MHA.
    Drop-in replacement for TransformerBlock in model.py.
    """
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int,
                 ff_dim: int, dropout: float = 0.0, max_seq_len: int = 4096):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn  = GroupedQueryAttention(dim, n_heads, n_kv_heads, dropout, max_seq_len=max_seq_len)
        self.norm2 = RMSNorm(dim)
        self.ffn   = FeedForward(dim, ff_dim, dropout)

    def forward(self, x, mask=None, use_cache=False, kv_cache=None, offset: Optional[int] = None):
        attn_out, new_cache = self.attn(self.norm1(x), mask=mask, use_cache=use_cache, kv_cache=kv_cache, offset=offset)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_cache