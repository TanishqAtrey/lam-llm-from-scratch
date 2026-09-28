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
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"

        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads or max(1, n_heads // 4)   # default: n_heads/4
        self.n_rep      = self.n_heads // self.n_kv_heads       # repetitions per KV group
        self.head_dim   = dim // n_heads
        self.scale      = self.head_dim ** -0.5

        assert n_heads % self.n_kv_heads == 0, \
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"

        # Q has n_heads, K/V have n_kv_heads
        self.q_proj  = nn.Linear(dim, n_heads    * self.head_dim, bias=False)
        self.k_proj  = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj  = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj= nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.rope    = RotaryEmbedding(self.head_dim)
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
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K — FIX: pass offset for correct KV-cache positional encoding
        offset = kv_cache[0].shape[2] if kv_cache is not None else 0
        cos, sin = self.rope(S, offset=offset)
        q, k = apply_rope(q, k, cos, sin)

        # KV cache concatenation
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_kv_cache = (k, v)

        # Expand KV to match query heads
        k = self._repeat_kv(k)   # (B, n_heads, S_full, head_dim)
        v = self._repeat_kv(v)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn + mask

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        out = self.out_proj(out)
        return out, new_kv_cache


class TransformerBlockGQA(nn.Module):
    """
    Transformer block using GQA instead of MHA.
    Drop-in replacement for TransformerBlock in model.py.
    """
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int,
                 ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn  = GroupedQueryAttention(dim, n_heads, n_kv_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn   = FeedForward(dim, ff_dim, dropout)

    def forward(self, x, mask=None, kv_cache=None):
        attn_out, new_cache = self.attn(self.norm1(x), mask=mask, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_cache