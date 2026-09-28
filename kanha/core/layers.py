"""
kanha/core/layers.py
Transformer building blocks — optimised for Apple Silicon MPS.

Key optimisations over the original:
  1. F.scaled_dot_product_attention  — single fused kernel replacing
     3 matmuls + softmax + dropout (3-5× faster on MPS)
  2. Causal mask removed from attention — SDPA handles it with
     is_causal=True, no Python-side mask tensor needed
  3. KV cache only allocated when explicitly requested (use_cache=True)
     so training never pays for cache bookkeeping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ─── 1. RMSNorm ───────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ─── 2. Rotary Positional Embedding (RoPE) ────────────────────────────────────
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t     = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, seq_len: int, offset: int = 0):
        return (
            self.cos_cached[offset:offset + seq_len].unsqueeze(0).unsqueeze(0),
            self.sin_cached[offset:offset + seq_len].unsqueeze(0).unsqueeze(0),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


# ─── 3. Multi-Head Attention (with SDPA) ──────────────────────────────────────
class MultiHeadAttention(nn.Module):
    """
    Attention using F.scaled_dot_product_attention — the single biggest
    speed lever. One fused kernel replaces:
        matmul(Q, K) → scale → add mask → softmax → dropout → matmul(attn, V)

    is_causal=True tells SDPA to apply causal masking internally —
    no Python-side mask tensor is created or allocated.

    use_cache=True only during inference — training never builds the cache.
    """
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.dropout  = dropout

        self.q_proj   = nn.Linear(dim, dim, bias=False)
        self.k_proj   = nn.Linear(dim, dim, bias=False)
        self.v_proj   = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.rope     = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # RoPE — offset by cached context length during inference
        offset = kv_cache[0].shape[2] if kv_cache is not None else 0
        cos, sin = self.rope(S, offset=offset)
        q, k = apply_rope(q, k, cos, sin)

        # KV cache append (inference only — skipped entirely during training)
        new_cache = None
        if use_cache:
            if kv_cache is not None:
                k = torch.cat([kv_cache[0], k], dim=2)
                v = torch.cat([kv_cache[1], v], dim=2)
            new_cache = (k, v)

        # ── SDPA: single fused kernel — no manual mask, no manual matmuls ──
        # is_causal=True is needed during training AND during prefill (first
        # inference pass where kv_cache is None and the full prompt is fed).
        # is_causal=False only during decode steps where kv_cache is already
        # populated — the cached K/V are already causally ordered.
        is_causal = not use_cache or kv_cache is None
        dropout_p = self.dropout if self.training else 0.0

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out), new_cache


# ─── 4. FeedForward (SwiGLU) ──────────────────────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1      = nn.Linear(dim, ff_dim, bias=False)
        self.w2      = nn.Linear(ff_dim, dim, bias=False)
        self.w3      = nn.Linear(dim, ff_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


# ─── 5. Transformer Block ─────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn  = MultiHeadAttention(dim, n_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn   = FeedForward(dim, ff_dim, dropout)

    def forward(self, x, use_cache=False, kv_cache=None):
        attn_out, new_cache = self.attn(self.norm1(x), use_cache=use_cache, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_cache