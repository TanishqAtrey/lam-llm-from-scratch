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
        orig_dtype = x.dtype
        x_fp32 = x.float()
        normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight).to(orig_dtype)


# ─── 2. Rotary Positional Embedding (RoPE) ────────────────────────────────────
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        self.max_seq_len = max(seq_len, getattr(self, "max_seq_len", seq_len))
        t     = torch.arange(self.max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(self.inv_freq.dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(self.inv_freq.dtype), persistent=False)

    def forward(self, seq_len: int, offset: int = 0):
        if offset + seq_len > self.max_seq_len:
            self._build_cache(max(offset + seq_len, self.max_seq_len * 2))
        return (
            self.cos_cached[offset:offset + seq_len].unsqueeze(0).unsqueeze(0),
            self.sin_cached[offset:offset + seq_len].unsqueeze(0).unsqueeze(0),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out.to(q.dtype), k_out.to(k.dtype)


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
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0, max_seq_len: int = 4096):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads     = n_heads
        self.head_dim    = dim // n_heads
        self.dropout     = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len

        self.q_proj   = nn.Linear(dim, dim, bias=False)
        self.k_proj   = nn.Linear(dim, dim, bias=False)
        self.v_proj   = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.rope     = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        offset: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # RoPE — offset by cumulative position or cached context length during inference
        actual_offset = offset if offset is not None else (kv_cache[0].shape[2] if kv_cache is not None else 0)
        cos, sin = self.rope(S, offset=actual_offset)
        q, k = apply_rope(q, k, cos, sin)

        # KV cache append (inference only — skipped entirely during training)
        new_cache = None
        if use_cache:
            if kv_cache is not None:
                k = torch.cat([kv_cache[0], k], dim=2)
                v = torch.cat([kv_cache[1], v], dim=2)
            if k.shape[2] > self.max_seq_len:
                k = k[:, :, -self.max_seq_len:, :]
                v = v[:, :, -self.max_seq_len:, :]
            new_cache = (k, v)

        # ── SDPA: single fused kernel — no manual mask, no manual matmuls ──
        # is_causal=True when S > 1 and no past KV cache is present (training or prefill).
        # When decoding single token with past KV cache, no mask is needed (is_causal=False).
        # When evaluating a chunk (S > 1) with existing KV cache, an explicit lower-right
        # aligned mask is needed to prevent PyTorch's default upper-left masking of the cache.
        attn_mask = None
        if kv_cache is None:
            is_causal = S > 1
        else:
            is_causal = False
            if S > 1:
                row_idx = torch.arange(S, device=x.device).unsqueeze(1)
                col_idx = torch.arange(k.shape[2], device=x.device).unsqueeze(0)
                past_len = k.shape[2] - S
                attn_mask = col_idx <= (row_idx + past_len)

        # MPS backend may fail silently with dropout_p > 0 in SDPA
        dp = self.dropout.p if isinstance(self.dropout, nn.Dropout) else float(self.dropout)
        dropout_p = dp if self.training and x.device.type != 'mps' else 0.0

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
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
    def __init__(self, dim: int, n_heads: int, ff_dim: int, dropout: float = 0.0, max_seq_len: int = 4096):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn  = MultiHeadAttention(dim, n_heads, dropout, max_seq_len=max_seq_len)
        self.norm2 = RMSNorm(dim)
        self.ffn   = FeedForward(dim, ff_dim, dropout)

    def forward(self, x, use_cache=False, kv_cache=None, offset: Optional[int] = None):
        attn_out, new_cache = self.attn(self.norm1(x), use_cache=use_cache, kv_cache=kv_cache, offset=offset)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_cache