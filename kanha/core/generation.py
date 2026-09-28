"""
kanha/core/generation.py
Text generation engine — runs the model autoregressively.

Supports:
  - Greedy decoding
  - Top-k sampling
  - Top-p (nucleus) sampling
  - Temperature scaling
  - Repetition penalty
  - KV cache for fast inference
"""

import torch
import torch.nn.functional as F
from typing import List, Optional
from kanha.utils.helpers import get_device
from kanha.utils.logging import get_logger

log = get_logger(__name__)


# ─── Sampling helpers ─────────────────────────────────────────────────────────

def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zeros out all logits except the top-k."""
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    threshold = values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling: keeps the smallest set of tokens whose
    cumulative probability exceeds p.
    """
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > p
    sorted_logits[sorted_indices_to_remove] = float("-inf")
    logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)
    return logits


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Penalizes tokens that already appear in the context."""
    if penalty == 1.0:
        return logits
    score = torch.gather(logits, 1, input_ids)
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(1, input_ids, score)
    return logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> torch.Tensor:
    """
    Samples one token from logits.
    logits: (1, S, vocab_size) — uses last position only.
    Returns: (1, 1) tensor.
    """
    logits = logits[:, -1, :]            # last position → (1, vocab)

    if temperature == 0.0:               # greedy
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    probs  = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ─── Main generation function ─────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int       = None,
    temperature: float        = None,
    top_k: int                = None,
    top_p: float              = None,
    repetition_penalty: float = None,
    stop_tokens: Optional[List[int]] = None,
    device: torch.device      = None,
) -> str:
    """
    Generates a response to a prompt.

    Returns ONLY the newly generated text (not the prompt).
    Fix: decodes only new token ids directly — avoids the fragile
    tokenize→decode→startswith stripping that breaks after SentencePiece
    round-trip.

    Args:
        model             : KanhaModel in eval mode
        tokenizer         : KanhaTokenizer
        prompt            : input string
        max_new_tokens    : max tokens to generate
        temperature       : sampling temperature (0 = greedy)
        top_k             : top-k filter (0 = disabled)
        top_p             : nucleus filter
        repetition_penalty: penalize repeated tokens
        stop_tokens       : token ids that stop generation early
        device            : torch device

    Returns:
        response : str — the generated continuation only (no prompt)
    """
    from kanha.utils.config import cfg

    ic                 = cfg.inference
    max_new_tokens     = max_new_tokens     or ic.max_new_tokens
    temperature        = temperature        if temperature        is not None else ic.temperature
    top_k              = top_k              if top_k              is not None else ic.top_k
    top_p              = top_p              if top_p              is not None else ic.top_p
    repetition_penalty = repetition_penalty if repetition_penalty is not None else ic.repetition_penalty
    device             = device             or get_device()

    if stop_tokens is None:
        stop_tokens = [tokenizer.EOS_ID]

    model.eval()
    model.to(device)

    # Encode prompt (BOS added, EOS not — we're continuing from here)
    input_ids    = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    prompt_len   = len(input_ids)          # remember where prompt ends
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated = list(input_ids)
    # BUG FIX: initialise as a list of None (one slot per layer), NOT plain None.
    # plain None → model.forward sees kv_caches is None → use_cache=False on
    # the very first pass → prompt K/V are never stored → from token 2 onwards
    # the model has zero context and generates garbage.
    kv_caches = [None] * model.n_layers

    for _ in range(max_new_tokens):
        logits, _, kv_caches = model(input_tensor, kv_caches=kv_caches)

        # Repetition penalty over full context
        if repetition_penalty != 1.0:
            ctx = torch.tensor([generated], dtype=torch.long, device=device)
            logits[:, -1, :] = apply_repetition_penalty(
                logits[:, -1, :], ctx, repetition_penalty,
            )

        next_token = sample_next_token(logits, temperature, top_k, top_p)
        next_id    = next_token.item()
        generated.append(next_id)

        if next_id in stop_tokens:
            break

        # Feed only the new token — KV cache handles the rest
        input_tensor = next_token

    # Decode ONLY the new tokens — never the prompt
    # This is the correct fix vs. fragile startswith() string stripping
    new_ids = generated[prompt_len:]
    return tokenizer.decode(new_ids, skip_special=True)


# ─── Streaming variant ────────────────────────────────────────────────────────

@torch.no_grad()
def generate_stream(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int  = 256,
    temperature: float   = 0.7,
    top_k: int           = 50,
    top_p: float         = 0.9,
    device: torch.device = None,
):
    """
    Generator that yields one decoded token string at a time.

    FIX: Uses incremental decoding instead of single-token decoding.
    SentencePiece strips the leading space (▁) when decoding a single
    token in isolation, causing "HelloWorld" instead of "Hello World".
    By decoding all generated tokens and yielding the diff, spaces
    are preserved correctly.

    Usage:
        for token_text in generate_stream(model, tok, "Hello"):
            print(token_text, end="", flush=True)
    """
    from kanha.utils.config import cfg

    device = device or get_device()
    model.eval()
    model.to(device)

    input_ids    = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    kv_caches = [None] * model.n_layers
    generated = list(input_ids)

    # For incremental decode: track all NEW token ids and previous decoded text
    new_ids = []
    prev_text = ""

    rep_penalty = cfg.inference.repetition_penalty

    for _ in range(max_new_tokens):
        logits, _, kv_caches = model(input_tensor, kv_caches=kv_caches)

        if rep_penalty != 1.0:
            ctx = torch.tensor([generated], dtype=torch.long, device=device)
            logits[:, -1, :] = apply_repetition_penalty(
                logits[:, -1, :], ctx, rep_penalty,
            )

        next_token = sample_next_token(logits, temperature, top_k, top_p)
        next_id    = next_token.item()
        generated.append(next_id)

        if next_id == tokenizer.EOS_ID:
            break

        # Incremental decode: decode ALL new tokens, yield only the new part
        # This preserves SentencePiece's ▁ → space conversion correctly
        new_ids.append(next_id)
        full_text = tokenizer.decode(new_ids, skip_special=True)
        diff = full_text[len(prev_text):]
        prev_text = full_text

        if diff:
            yield diff

        input_tensor = next_token