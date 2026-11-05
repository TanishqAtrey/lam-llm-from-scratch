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

DEFAULT_STOP_SEQUENCES = [
    "\n### Instruction:",
    "### Instruction:",
    "### Conversation History:",
    "\nUser:",
    "\nHuman:",
]


def _has_sentence_repetition(text: str) -> bool:
    """
    Detects if the generated text is trapped in a repetitive sentence loop.
    Returns True if:
      - The same sentence (>15 chars) is repeated 2 times consecutively, OR
      - Any sentence (>15 chars) appears 3 or more times.
    """
    import re
    if len(text) < 30:
        return False
    sentences = [s.strip().lower() for s in re.split(r'[\.\n\?!]+', text) if len(s.strip()) > 15]
    if len(sentences) >= 2 and sentences[-1] == sentences[-2]:
        return True
    if len(sentences) >= 3 and sentences.count(sentences[-1]) >= 3:
        return True
    return False


def _clean_repetitive_tail(text: str) -> str:
    """If the trailing sentence is identical to the preceding sentence, trim the duplicate."""
    import re
    sentences = re.split(r'(?<=[.!?\n])\s+', text)
    if len(sentences) >= 2:
        last = sentences[-1].strip().lower()
        prev = sentences[-2].strip().lower()
        if len(last) > 15 and last == prev:
            return " ".join(sentences[:-1]).strip()
    return text


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zeros out all logits except the top-k."""
    if k <= 0 or k >= logits.shape[-1]:
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
    logits = logits.clone()
    if logits.ndim > 1:
        if logits.shape[0] == 1:
            unique_tokens = torch.unique(input_ids).unsqueeze(0)
            score = torch.gather(logits, 1, unique_tokens)
            score = torch.where(score < 0, score * penalty, score / penalty)
            logits.scatter_(1, unique_tokens, score)
        else:
            for b in range(logits.shape[0]):
                b_unique = torch.unique(input_ids[b]).unsqueeze(0)
                b_score = torch.gather(logits[b:b+1], 1, b_unique)
                b_score = torch.where(b_score < 0, b_score * penalty, b_score / penalty)
                logits[b:b+1].scatter_(1, b_unique, b_score)
    else:
        unique_tokens = torch.unique(input_ids)
        score = torch.gather(logits, 0, unique_tokens)
        score = torch.where(score < 0, score * penalty, score / penalty)
        logits.scatter_(0, unique_tokens, score)
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


def _enforce_context_budget(input_ids: list[int], max_context: int, max_new_tokens: int) -> list[int]:
    """
    Enforces that prompt length does not overflow the model's trained context window.
    If max_new_tokens < max_context, reserves max_new_tokens for generation.
    If prompt exceeds available budget, truncates while preserving BOS and critical
    header structure (e.g. ### Instruction / ### Goal) instead of dropping the task statement.
    """
    if max_new_tokens < max_context:
        max_prompt_len = max(1, max_context - max_new_tokens)
    else:
        max_prompt_len = max(1, max_context - 1)

    if len(input_ids) > max_prompt_len:
        log.warning(
            f"Prompt length ({len(input_ids)}) exceeds prompt budget ({max_prompt_len} tokens for "
            f"max_seq_len={max_context}, max_new_tokens={max_new_tokens}). Truncating prompt."
        )
        if max_prompt_len == 1:
            return [input_ids[0]]
        # Preserve BOS and task/goal header (first ~25% of budget) plus the latest context
        head_reserve = max(2, min(max_prompt_len // 4, 64)) if max_prompt_len >= 4 else 1
        tail_reserve = max_prompt_len - head_reserve
        if head_reserve > 1:
            return input_ids[:head_reserve] + input_ids[-tail_reserve:]
        return [input_ids[0]] + input_ids[-(max_prompt_len - 1):]
    return input_ids


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
    stop_sequences: Optional[List[str]] = None,
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
        stop_sequences    : multi-token string sequences that stop generation early
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

    effective_stop_sequences = list(stop_sequences) if stop_sequences is not None else list(DEFAULT_STOP_SEQUENCES)

    model.eval()
    model.to(device)

    # Encode prompt (BOS added, EOS not — we're continuing from here)
    input_ids    = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    input_ids    = _enforce_context_budget(input_ids, getattr(model, "max_seq_len", 1024), max_new_tokens)
    prompt_len   = len(input_ids)          # remember where prompt ends
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated = list(input_ids)
    # BUG FIX: initialise as a list of None (one slot per layer), NOT plain None.
    # plain None → model.forward sees kv_caches is None → use_cache=False on
    # the very first pass → prompt K/V are never stored → from token 2 onwards
    # the model has zero context and generates garbage.
    kv_caches = [None] * model.n_layers
    cur_pos = 0

    for _ in range(max_new_tokens):
        logits, _, kv_caches = model(input_tensor, kv_caches=kv_caches, offset=cur_pos)
        cur_pos += input_tensor.shape[1]

        # Repetition penalty: penalize both new tokens and recent context history
        if repetition_penalty != 1.0 and len(generated) > 1:
            penalty_start = max(0, prompt_len - 128)
            ctx = torch.tensor([generated[penalty_start:]], dtype=torch.long, device=device)
            logits[:, -1, :] = apply_repetition_penalty(
                logits[:, -1, :], ctx, repetition_penalty,
            )

        next_token = sample_next_token(logits, temperature, top_k, top_p)
        next_id    = next_token.item()
        generated.append(next_id)

        if next_id in stop_tokens:
            break

        if effective_stop_sequences or len(generated) > prompt_len + 10:
            current_text = tokenizer.decode(generated[prompt_len:], skip_special=True)
            if effective_stop_sequences and any(seq in current_text for seq in effective_stop_sequences):
                break
            if _has_sentence_repetition(current_text):
                break

        # Feed only the new token — KV cache handles the rest
        input_tensor = next_token

    # Decode ONLY the new tokens — never the prompt
    # This is the correct fix vs. fragile startswith() string stripping
    new_ids = generated[prompt_len:]
    text = tokenizer.decode(new_ids, skip_special=True)
    if effective_stop_sequences:
        matched_indices = [text.find(seq) for seq in effective_stop_sequences if seq in text]
        if matched_indices:
            text = text[:min(matched_indices)]
    text = _clean_repetitive_tail(text)
    return text


# ─── Generation with confidence scoring ───────────────────────────────────────

@torch.no_grad()
def generate_with_confidence(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int       = None,
    temperature: float        = None,
    top_k: int                = None,
    top_p: float              = None,
    repetition_penalty: float = None,
    stop_tokens: Optional[List[int]] = None,
    stop_sequences: Optional[List[str]] = None,
    confidence_metric: str    = "geometric",
    device: torch.device      = None,
) -> tuple[str, float, list[float]]:
    """
    Generates a response and returns (text, mean_confidence, per_token_confidences).

    Confidence = mean of softmax probabilities assigned to each selected token.
    Values near 1.0 = model is very sure. Values near 0.0 = model is guessing.

    This enables downstream safety decisions:
      - confidence < 0.3  → ask user for clarification
      - confidence < 0.5  → flag as low confidence
      - confidence >= 0.5 → proceed normally

    Returns:
        text             : str   — the generated continuation (no prompt)
        mean_confidence  : float — average token-level probability (0 to 1)
        per_token_confs  : list[float] — per-token softmax probabilities
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

    effective_stop_sequences = list(stop_sequences) if stop_sequences is not None else list(DEFAULT_STOP_SEQUENCES)

    model.eval()
    model.to(device)

    input_ids    = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    input_ids    = _enforce_context_budget(input_ids, getattr(model, "max_seq_len", 1024), max_new_tokens)
    prompt_len   = len(input_ids)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated = list(input_ids)
    kv_caches = [None] * model.n_layers
    per_token_confs: list[float] = []
    cur_pos = 0

    for _ in range(max_new_tokens):
        logits, _, kv_caches = model(input_tensor, kv_caches=kv_caches, offset=cur_pos)
        cur_pos += input_tensor.shape[1]

        # Compute true model probabilities before heuristic distortion
        raw_last_logits = logits[:, -1, :].clone()
        raw_probs = F.softmax(raw_last_logits, dim=-1)

        # Repetition penalty: penalize both new tokens and recent context history
        if repetition_penalty != 1.0 and len(generated) > 1:
            penalty_start = max(0, prompt_len - 128)
            ctx = torch.tensor([generated[penalty_start:]], dtype=torch.long, device=device)
            logits[:, -1, :] = apply_repetition_penalty(
                logits[:, -1, :], ctx, repetition_penalty,
            )

        next_token = sample_next_token(logits, temperature, top_k, top_p)
        next_id    = next_token.item()

        # Record confidence: true probability assigned to the chosen token
        token_confidence = raw_probs[0, next_id].item()
        per_token_confs.append(token_confidence)

        generated.append(next_id)

        if next_id in stop_tokens:
            break

        if effective_stop_sequences or len(generated) > prompt_len + 10:
            current_text = tokenizer.decode(generated[prompt_len:], skip_special=True)
            if effective_stop_sequences and any(seq in current_text for seq in effective_stop_sequences):
                break
            if _has_sentence_repetition(current_text):
                break

        input_tensor = next_token

    # Decode only new tokens
    new_ids = generated[prompt_len:]
    text = tokenizer.decode(new_ids, skip_special=True)
    if effective_stop_sequences:
        matched_indices = [text.find(seq) for seq in effective_stop_sequences if seq in text]
        if matched_indices:
            text = text[:min(matched_indices)]
    text = _clean_repetitive_tail(text)

    # Exclude stop tokens and stop sequences from confidence calculations
    # so confidence accurately reflects only the generated content tokens
    if next_id in stop_tokens and len(per_token_confs) > 1:
        per_token_confs.pop()
    elif effective_stop_sequences and len(per_token_confs) > 1:
        full_decoded = tokenizer.decode(new_ids, skip_special=True)
        if len(text) < len(full_decoded):
            trimmed_ids = tokenizer.encode(text, add_bos=False, add_eos=False)
            retained_len = min(len(trimmed_ids), len(per_token_confs))
            if retained_len > 0:
                per_token_confs = per_token_confs[:retained_len]
            else:
                per_token_confs.pop()

    import math
    mean_confidence = sum(per_token_confs) / max(len(per_token_confs), 1)
    min_confidence = min(per_token_confs) if per_token_confs else 1.0
    geom_confidence = math.exp(
        sum(math.log(max(p, 1e-12)) for p in per_token_confs) / max(len(per_token_confs), 1)
    ) if per_token_confs else 1.0

    if confidence_metric == "arithmetic":
        chosen_conf = mean_confidence
    elif confidence_metric == "min":
        chosen_conf = min_confidence
    else:
        # "geometric" default: severely penalizes uncertain slot/argument tokens
        chosen_conf = geom_confidence

    return text, chosen_conf, per_token_confs


def _find_stop_prefix_len(text: str, stop_sequences: list[str]) -> int:
    """Finds the length of the longest suffix of `text` that is a non-empty prefix of any stop sequence."""
    if not text or not stop_sequences:
        return 0
    max_prefix_len = 0
    for seq in stop_sequences:
        max_check = min(len(text), len(seq))
        for k in range(max_check, 0, -1):
            if text.endswith(seq[:k]):
                if k > max_prefix_len:
                    max_prefix_len = k
                break
    return max_prefix_len


# ─── Streaming variant ────────────────────────────────────────────────────────

@torch.no_grad()
def generate_stream(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int       = None,
    temperature: float        = None,
    top_k: int                = None,
    top_p: float              = None,
    repetition_penalty: float = None,
    stop_tokens: Optional[List[int]] = None,
    stop_sequences: Optional[List[str]] = None,
    device: torch.device      = None,
):
    """
    Generator that yields one decoded token string at a time.

    FIX: Uses incremental decoding instead of single-token decoding.
    SentencePiece strips the leading space ( ) when decoding a single
    token in isolation, causing "HelloWorld" instead of "Hello World".
    By decoding all generated tokens and yielding the diff, spaces
    are preserved correctly. Buffers potential prefixes of stop_sequences
    to prevent leaking partial markers into the stream.

    Usage:
        for token_text in generate_stream(model, tok, "Hello"):
            print(token_text, end="", flush=True)
    """
    from kanha.utils.config import cfg

    ic                 = cfg.inference
    max_new_tokens     = max_new_tokens     if max_new_tokens     is not None else ic.max_new_tokens
    temperature        = temperature        if temperature        is not None else ic.temperature
    top_k              = top_k              if top_k              is not None else ic.top_k
    top_p              = top_p              if top_p              is not None else ic.top_p
    repetition_penalty = repetition_penalty if repetition_penalty is not None else ic.repetition_penalty
    device             = device or get_device()

    if stop_tokens is None:
        stop_tokens = [tokenizer.EOS_ID]

    effective_stop_sequences = list(stop_sequences) if stop_sequences is not None else list(DEFAULT_STOP_SEQUENCES)

    model.eval()
    model.to(device)

    input_ids    = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    input_ids    = _enforce_context_budget(input_ids, getattr(model, "max_seq_len", 1024), max_new_tokens)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    kv_caches = [None] * model.n_layers
    generated = list(input_ids)
    cur_pos = 0

    # For incremental decode: track all NEW token ids and previous decoded text
    new_ids = []
    prev_text = ""
    stopped_by_seq = False

    for _ in range(max_new_tokens):
        logits, _, kv_caches = model(input_tensor, kv_caches=kv_caches, offset=cur_pos)
        cur_pos += input_tensor.shape[1]

        # Repetition penalty: penalize both new tokens and recent context history
        if repetition_penalty != 1.0 and len(generated) > 1:
            penalty_start = max(0, len(input_ids) - 128)
            ctx = torch.tensor([generated[penalty_start:]], dtype=torch.long, device=device)
            logits[:, -1, :] = apply_repetition_penalty(
                logits[:, -1, :], ctx, repetition_penalty,
            )

        next_token = sample_next_token(logits, temperature, top_k, top_p)
        next_id    = next_token.item()
        generated.append(next_id)

        if next_id in stop_tokens:
            break

        new_ids.append(next_id)
        full_text = tokenizer.decode(new_ids, skip_special=True)

        if not full_text.endswith("\ufffd"):
            # Check for complete stop sequence match
            if effective_stop_sequences and any(seq in full_text for seq in effective_stop_sequences):
                earliest_idx = min(full_text.find(seq) for seq in effective_stop_sequences if seq in full_text)
                commit_text = full_text[:earliest_idx]
                if len(commit_text) > len(prev_text):
                    diff = commit_text[len(prev_text):]
                    prev_text = commit_text
                    if diff:
                        yield diff
                stopped_by_seq = True
                break

            # Check for runaway sentence loops
            if _has_sentence_repetition(full_text):
                stopped_by_seq = True
                break

            # Buffer partial prefixes of any stop sequences
            prefix_len = _find_stop_prefix_len(full_text, effective_stop_sequences) if effective_stop_sequences else 0
            safe_len = len(full_text) - prefix_len
            if safe_len > len(prev_text):
                diff = full_text[len(prev_text):safe_len]
                prev_text = full_text[:safe_len]
                if diff:
                    yield diff

        input_tensor = next_token

    if new_ids and not stopped_by_seq:
        final_full = tokenizer.decode(new_ids, skip_special=True)
        if effective_stop_sequences and any(seq in final_full for seq in effective_stop_sequences):
            earliest_idx = min(final_full.find(seq) for seq in effective_stop_sequences if seq in final_full)
            final_full = final_full[:earliest_idx]
        elif effective_stop_sequences:
            prefix_len = _find_stop_prefix_len(final_full, effective_stop_sequences)
            if prefix_len > 0:
                final_full = final_full[:-prefix_len]
        final_full = _clean_repetitive_tail(final_full)
        final_diff = final_full[len(prev_text):]
        if final_diff and "\ufffd" not in final_diff:
            yield final_diff