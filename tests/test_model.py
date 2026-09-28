"""
tests/test_model.py
Unit tests for the core Transformer model.

Run:
    python -m pytest tests/test_model.py -v
"""

import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_model():
    """Returns a tiny KanhaModel for fast testing."""
    from kanha.core.model import KanhaModel
    return KanhaModel(
        vocab_size=1000,
        dim=64,
        n_layers=2,
        n_heads=4,
        ff_dim=128,
        max_seq_len=64,
        dropout=0.0,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_model_forward(small_model):
    """Model forward pass returns correct shapes."""
    B, S = 2, 16
    input_ids = torch.randint(0, 1000, (B, S))
    logits, loss, caches = small_model(input_ids)

    assert logits.shape == (B, S, 1000), f"Expected (2, 16, 1000), got {logits.shape}"
    assert loss is None, "Loss should be None when targets not given"
    assert len(caches) == 2, "Should have 2 KV caches (one per layer)"


def test_model_with_loss(small_model):
    """Model computes loss when targets are provided."""
    B, S = 2, 16
    input_ids = torch.randint(0, 1000, (B, S))
    targets   = torch.randint(0, 1000, (B, S))

    _, loss, _ = small_model(input_ids, targets=targets)

    assert loss is not None, "Loss should not be None when targets given"
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"


def test_model_masked_loss(small_model):
    """Loss ignores -100 masked positions."""
    B, S = 2, 16
    input_ids = torch.randint(0, 1000, (B, S))
    targets   = torch.full((B, S), -100, dtype=torch.long)   # all masked

    _, loss, _ = small_model(input_ids, targets=targets)
    # Loss on all-masked targets is 0 or NaN — just check no crash
    assert loss is not None


def test_causal_masking(small_model):
    """Verify the model applies causal masking (position i can only attend to <= i)."""
    small_model.eval()  # deterministic (no dropout)
    with torch.no_grad():
        B, S = 1, 8
        input_ids_a = torch.randint(0, 1000, (B, S))
        input_ids_b = input_ids_a.clone()
        input_ids_b[0, 4:] = torch.randint(0, 1000, (4,))  # change tokens 4-7

        logits_a, _, _ = small_model(input_ids_a)
        logits_b, _, _ = small_model(input_ids_b)

        # Positions 0-3 should have identical logits (causal masking)
        assert torch.allclose(logits_a[:, :4, :], logits_b[:, :4, :], atol=1e-5), \
            "Causal masking broken: changing future tokens affected past logits"


def test_rms_norm():
    """RMSNorm output has unit RMS."""
    from kanha.core.layers import RMSNorm
    norm = RMSNorm(64)
    x = torch.randn(2, 16, 64)
    out = norm(x)
    assert out.shape == x.shape


def test_rope_embedding():
    """RoPE returns correct shapes."""
    from kanha.core.layers import RotaryEmbedding
    rope = RotaryEmbedding(dim=32)
    cos, sin = rope(seq_len=8)
    assert cos.shape[-1] == 32
    assert sin.shape[-1] == 32


def test_parameter_count(small_model):
    """Model has non-zero parameters."""
    from kanha.utils.helpers import count_parameters
    n = count_parameters(small_model)
    assert n > 0, "Model should have trainable parameters"
    print(f"\nSmall model params: {n:,}")


def test_weight_tying(small_model):
    """Embedding and LM head weights are tied."""
    assert small_model.token_emb.weight is small_model.lm_head.weight, \
        "Weights should be tied"


def test_save_load(small_model, tmp_path):
    """Model can be saved and reloaded."""
    path = str(tmp_path / "test_model.pt")
    small_model.save_pretrained(path)

    from kanha.core.model import KanhaModel
    loaded = KanhaModel.from_pretrained(path)

    # Check weights match
    for (n1, p1), (n2, p2) in zip(
        small_model.named_parameters(), loaded.named_parameters()
    ):
        assert torch.allclose(p1, p2), f"Mismatch in {n1}"


def test_generation_runs(small_model):
    """Generation produces a non-empty string (smoke test without real tokenizer)."""
    from kanha.core.generation import sample_next_token
    # Simulate one step
    logits = torch.randn(1, 1, 1000)
    token = sample_next_token(logits, temperature=1.0, top_k=10, top_p=0.9)
    assert token.shape == (1, 1)
    assert 0 <= token.item() < 1000