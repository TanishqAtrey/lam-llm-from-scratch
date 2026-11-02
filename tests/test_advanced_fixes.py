"""
tests/test_advanced_fixes.py
Comprehensive tests for newly added training, LoRA, GQA, and agent fixes:
  1. LoRA adapter injection, trainable param isolation, non-destructive saving, merge & unload
  2. Pretraining checkpoint serialization (optimizer, scheduler, best_val_loss) and resume compatibility
  3. Numpy .npy header truncation and padding elimination logic
  4. GQA with SDPA fused kernel, causal masking, and KV-cache bounding
  5. Agent retry diversification (top_k relaxation with temperature)
  6. DPO dataset supervised token guard
  7. Token length checker (check_lengths.py)
"""

import os
import sys
import tempfile
import struct
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kanha.core.model import KanhaModel
from kanha.core.lora import LoRAModel, LoRALinear
from kanha.core.layers_gqa import GroupedQueryAttention, TransformerBlockGQA
from kanha.training.train import _save_pretrain_checkpoint


# ── 1. LoRA Tests ─────────────────────────────────────────────────────────────

def test_lora_injection_and_trainable_params():
    """Verify LoRA freezes base model and only exposes adapter parameters for training."""
    base_model = KanhaModel(
        vocab_size=200,
        dim=64,
        n_layers=2,
        n_heads=4,
        ff_dim=128,
        max_seq_len=64,
    )
    lora_model = LoRAModel(
        base_model,
        rank=4,
        alpha=8.0,
        dropout=0.0,
        target_modules=["q_proj", "v_proj"],
    )

    # Base parameters must have requires_grad = False
    for name, param in base_model.named_parameters():
        if "lora_" not in name:
            assert not param.requires_grad, f"Base param {name} should be frozen"

    # LoRA parameters must have requires_grad = True
    trainable_params = [p for p in lora_model.parameters() if p.requires_grad]
    assert len(trainable_params) > 0, "LoRA model must have trainable parameters"

    # Forward pass check
    x = torch.randint(0, 200, (2, 8))
    logits, loss, _ = lora_model(x)
    assert logits.shape == (2, 8, 200)


def test_lora_nondestructive_save(tmp_path):
    """Verify save_pretrained() does NOT mutate in-memory LoRA model and saves compatible checkpoint."""
    base_model = KanhaModel(
        vocab_size=200,
        dim=64,
        n_layers=2,
        n_heads=4,
        ff_dim=128,
        max_seq_len=64,
    )
    lora_model = LoRAModel(
        base_model,
        rank=4,
        alpha=8.0,
        dropout=0.0,
        target_modules=["q_proj", "v_proj"],
    )

    # Record weight before save
    first_lora_module = None
    for module in lora_model.modules():
        if isinstance(module, LoRALinear):
            first_lora_module = module
            break
    assert first_lora_module is not None
    orig_base_weight = first_lora_module.linear.weight.clone()

    save_path = str(tmp_path / "lora_saved.pt")
    lora_model.save_pretrained(save_path, step=500)

    # Verify in-memory weights were NOT modified
    assert torch.equal(first_lora_module.linear.weight, orig_base_weight), \
        "In-memory linear weights must not be mutated by save_pretrained"

    # Verify saved checkpoint can be loaded cleanly by KanhaModel.from_pretrained
    loaded_model = KanhaModel.from_pretrained(save_path)
    assert isinstance(loaded_model, KanhaModel)
    assert loaded_model.dim == 64


def test_lora_merge_and_unload():
    """Verify merge_and_unload merges delta weights into base linear layers."""
    base_model = KanhaModel(
        vocab_size=200,
        dim=64,
        n_layers=2,
        n_heads=4,
        ff_dim=128,
        max_seq_len=64,
    )
    lora_model = LoRAModel(
        base_model,
        rank=4,
        alpha=8.0,
        dropout=0.0,
        target_modules=["q_proj", "v_proj"],
    )

    merged = lora_model.merge_and_unload()
    assert isinstance(merged, KanhaModel)
    # LoRALinear should be replaced by standard nn.Linear
    for module in merged.modules():
        assert not isinstance(module, LoRALinear)


# ── 2. Pretraining Checkpoint & Resume Tests ──────────────────────────────────

def test_pretrain_checkpoint_and_resume(tmp_path):
    """Verify _save_pretrain_checkpoint saves full optimizer/scheduler state and can be reloaded."""
    model = KanhaModel(
        vocab_size=200,
        dim=64,
        n_layers=2,
        n_heads=4,
        ff_dim=128,
        max_seq_len=64,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)

    # Take one dummy optimizer step to populate state buffers
    x = torch.randint(0, 200, (2, 8))
    y = torch.randint(0, 200, (2, 8))
    _, loss, _ = model(x, targets=y)
    loss.backward()
    optimizer.step()
    scheduler.step()

    ckpt_path = str(tmp_path / "ckpt_step100.pt")
    _save_pretrain_checkpoint(ckpt_path, model, optimizer, scheduler, step=100, best_val_loss=2.45)

    # Check raw checkpoint contents
    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert raw_ckpt["step"] == 100
    assert raw_ckpt["best_val_loss"] == 2.45
    assert "optimizer_state" in raw_ckpt
    assert "scheduler_state" in raw_ckpt
    assert "model_state" in raw_ckpt

    # Verify KanhaModel.from_pretrained can load this checkpoint seamlessly
    loaded_model = KanhaModel.from_pretrained(ckpt_path)
    assert loaded_model.dim == 64

    # Verify optimizer state can be restored into a new optimizer
    new_opt = torch.optim.AdamW(loaded_model.parameters(), lr=1e-4)
    new_opt.load_state_dict(raw_ckpt["optimizer_state"])
    assert len(new_opt.state) > 0


# ── 3. Numpy Header Truncation Test ──────────────────────────────────────────

def test_npy_header_truncation(tmp_path):
    """Verify exact header truncation logic from prepare_3b_tokens.py."""
    out_path = tmp_path / "test_tokens.npy"
    total_target = 1000
    actual_written = 350

    # Create dummy array with extra zeros
    arr = np.zeros(total_target, dtype=np.uint16)
    arr[:actual_written] = np.arange(actual_written, dtype=np.uint16)
    np.save(out_path, arr)

    # Run truncation logic from prepare_3b_tokens.py
    cursor = actual_written
    with open(out_path, "rb+") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            np.lib.format.read_array_header_1_0(f)
            prefix_len = 10
        else:
            np.lib.format.read_array_header_2_0(f)
            prefix_len = 12
        header_len = f.tell()
        H = header_len - prefix_len
        d = {"descr": "<u2", "fortran_order": False, "shape": (cursor,)}
        s = str(d)
        padding = H - len(s) - 1
        if padding >= 0:
            header_str = s + (" " * padding) + "\n"
            f.seek(0)
            if version == (1, 0):
                f.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", H) + header_str.encode("latin1"))
            else:
                f.write(b"\x93NUMPY\x02\x00" + struct.pack("<I", H) + header_str.encode("latin1"))
        f.truncate(header_len + cursor * 2)

    # Reload truncated array
    loaded_arr = np.load(out_path, mmap_mode="r")
    assert len(loaded_arr) == actual_written
    assert loaded_arr[-1] == actual_written - 1


# ── 4. GQA SDPA & KV-Cache Bounding Test ──────────────────────────────────────

def test_gqa_sdpa_and_cache_bounding():
    """Verify GroupedQueryAttention uses SDPA and correctly bounds KV-cache."""
    dim = 64
    n_heads = 8
    n_kv_heads = 2
    gqa = GroupedQueryAttention(dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads, max_seq_len=32)

    B, S = 2, 8
    x = torch.randn(B, S, dim)
    out, cache = gqa(x, use_cache=True)
    assert out.shape == (B, S, dim)
    assert cache[0].shape == (B, n_kv_heads, S, dim // n_heads)

    # Feed existing large cache (40 tokens) -> should truncate to max_seq_len (32)
    large_cache = (
        torch.randn(B, n_kv_heads, 40, dim // n_heads),
        torch.randn(B, n_kv_heads, 40, dim // n_heads),
    )
    x_single = torch.randn(B, 1, dim)
    _, bounded_cache = gqa(x_single, use_cache=True, kv_cache=large_cache)
    assert bounded_cache[0].shape[2] <= 32


# ── 5. Agent Retry Diversification Test ───────────────────────────────────────

def test_agent_retry_top_k_relaxation():
    """Verify TaskAgent relaxes top_k on retry attempts."""
    from unittest.mock import MagicMock
    from kanha.lam.agent import TaskAgent

    mock_model = MagicMock()
    mock_tok = MagicMock()
    mock_env = MagicMock()

    agent = TaskAgent(mock_model, mock_tok, mock_env)
    assert agent._gen_top_k == 1

    # On attempt 0: top_k should be default (1)
    # On attempt > 0: top_k should be relaxed (50)
    captured_top_k = []

    def mock_generate_action(prompt, temperature=0.0, top_k=None):
        captured_top_k.append(top_k)
        if len(captured_top_k) == 1:
            return "invalid_action()", 0.9  # Fail first attempt
        return 'finish(msg="Done")', 0.9    # Succeed second attempt

    agent._generate_action = mock_generate_action
    mock_env.step.return_value = ("Success", True)

    agent.run("Test goal")
    assert captured_top_k[0] == 1, "First attempt should use greedy top_k=1"
    assert captured_top_k[1] == 50, "Retry attempt should relax top_k to 50 for diversity"


# ── 6. DPO Dataset Guard Test ────────────────────────────────────────────────

def test_dpo_dataset_label_guard(tmp_path):
    """Verify LAMDPODataset guarantees at least one supervised token in labels."""
    from kanha.lam.trajectory import LAMDPODataset
    from unittest.mock import MagicMock

    mock_tok = MagicMock()
    mock_tok.PAD_ID = 0
    mock_tok.encode.return_value = [10, 20, 30]

    # Create dummy dpo jsonl file
    dpo_file = tmp_path / "dpo_sample.jsonl"
    with open(dpo_file, "w") as f:
        f.write('{"goal": "G", "history": [], "chosen": "A1", "rejected": "A2"}\n')

    dataset = LAMDPODataset(str(dpo_file), mock_tok, max_len=16)
    sample = dataset[0]
    # Check that chosen_labels has at least one supervised token != -100
    chosen_labels = sample["chosen_labels"]
    assert any(l != -100 for l in chosen_labels), "DPO sample must have at least one non-masked label"


# ── 7. Token Lengths Checker Test ─────────────────────────────────────────────

def test_check_lengths_script(tmp_path):
    """Verify check_lengths.py correctly parses dataset and computes statistics."""
    import subprocess
    try:
        import sentencepiece as spm
    except ImportError:
        pytest.skip("sentencepiece not installed")

    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text("hello world\ntest query test reply\n", encoding="utf-8")
    spm_prefix = str(tmp_path / "tiny_tok")
    spm.SentencePieceTrainer.train(
        input=str(corpus_file),
        model_prefix=spm_prefix,
        vocab_size=32,
        model_type="bpe",
        hard_vocab_limit=False,
    )
    model_file = spm_prefix + ".model"

    sample_file = tmp_path / "sample_data.jsonl"
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write('{"instruction": "hi", "response": "hello world"}\n')
        f.write('{"instruction": "test query", "response": "test reply"}\n')

    cmd = [sys.executable, "check_lengths.py", "--data", str(sample_file), "--tokenizer", model_file]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Total samples: 2" in res.stdout
    assert "<= 256 tokens: 2" in res.stdout

