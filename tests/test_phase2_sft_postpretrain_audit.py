"""
tests/test_phase2_sft_postpretrain_audit.py
Unit tests verifying fixes for Phase 2 SFT and post-pretraining pipeline:
  1. Boundary-safe SFT tokenization when response contains delimiter
  2. SFT filtering of empty / whitespace-only responses
  3. PromptBuilder exact prefix consistency for raw and role-prefixed prompts
  4. DPODataset boundary-safe tokenization when response contains delimiter
  5. generate() order-independent stop sequence slicing (earliest prefix wins)
  6. generate_stream() trimming of trailing partial stop sequences
  7. diagnose_model checkpoint loader compatibility with PyTorch 2.6+
  8. check_lengths template overhead calculation
"""

import os
import sys
import json
import tempfile
import torch
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kanha.core.tokenizer import KanhaTokenizer
from kanha.prompting.builder import PromptBuilder
from kanha.finetune.sft_train import SFTDataset
from kanha.finetune.dpo_train import DPODataset
from kanha.core.generation import generate, generate_stream, _find_stop_prefix_len
from scripts.diagnose_model import check_checkpoint


def test_sft_dataset_response_containing_delimiter(tmp_path):
    """Verify SFTDataset does not truncate response when it contains ### Response: substring."""
    tok = KanhaTokenizer()
    data_file = tmp_path / "sft_with_delimiter.jsonl"
    instruction = "Write a sample showing instruction tuning format:"
    response = "Here is an example:\n### Instruction:\nHello\n### Response:\nWorld\nEnd of example."
    
    with open(data_file, "w") as f:
        f.write(json.dumps({"instruction": instruction, "response": response}) + "\n")

    dataset = SFTDataset(str(data_file), tok, max_len=256)
    assert len(dataset) == 1
    sample = dataset[0]

    input_ids = sample["input_ids"]
    labels = sample["labels"]
    
    # Get unmasked response token IDs
    resp_token_mask = (labels != -100)
    supervised_target_ids = labels[resp_token_mask].tolist()
    decoded_resp = tok.decode(supervised_target_ids, skip_special=True)
    
    # The entire response including the inner "### Response:" must be supervised
    assert "Here is an example:" in decoded_resp
    assert "### Instruction:" in decoded_resp
    assert "### Response:" in decoded_resp
    assert "End of example." in decoded_resp


def test_sft_dataset_rejects_empty_and_whitespace_responses(tmp_path):
    """Verify SFTDataset skips samples where response is empty or only whitespace."""
    tok = KanhaTokenizer()
    data_file = tmp_path / "sft_empty.jsonl"
    with open(data_file, "w") as f:
        f.write(json.dumps({"instruction": "Hello", "response": ""}) + "\n")
        f.write(json.dumps({"instruction": "Hello", "response": "   \n\t  "}) + "\n")
        f.write(json.dumps({"instruction": "Hello", "response": "Valid answer."}) + "\n")

    dataset = SFTDataset(str(data_file), tok, max_len=128)
    # Only the valid sample should be retained
    assert len(dataset) == 1
    sample = dataset[0]
    assert (sample["labels"] != -100).any()


def test_prompt_builder_exact_prefix_consistency():
    """Verify PromptBuilder.build() is an exact prefix of build_training_pair() for all inputs."""
    builder = PromptBuilder(include_rules=False)
    
    test_queries = [
        "What is Python?",
        "User: Explain gravity.",
        "Human: Tell me a joke.",
        "User: Hi\nAssistant: Hello!\nUser: What is 2+2?",
    ]
    
    for q in test_queries:
        resp = "Test response."
        prefix, _ = builder.build_prompt_prefix(q)
        infer_prompt = builder.build(instruction=q)
        train_pair = builder.build_training_pair(q, resp)
        
        assert prefix == infer_prompt, f"Prefix mismatch for query: {q}"
        assert train_pair.startswith(infer_prompt), f"Training pair does not start with infer prompt for query: {q}"
        assert train_pair[len(infer_prompt):] == resp


def test_dpo_dataset_response_containing_delimiter(tmp_path):
    """Verify DPODataset correctly handles chosen/rejected responses containing ### Response:."""
    tok = KanhaTokenizer()
    data_file = tmp_path / "dpo_delimiter.jsonl"
    prompt = "Show the response format"
    chosen = "Format:\n### Response:\n42"
    rejected = "Format:\n### Response:\n99"
    
    with open(data_file, "w") as f:
        f.write(json.dumps({"prompt": prompt, "chosen": chosen, "rejected": rejected}) + "\n")

    dataset = DPODataset(str(data_file), tok, max_len=128)
    assert len(dataset) == 1
    sample = dataset[0]

    c_labels = sample["chosen_labels"]
    c_unmasked = c_labels[c_labels != -100].tolist()
    c_decoded = tok.decode(c_unmasked, skip_special=True)
    assert "Format:" in c_decoded
    assert "### Response:" in c_decoded
    assert "42" in c_decoded


def test_generate_stop_sequences_order_independent():
    """Verify generate slices off at the earliest stop sequence regardless of list order."""
    mock_tok = MagicMock()
    mock_tok.EOS_ID = 99
    # Slices text after prompt
    mock_tok.encode.return_value = [1]
    
    # Model generates: "Hello world\n### Instruction: next question"
    mock_tok.decode.side_effect = lambda ids, **kw: "Hello world\n### Instruction: next question"

    mock_model = MagicMock()
    mock_model.n_layers = 1
    mock_model.max_seq_len = 100
    mock_model.eval = MagicMock()
    mock_model.to = MagicMock()
    mock_model.return_value = (torch.zeros(1, 1, 100), None, [None])

    # Case A: Shorter stop sequence first
    stop_seqs_A = ["### Instruction:", "\n### Instruction:"]
    with pytest.MonkeyPatch().context() as m:
        m.setattr("kanha.core.generation.get_device", lambda: "cpu")
        m.setattr("kanha.core.generation.sample_next_token", lambda *a, **kw: torch.tensor([[99]]))
        res_A = generate(mock_model, mock_tok, "prompt", max_new_tokens=2, stop_sequences=stop_seqs_A)

    # Case B: Longer stop sequence first
    stop_seqs_B = ["\n### Instruction:", "### Instruction:"]
    with pytest.MonkeyPatch().context() as m:
        m.setattr("kanha.core.generation.get_device", lambda: "cpu")
        m.setattr("kanha.core.generation.sample_next_token", lambda *a, **kw: torch.tensor([[99]]))
        res_B = generate(mock_model, mock_tok, "prompt", max_new_tokens=2, stop_sequences=stop_seqs_B)

    # Both must cleanly strip the leading newline before ### Instruction:
    assert res_A == "Hello world"
    assert res_B == "Hello world"


def test_generate_stream_trims_partial_stop_sequences():
    """Verify generate_stream does not leak trailing partial stop prefixes (e.g. '\n###') when token budget ends."""
    mock_tok = MagicMock()
    mock_tok.EOS_ID = 99
    mock_tok.encode.return_value = [1]
    
    # Simulate generating "Action" then "\n###"
    step = 0
    def mock_decode(ids, **kw):
        if len(ids) == 1:
            return "Action"
        elif len(ids) == 2:
            return "Action\n###"
        return "Action"

    mock_tok.decode.side_effect = mock_decode

    mock_model = MagicMock()
    mock_model.n_layers = 1
    mock_model.max_seq_len = 100
    mock_model.eval = MagicMock()
    mock_model.to = MagicMock()
    mock_model.return_value = (torch.zeros(1, 1, 100), None, [None])

    stop_seqs = ["\n### Observation", "### Observation"]

    with pytest.MonkeyPatch().context() as m:
        m.setattr("kanha.core.generation.get_device", lambda: "cpu")
        m.setattr("kanha.core.generation.sample_next_token", lambda *a, **kw: torch.tensor([[10]]))
        chunks = list(generate_stream(mock_model, mock_tok, "prompt", max_new_tokens=2, stop_sequences=stop_seqs))

    streamed = "".join(chunks)
    # The partial stop prefix "\n###" should NOT leak into streamed text
    assert "\n###" not in streamed
    assert streamed == "Action"


def test_diagnose_checkpoint_pytorch26_compatibility(tmp_path):
    """Verify check_checkpoint executes without error on PyTorch 2.6+."""
    dummy_model_path = tmp_path / "model.pt"
    ckpt_data = {
        "model_config": {"vocab_size": 100, "dim": 32, "n_layers": 1, "n_heads": 1, "ff_dim": 64, "max_seq_len": 32, "dropout": 0.0},
        "model_state": {"token_emb.weight": torch.randn(100, 32), "lm_head.weight": torch.randn(100, 32)}
    }
    torch.save(ckpt_data, dummy_model_path)
    
    # Should run and return True/dictionary without UnpicklerError
    res = check_checkpoint(str(dummy_model_path))
    assert res is not None


def test_sft_train_split_small_dataset_no_empty_train(tmp_path):
    """Verify SFT training handles 1-sample or tiny datasets without ValueError: num_samples=0."""
    from types import SimpleNamespace
    from kanha.finetune.sft_train import sft_train
    
    data_file = tmp_path / "tiny_sft.jsonl"
    with open(data_file, "w") as f:
        f.write(json.dumps({"instruction": "Hello", "response": "Hi there!"}) + "\n")

    out_dir = tmp_path / "out"
    args = SimpleNamespace(
        base_model="models/base/final_model.pt",
        data=str(data_file),
        output=str(out_dir),
        epochs=1,
        batch_size=1,
        grad_accum_steps=1,
        lr=3e-5,
        val_fraction=0.1,
        patience=1,
        lora=False,
    )
    # Must run and complete without raising ValueError: num_samples should be a positive integer value
    sft_train(args)
    assert os.path.exists(out_dir / "sft_final.pt")


def test_lam_sft_train_split_single_goal_no_empty_train(tmp_path):
    """Verify LAM SFT training handles single-goal datasets without empty train set."""
    from types import SimpleNamespace
    from kanha.finetune.lam_train import lam_sft_train

    data_file = tmp_path / "tiny_lam.jsonl"
    with open(data_file, "w") as f:
        f.write(json.dumps({
            "goal": "create task",
            "steps": [{"action": "create(title=\"t\")", "observation": "done"}]
        }) + "\n")

    out_dir = tmp_path / "lam_out"
    args = SimpleNamespace(
        base_model="models/base/final_model.pt",
        data=str(data_file),
        output=str(out_dir),
        epochs=1,
        batch_size=1,
        grad_accum_steps=1,
        lr=3e-5,
        val_fraction=0.5,
        patience=1,
        lora=False,
    )
    lam_sft_train(args)
    assert os.path.exists(out_dir / "lam_final.pt")

