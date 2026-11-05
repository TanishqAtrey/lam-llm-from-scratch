"""
tests/test_comprehensive_audit_fixes.py
Unit and regression tests verifying all 10 user findings and post-pretraining audit fixes.
"""

import os
import sys
import json
import tempfile
import torch
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kanha.prompting.builder import PromptBuilder, FULL_TEMPLATE
from kanha.core.generation import _enforce_context_budget, apply_repetition_penalty, generate
from kanha.core.model import KanhaModel
from kanha.core.tokenizer import KanhaTokenizer
from kanha.finetune.sft_train import SFTDataset
from kanha.finetune.dpo_train import DPODataset
from kanha.lam.trajectory import LAMDPODataset
from kanha.lam.data.generator import TrajectoryGenerator
from kanha.lam.data.oracle import Oracle
from kanha.lam.environment import TaskEnvironment


def test_single_turn_user_prefix_cleaning():
    """Verify single-turn instructions starting with User: or Human: are cleaned in training pair."""
    builder = PromptBuilder(include_rules=False)
    
    # Single turn with User: prefix
    pair = builder.build_training_pair("User: What is Python?", "A programming language.")
    assert "### Instruction:\nWhat is Python?" in pair
    assert "User: What is Python?" not in pair
    
    # Single turn with Human: prefix
    pair_h = builder.build_training_pair("Human: Explain gravity.", "Force of attraction.")
    assert "### Instruction:\nExplain gravity." in pair_h
    assert "Human: Explain gravity." not in pair_h


def test_dialogue_preamble_and_trailing_turn_preservation():
    """Verify non-role preamble is preserved as task, and trailing assistant turns are not dropped."""
    builder = PromptBuilder(include_rules=False)
    
    # Preamble before dialogue
    task_prompt = "Summarize the following chat:\nUser: Hi\nAssistant: Hello"
    pair = builder.build_training_pair(task_prompt, "They greeted each other.")
    # Because it starts with "Summarize", it shouldn't be butchered into empty instruction
    assert "Summarize the following chat:" in pair
    
    # Dialogue ending in assistant turn
    dialogue = "User: Can you code?\nAssistant: Yes.\nUser: In Python?\nAssistant: Yes, Python too."
    history, inst = builder.parse_dialogue(dialogue)
    assert inst == "In Python?"
    # Trailing turn must be in history!
    assert "Assistant: Yes, Python too." in history


def test_context_budget_preserves_prompt_header():
    """Verify _enforce_context_budget preserves BOS and task/goal tokens rather than dropping prompt start."""
    input_ids = [1, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
    # Budget of 6 tokens
    budgeted = _enforce_context_budget(input_ids, max_context=10, max_new_tokens=4)
    assert len(budgeted) <= 6
    assert budgeted[0] == 1  # BOS preserved
    # Check that initial goal/instruction tokens are preserved (not dropped from the left)
    assert budgeted[1] == 50


def test_repetition_penalty_only_on_new_tokens():
    """Verify repetition penalty over generated[prompt_len:] doesn't penalize prompt words."""
    logits = torch.zeros(1, 100)
    prompt_ids = [10, 20, 30]
    generated_new = [40, 50]
    
    # Setting logit for token 20 (prompt word) and token 40 (newly generated word)
    logits[0, 20] = 5.0
    logits[0, 40] = 5.0
    
    # Apply penalty only to new tokens
    ctx = torch.tensor([generated_new], dtype=torch.long)
    penalized_logits = apply_repetition_penalty(logits, ctx, penalty=1.2)
    
    # Token 20 should be UNTOUCHED (5.0), token 40 should be PENALIZED (5.0 / 1.2 = 4.1667)
    assert torch.isclose(penalized_logits[0, 20], torch.tensor(5.0))
    assert penalized_logits[0, 40] < 5.0


def test_from_pretrained_directory_resolution(tmp_path):
    """Verify KanhaModel.from_pretrained resolves correctly when given a directory path."""
    model = KanhaModel(vocab_size=100, dim=64, n_layers=2, n_heads=2, ff_dim=128, max_seq_len=64)
    save_dir = tmp_path / "models_dir"
    save_dir.mkdir()
    model.save_pretrained(str(save_dir / "final_model.pt"))
    
    # Load passing directory instead of file path
    loaded = KanhaModel.from_pretrained(str(save_dir))
    assert isinstance(loaded, KanhaModel)
    assert loaded.dim == 64


def test_sft_dataset_terminal_eos_preservation(tmp_path):
    """Verify that SFTDataset preserves terminal EOS even on sequences sliced to max_len."""
    # Create mock dataset
    data_file = tmp_path / "sft.jsonl"
    with open(data_file, "w") as f:
        f.write(json.dumps({"instruction": "hi", "response": "hello world"}) + "\n")
    
    # Mock tokenizer
    class DummyTokenizer:
        BOS_ID = 1
        EOS_ID = 2
        def encode(self, text, add_bos=False, add_eos=False):
            ids = [10, 11, 12]
            if add_bos: ids = [self.BOS_ID] + ids
            if add_eos: ids = ids + [self.EOS_ID]
            return ids
            
    tok = DummyTokenizer()
    # Force max_len shorter than full length
    dataset = SFTDataset.__new__(SFTDataset)
    dataset.tokenizer = tok
    dataset.max_len = 5
    dataset.prompt_builder = PromptBuilder(include_rules=False)
    dataset.samples = [{"instruction": "hi", "response": "hello world"}]
    
    item = dataset[0]
    # Last label should be EOS_ID
    assert item["labels"][-1].item() == tok.EOS_ID


def test_lam_world_generation_has_mixed_statuses():
    """Verify TrajectoryGenerator._sample_world generates mixed task statuses including done."""
    gen = TrajectoryGenerator(num_trajectories=5)
    import random
    rng = random.Random(42)
    world = gen._sample_world(rng, min_size=10, max_size=15)
    
    statuses = {t.get("status") for t in world}
    assert "done" in statuses
    assert "todo" in statuses


def test_oracle_find_then_act_priority_not_swallowed_by_mark():
    """Verify goal 'mark X as high priority' correctly edits priority rather than marking done."""
    oracle = Oracle()
    info = {
        "tier": "tier3",
        "solver": "_solve_lookup_then_act",
        "goal": "mark submit taxes as high priority",
        "prio_val_val": "high",
        "slot_values": {"query": "submit taxes"},
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        env = TaskEnvironment(task_store_path=os.path.join(tmpdir, "tasks.json"), auto_save=False)
        env.reset(initial_tasks=[{"title": "submit taxes", "prio": "low", "status": "todo"}])
        
        steps = oracle.solve(info, env)
        actions = [s[0] for s in steps]
        # Should contain edit(..., field=priority, value=high), NOT done(id=...)
        assert any("field=priority" in a for a in actions)
        assert not any(a.startswith("done(") for a in actions)
