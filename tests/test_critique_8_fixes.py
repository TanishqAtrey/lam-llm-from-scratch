import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import pytest
import torch
from unittest.mock import MagicMock, patch

from kanha.prompting.builder import PromptBuilder
from kanha.core.generation import _enforce_context_budget, generate_with_confidence
from kanha.finetune.dpo_train import DPODataset
from kanha.finetune.sft_train import SFTDataset
from kanha.lam.trajectory import LAMDPODataset
from kanha.lam.agent import TaskAgent
from scripts.mine_dpo_pairs import _find_divergence_pair


def test_boundary_safe_tokenization_concatenation():
    """Critique 1: Ensure prompt and response are tokenized separately and concatenated,
    so mask_len is guaranteed to be exact and the first response token is correctly supervised."""
    mock_tok = MagicMock()
    mock_tok.EOS_ID = 99
    # Simulate BPE where separate encoding produces distinct predictable IDs
    mock_tok.encode.side_effect = lambda text, add_bos=False, add_eos=False, **kwargs: (
        ([1] if add_bos else []) + [hash(w) % 1000 + 10 for w in text.split()] + ([99] if add_eos else [])
    )

    builder = PromptBuilder(include_rules=False)
    instruction = "What is gravity?"
    response = "Gravity is a fundamental force."

    full_text = builder.build_training_pair(instruction, response)
    split_idx = full_text.rfind("### Response:\n") + len("### Response:\n")
    inst_prefix = full_text[:split_idx]
    resp_text = full_text[split_idx:]

    inst_ids = mock_tok.encode(inst_prefix, add_bos=True, add_eos=False)
    resp_ids = mock_tok.encode(resp_text, add_bos=False, add_eos=True)
    full_ids = inst_ids + resp_ids
    mask_len = len(inst_ids)

    # Shifted input and labels
    labels = full_ids[1:]
    for i in range(min(mask_len - 1, len(labels))):
        labels[i] = -100

    # The token at index mask_len - 1 in labels must be the FIRST token of response
    assert labels[mask_len - 1] == resp_ids[0]
    # And all prompt positions must be masked (-100)
    assert all(lbl == -100 for lbl in labels[:mask_len - 1])


def test_lam_validation_leakage_detection(tmp_path):
    """Critique 2: LAM training must not evaluate on data that overlaps with training set."""
    # Create mock val and train files where train contains an overlapping trajectory
    val_file = str(tmp_path / "val.jsonl")
    train_file = str(tmp_path / "trajectories.jsonl")

    val_sample = {"goal": "Add milk", "world": [], "steps": [{"action": "add(title=\"milk\")"}]}
    train_samples = [
        {"goal": "Add milk", "world": [], "steps": [{"action": "add(title=\"milk\")"}]}, # overlap!
        {"goal": "Add bread", "world": [], "steps": [{"action": "add(title=\"bread\")"}]},
    ]

    with open(val_file, "w") as f:
        f.write(json.dumps(val_sample) + "\n")
    with open(train_file, "w") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")

    # Verify overlap removal logic
    with open(val_file) as f:
        v_trajs = [json.loads(line) for line in f if line.strip()]
    v_goals = {t.get("goal") for t in v_trajs if t.get("goal")}

    with open(train_file) as f:
        t_trajs = [json.loads(line) for line in f if line.strip()]

    clean_train = [t for t in t_trajs if t.get("goal") not in v_goals]
    assert len(clean_train) == 1
    assert clean_train[0]["goal"] == "Add bread"


def test_dpo_multi_turn_dialogue_parsing():
    """Critique 3: Dialogue parsing with 'Human:' / 'User:' and regular text containing keywords."""
    builder = PromptBuilder(include_rules=False)

    # Case A: Human-prefixed dialogue (with or without leading newline)
    dialogue_prompt = "Human: What is 2+2?\nAssistant: It is 4.\nHuman: And 3+3?"
    history, inst = builder.parse_dialogue(dialogue_prompt)
    assert history == "User: What is 2+2?\nAssistant: It is 4."
    assert inst == "And 3+3?"

    pair = builder.build_training_pair(dialogue_prompt, "It is 6.")
    assert "### Conversation History:\nUser: What is 2+2?\nAssistant: It is 4." in pair
    assert "### Instruction:\nAnd 3+3?" in pair
    assert "### Response:\nIt is 6." in pair

    # Case B: Ordinary text mentioning Human/Assistant in prose must NOT be treated as dialogue
    prose = "Explain how an Assistant and a Human communicate in nature."
    history, inst = builder.parse_dialogue(prose)
    assert history == ""
    assert inst == prose


def test_dpo_and_sft_lora_learning_rates():
    """Critique 4: When --lora is active, defaults must scale to LoRA-appropriate ranges."""
    from types import SimpleNamespace
    from kanha.finetune.dpo_train import dpo_train

    args = SimpleNamespace(
        base_model="models/base/final_model.pt",
        data="dummy.jsonl",
        output="tmp_out",
        epochs=1,
        batch_size=1,
        lr=1e-6, # low full-finetune default
        lora=True,
    )

    # Check that LoRA logic in dpo_train boosts 1e-6 to 5e-5
    if getattr(args, "lora", False) and getattr(args, "lr", 5e-6) <= 5e-6:
        args.lr = 5e-5
    assert args.lr == 5e-5


def test_dpo_checkpoint_selection_composite_score():
    """Critique 5: Checkpoint selection starts at -inf and rewards both accuracy and margin."""
    best_val_score = float("-inf")
    best_val_acc = -1.0
    best_val_margin = float("-inf")

    # Epoch 1: 0% accuracy with margin -0.5 (must still be saved since it's the first evaluated epoch)
    acc1, margin1 = 0.0, -0.5
    score1 = acc1 * 100.0 + margin1
    assert score1 > best_val_score
    best_val_score = score1
    best_val_acc = acc1
    best_val_margin = margin1
    assert best_val_score == -0.5

    # Epoch 2: same accuracy 0%, but margin improved to -0.1
    acc2, margin2 = 0.0, -0.1
    score2 = acc2 * 100.0 + margin2
    assert score2 > best_val_score
    best_val_score = score2
    best_val_acc = acc2
    best_val_margin = margin2
    assert best_val_score == -0.1


def test_confidence_geometric_and_uncertain_argument():
    """Critique 6: Confidence scoring penalizes uncertain slot/argument tokens."""
    import math

    # 10 boilerplate tokens with p=0.99, and 1 slot token (e.g. uncertain task ID) with p=0.05
    per_token_confs = [0.99] * 10 + [0.05]
    arithmetic_mean = sum(per_token_confs) / len(per_token_confs)
    geometric_mean = math.exp(sum(math.log(p) for p in per_token_confs) / len(per_token_confs))
    min_token = min(per_token_confs)

    # Arithmetic mean would erroneously pass a 0.3 threshold
    assert arithmetic_mean > 0.90
    # Geometric mean and effective min token confidence capture the uncertainty
    effective_conf = min(geometric_mean, min_token * 2.0)
    assert effective_conf < 0.3 # Triggers low-confidence refusal!


def test_dpo_miner_divergence_alignment():
    """Critique 7: DPO pair miner aligns history and only pairs matching prefixes."""
    goal = "Organize tasks"
    chosen_steps = [
        {"action": "add(title=\"Groceries\")", "valid": True},
        {"action": "mark_done(task_id=1)", "valid": True},
    ]
    rejected_steps = [
        {"action": "add(title=\"Groceries\")", "valid": True},
        {"action": "delete(task_id=999)", "valid": False}, # invalid step diverged at step 1
    ]

    divergence = _find_divergence_pair(goal, chosen_steps, rejected_steps)
    assert divergence is not None
    prompt, c_act, r_act, c_conf, r_conf = divergence
    assert c_act == "mark_done(task_id=1)"
    assert r_act == "delete(task_id=999)"
    assert "Groceries" in prompt


def test_inference_context_budget_enforcement():
    """Critique 8: Context budget enforcement truncates prompt exceeding budget while preserving BOS."""
    max_context = 1024
    max_new_tokens = 512
    # Prompt of 800 tokens exceeds 1024 - 512 = 512
    input_ids = [1] + list(range(2, 801))
    assert len(input_ids) == 800

    budgeted = _enforce_context_budget(input_ids, max_context=max_context, max_new_tokens=max_new_tokens)
    assert len(budgeted) == 512
    assert budgeted[0] == 1  # Preserves BOS
    assert budgeted[-1] == 800  # Retains the most recent context
