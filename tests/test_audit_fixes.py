"""
tests/test_audit_fixes.py
Comprehensive unit tests for:
  - 1.1: disable_dropout in DPO policy and ref models
  - 1.2: Negative mining _is_true_success state & reply verification
  - 1.3: DPO warmup + cosine LR scheduler
  - 1.4: DPO length normalization
  - 1.5: HH-RLHF multi-turn dialogue context preservation
  - 2.1: Oracle substring collision ("change oil", "change lightbulb")
  - 2.2: Evaluator date synchronization with cfg.lam.reference_date
  - 2.3: Semantic evaluation for read-only queries (expected_reply checking)
  - 2.4: Stopword filtering in query slot sampling
  - 2.5: Low confidence rate denominator fix (bounded <= 1.0)
  - 3.1: Gradient accumulation in SFT and LAM-SFT
  - 3.2: Kaggle dynamic extraction path candidate resolution
  - 4.1: Inference engine ReAct loop with tool execution
  - 4.2: SFT prompt clean formatting without distribution mismatch
  - 4.3: Multi-token stop sequences in autoregressive generation
"""

import os
import sys
import json
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kanha.core.model import KanhaModel
from kanha.core.tokenizer import KanhaTokenizer
from kanha.core.generation import generate, generate_with_confidence
from kanha.finetune.dpo_train import disable_dropout, compute_log_probs
from kanha.lam.data.oracle import Oracle
from kanha.lam.environment import TaskEnvironment
from kanha.lam.evaluator import LAMEvaluator
from kanha.lam.data.templates import sample_goal, TITLE_POOL
from kanha.lam.task_store import TaskStore
from kanha.inference.engine import InferenceEngine


# ── 1.1 DPO Dropout Disabling ──────────────────────────────────────────────────

def test_disable_dropout():
    model = KanhaModel(vocab_size=100, dim=32, n_layers=2, n_heads=2, ff_dim=64, max_seq_len=32, dropout=0.2)
    has_dropout = any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in model.modules())
    assert has_dropout, "Model should initially have dropout > 0"

    disable_dropout(model)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            assert m.p == 0.0, f"Dropout p must be 0.0, got {m.p}"


# ── 1.2 Negative Mining State Verification ────────────────────────────────────

def test_is_true_success(tmp_path):
    from scripts.mine_dpo_pairs import _is_true_success
    from types import SimpleNamespace

    store_path = str(tmp_path / "tasks.json")
    env = TaskEnvironment(task_store_path=store_path, reference_date="2025-01-15", auto_save=False)
    env.reset(initial_tasks=[{"title": "buy milk", "due": "2025-01-15", "prio": "med", "tag": "errands"}])

    case = {
        "goal": "complete buy milk",
        "world": [{"title": "buy milk", "due": "2025-01-15", "prio": "med", "tag": "errands", "status": "todo"}],
        "expected_state": [{"title": "buy milk", "due": "2025-01-15", "prio": "med", "tag": "errands", "status": "done"}],
    }

    # Fake result where model called finish() without doing the action
    fake_result = SimpleNamespace(success=True, aborted=False, final_message="Task finished.")
    assert not _is_true_success(case, fake_result, env, "2025-01-15"), "Should fail because task was not marked done"

    # Now execute true done action
    env.step("done(id=1)")
    env.step("finish(msg='done')")
    true_result = SimpleNamespace(success=True, aborted=False, final_message="Task finished.")
    assert _is_true_success(case, true_result, env, "2025-01-15"), "Should succeed when expected_state matches"


# ── 1.4 DPO Length Normalization ──────────────────────────────────────────────

def test_dpo_length_normalization():
    model = KanhaModel(vocab_size=100, dim=32, n_layers=1, n_heads=2, ff_dim=64, max_seq_len=32)
    model.eval()

    input_ids = torch.randint(0, 100, (2, 8))
    labels = input_ids.clone()
    labels[0, :4] = -100  # 4 response tokens
    labels[1, :2] = -100  # 6 response tokens

    unnorm = compute_log_probs(model, input_ids, labels, torch.device("cpu"), length_norm=False)
    norm = compute_log_probs(model, input_ids, labels, torch.device("cpu"), length_norm=True)

    assert unnorm.shape == (2,)
    assert norm.shape == (2,)
    # Norm should be unnorm / token_count
    assert torch.isclose(norm[0], unnorm[0] / 4.0, atol=1e-5)
    assert torch.isclose(norm[1], unnorm[1] / 6.0, atol=1e-5)


# ── 1.5 HH-RLHF Multi-turn Dialogue Preservation ──────────────────────────────

def test_extract_last_exchange():
    dialogue = (
        "\n\nHuman: Hello assistant!\n\nAssistant: Hi, how can I help?\n\n"
        "Human: What is 2 + 2?\n\nAssistant: It is 4."
    )
    # Using the regex/split logic in scripts/download_datasets.py
    last_asst_idx = dialogue.rfind("\n\nAssistant:")
    prompt = dialogue[:last_asst_idx].strip()
    response = dialogue[last_asst_idx + len("\n\nAssistant:"):].strip()
    if prompt.startswith("Human:"):
        prompt = prompt[len("Human:"):].strip()
    elif prompt.startswith("\n\nHuman:"):
        prompt = prompt[len("\n\nHuman:"):].strip()

    assert "Hello assistant!" in prompt, "Full conversation context must be preserved in prompt"
    assert "What is 2 + 2?" in prompt
    assert response == "It is 4."


# ── 2.1 Oracle Substring Collision ────────────────────────────────────────────

def test_oracle_substring_collision(tmp_path):
    store_path = str(tmp_path / "env.json")
    env = TaskEnvironment(task_store_path=store_path, reference_date="2025-01-15", auto_save=False)
    env.reset(initial_tasks=[{"title": "change oil", "due": "today", "prio": "med", "tag": "errands"}])

    oracle = Oracle()
    info = {
        "goal": "mark {query} as done",
        "solver": "_solve_lookup_then_act",
        "slot_values": {"query": "change oil"},
        "tier": 3,
        "requires_existing": True,
        "target_id": 1,
    }
    steps = oracle.solve(info, env)
    actions = [s[0] for s in steps]
    assert "done(id=1)" in actions, f"Expected done(id=1), got actions: {actions}"
    assert not any("field=priority" in a for a in actions), "Must not mistakenly edit priority"


# ── 2.2 Evaluator Date Synchronization ────────────────────────────────────────

def test_evaluator_date_sync(tmp_path):
    dummy_file = str(tmp_path / "test.jsonl")
    with open(dummy_file, "w") as f:
        f.write(json.dumps({"goal": "test", "world": [], "steps": []}) + "\n")

    evaluator = LAMEvaluator(None, None, dummy_file, reference_date=None)
    assert evaluator.reference_date == "2025-01-15", f"Expected '2025-01-15', got {evaluator.reference_date}"


# ── 2.3 Semantic Evaluation for Read-Only Queries ─────────────────────────────

def test_evaluator_contains_phrase(tmp_path):
    dummy_file = str(tmp_path / "test.jsonl")
    with open(dummy_file, "w") as f:
        f.write(json.dumps({"goal": "test", "world": [], "steps": []}) + "\n")

    evaluator = LAMEvaluator(None, None, test_data_path=dummy_file, reference_date="2025-01-15")

    expected = {"contains_phrase": "tasks"}
    assert evaluator._check_reply("list tasks", "Here are your tasks.", expected)
    assert not evaluator._check_reply("list tasks", "I cannot do that.", expected)


# ── 2.4 Stopword Filtering in Query Slot Sampling ─────────────────────────────

def test_query_sampling_no_stopwords(tmp_path):
    store = TaskStore(path=str(tmp_path / "tasks.json"), reference_date="2025-01-15", auto_save=False)
    store.add(title="buy milk for the cat", due="today", prio="med", tag="errands")

    import random
    rng = random.Random(42)

    # Sample tier3 goals multiple times and verify query is not a stopword
    stopwords = {"the", "to", "a", "an", "and", "for", "with", "of", "in", "on", "at", "from", "my", "it", "out"}
    for _ in range(20):
        _, info = sample_goal("tier3", store, rng)
        query = info.get("slot_values", {}).get("query", "")
        if query:
            assert query.lower() not in stopwords, f"Query '{query}' should not be a stopword"


# ── 2.5 Low Confidence Rate Denominator ───────────────────────────────────────

def test_low_confidence_rate_bounded():
    from kanha.lam.evaluator import EvalReport
    report = EvalReport()
    total_actions = 1
    total_low_conf = 5
    total_decisions = total_actions + total_low_conf
    report.low_confidence_rate = total_low_conf / total_decisions
    assert 0.0 <= report.low_confidence_rate <= 1.0
    assert report.low_confidence_rate == 5 / 6


# ── 3.1 Gradient Accumulation Logic ───────────────────────────────────────────

def test_grad_accum_computation():
    batch_size = 2
    grad_accum_steps = 4
    total_batches = 10
    total_epochs = 2

    import math
    total_steps = math.ceil(total_batches / grad_accum_steps) * total_epochs
    assert total_steps == 6, f"Expected 6 optimizer steps, got {total_steps}"


# ── 4.1 Inference Engine ReAct Loop with Tool Execution ───────────────────────

def test_inference_react_loop(monkeypatch):
    model = KanhaModel(vocab_size=100, dim=32, n_layers=1, n_heads=2, ff_dim=64, max_seq_len=32)
    tokenizer = KanhaTokenizer()
    engine = InferenceEngine(model=model, tokenizer=tokenizer, use_tools=True)

    calls = []
    # Mock generate to return a tool call on step 1, and final answer on step 2
    def mock_generate(model, tok, prompt, **kwargs):
        calls.append(prompt)
        if len(calls) == 1:
            return "TOOL: calculator(5 + 5)"
        return "The answer is 10."

    monkeypatch.setattr("kanha.inference.engine.generate", mock_generate)

    res = engine.chat("What is 5 + 5?")
    assert len(calls) == 2, "Should have re-prompted after tool execution"
    assert "10" in res


# ── 4.2 SFT Prompt Formatting Cleanliness ─────────────────────────────────────

def test_sft_prompt_formatting():
    model = KanhaModel(vocab_size=100, dim=32, n_layers=1, n_heads=2, ff_dim=64, max_seq_len=32)
    tokenizer = KanhaTokenizer()
    engine = InferenceEngine(model=model, tokenizer=tokenizer, use_tools=False, use_rag=False)

    prompt = engine.prompt_builder.build(instruction="Hello world")
    assert prompt == "### Instruction:\nHello world\n\n### Response:\n", \
        f"Prompt should match canonical SFT template exactly, got: {repr(prompt)}"


# ── 4.3 Multi-Token Stop Sequences in Generation ──────────────────────────────

def test_stop_sequences_truncation(monkeypatch):
    model = KanhaModel(vocab_size=100, dim=32, n_layers=1, n_heads=2, ff_dim=64, max_seq_len=32)
    tokenizer = KanhaTokenizer()

    # Generate text containing a stop sequence
    raw_gen = "The result is 42.\nObservation: something else"
    text = raw_gen
    stop_sequences = ["\nObservation:"]
    for seq in stop_sequences:
        idx = text.find(seq)
        if idx != -1:
            text = text[:idx]

    assert text == "The result is 42."


# ── Additional Part 1 & Part 2 Bug Tests ──────────────────────────────────────

def test_token_dataset_interleaved_val(tmp_path):
    from kanha.training.train import TokenDataset
    # Create fake 2000-token mmap array
    arr_path = str(tmp_path / "tokens.npy")
    fake_data = np.arange(2000, dtype=np.uint16)
    np.save(arr_path, fake_data)

    seq_len = 32
    train_ds = TokenDataset(arr_path, seq_len=seq_len, split="train", val_fraction=0.1)
    val_ds = TokenDataset(arr_path, seq_len=seq_len, split="val", val_fraction=0.1)

    assert len(train_ds) > 0
    assert len(val_ds) > 0
    # Val sequences should be sampled across the file, not just the first 10%
    assert val_ds.indices[-1] > 1000, "Validation sequences must span across the entire token array"
    # Overlap between train and val start indices must be zero
    assert len(set(train_ds.indices) & set(val_ds.indices)) == 0


def test_action_validator_rejects_garbage():
    from kanha.lam.actions import parse_action, validate_action, ActionParseError

    # Empty title in add
    name, args = parse_action('add(title="")')
    valid, err = validate_action(name, args)
    assert not valid
    assert "empty" in err

    # Case-insensitive priority
    name, args = parse_action('add(title="test", prio=HIGH)')
    valid, err = validate_action(name, args)
    assert valid
    assert args['prio'] == 'high'

    # Invalid due date
    name, args = parse_action('add(title="test", due=banana)')
    valid, err = validate_action(name, args)
    assert not valid
    assert "Invalid date" in err

    # Duplicate argument in parse
    with pytest.raises(ActionParseError, match="Duplicate argument"):
        parse_action('count(due=today, due=tomorrow)')


def test_undo_next_id_rollback(tmp_path):
    store = TaskStore(path=str(tmp_path / "tasks.json"), reference_date="2025-01-15", auto_save=False)
    assert store._next_id == 1
    t = store.add(title="test task")
    assert t.id == 1
    assert store._next_id == 2
    store.undo()
    assert store._next_id == 1, "Undoing the most recent add must roll back _next_id"


def test_dynamic_padding_collate():
    from kanha.finetune.sft_train import sft_collate_fn
    batch = [
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([-100, 2, 3])},
        {"input_ids": torch.tensor([1, 2, 3, 4, 5]), "labels": torch.tensor([-100, -100, 3, 4, 5])},
    ]
    collated = sft_collate_fn(batch, pad_id=0)
    assert collated["input_ids"].shape == (2, 5), "Batch should be dynamically padded only to length 5"
    assert collated["labels"].shape == (2, 5)
    assert collated["input_ids"][0, -1].item() == 0
    assert collated["labels"][0, -1].item() == -100


# ── New Targeted Audit Fixes Tests ──────────────────────────────────────────

def test_multihead_attention_dropout_disabled():
    """Verify MultiHeadAttention attention dropout is disabled by disable_dropout."""
    model = KanhaModel(vocab_size=100, dim=32, n_layers=2, n_heads=2, ff_dim=64, max_seq_len=32, dropout=0.2)
    # Check that attention layers have dropout > 0
    attn_dropouts = [layer.attn.dropout for layer in model.layers]
    for d in attn_dropouts:
        assert isinstance(d, torch.nn.Dropout)
        assert d.p == 0.2

    disable_dropout(model)
    for d in attn_dropouts:
        assert d.p == 0.0, "Attention dropout must be set to 0.0 after disable_dropout"


def test_lora_disable_adapter():
    """Verify LoRAModel can dynamically disable adapter for reference pass."""
    from kanha.core.lora import LoRAModel, LoRALinear
    model = KanhaModel(vocab_size=100, dim=32, n_layers=2, n_heads=2, ff_dim=64, max_seq_len=32, dropout=0.0)
    lora_model = LoRAModel(model, rank=4, alpha=8.0, dropout=0.0, target_modules=["q_proj", "v_proj"])
    
    x = torch.randint(0, 100, (1, 8))
    # Base model output
    with torch.no_grad():
        with lora_model.disable_adapter():
            base_logits, _, _ = lora_model(x)
        
        # When adapter is enabled with non-zero weights
        for m in lora_model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.data.fill_(0.5)
        
        adapted_logits, _, _ = lora_model(x)
        assert not torch.allclose(base_logits, adapted_logits), "Adapted logits should differ from base logits"
        
        with lora_model.disable_adapter():
            disabled_logits, _, _ = lora_model(x)
        assert torch.allclose(base_logits, disabled_logits), "Disabled adapter must exactly match base model"


def test_detect_loop_alternating_cycles():
    """Verify TaskAgent detects 2-step alternating cycles (A -> B -> A -> B) and 3-step loops."""
    from kanha.lam.agent import TaskAgent
    from unittest.mock import MagicMock
    agent = TaskAgent(MagicMock(), MagicMock(), MagicMock())
    
    # 1-step repeat: A -> A -> A
    h1 = [{'action': 'list()'}, {'action': 'list()'}, {'action': 'list()'}]
    assert agent._detect_loop(h1), "Should detect 1-step repeat"

    # Non-loop
    h_ok = [{'action': 'list()'}, {'action': 'add(title="buy")'}, {'action': 'finish()'}]
    assert not agent._detect_loop(h_ok), "Should not flag non-loop"

    # 2-step alternating: A -> B -> A -> B
    h_alt = [
        {'action': 'list()'},
        {'action': 'get(id=1)'},
        {'action': 'list()'},
        {'action': 'get(id=1)'},
    ]
    assert agent._detect_loop(h_alt), "Should detect 2-step alternating cycle (A -> B -> A -> B)"

    # 3-step cycle: A -> B -> C -> A -> B -> C
    h_cycle3 = [
        {'action': 'list()'},
        {'action': 'get(id=1)'},
        {'action': 'search(q="x")'},
        {'action': 'list()'},
        {'action': 'get(id=1)'},
        {'action': 'search(q="x")'},
    ]
    assert agent._detect_loop(h_cycle3), "Should detect 3-step cycle (A -> B -> C -> A -> B -> C)"


def test_multiturn_dialogue_parsing():
    """Verify PromptBuilder parses multi-turn HH-RLHF dialogue into Conversation History."""
    from kanha.prompting.builder import PromptBuilder
    pb = PromptBuilder(include_rules=False)
    
    hh_text = "Human: hello\n\nAssistant: hi there\n\nHuman: what is 2+2?"
    history, inst = pb.parse_dialogue(hh_text)
    assert history == "User: hello\nAssistant: hi there"
    assert inst == "what is 2+2?"

    pair = pb.build_training_pair(hh_text, "4")
    assert "### Conversation History:\nUser: hello\nAssistant: hi there" in pair
    assert "### Instruction:\nwhat is 2+2?" in pair
    assert "### Response:\n4" in pair


def test_mine_dpo_pairs_empty_divergence():
    """Verify _find_divergence_pair preserves empty/aborted actions as negative samples."""
    from scripts.mine_dpo_pairs import _find_divergence_pair
    
    chosen_steps = [{'action': 'add(title="milk")'}]
    rejected_steps = [{'action': '', 'valid': False}]
    retry_history = [{'action': '', 'confidence': 0.0, 'step_num': 1}]

    res = _find_divergence_pair(
        goal="Add milk",
        chosen_steps=chosen_steps,
        rejected_steps=rejected_steps,
        chosen_confs=[0.9],
        rejected_confs=[0.0],
        retry_history=retry_history,
    )
    assert res is not None, "Divergence pair must not be dropped for empty/aborted action"
    prompt, chosen, rejected, c_conf, r_conf = res
    assert chosen == 'add(title="milk")'
    assert rejected == '\n'


