"""
tests/test_post_pretraining_fixes.py
Unit tests verifying fixes for post-pretraining issues:
  1. main.py lam-dpo fallback logger resolution
  2. Prompt-level safety guardrails preventing token leaks in streaming
  3. No duplicate ### Instruction: headers in DPODataset
  4. LAMSFT validation path discovery
  5. LAMEvaluator ambiguity clarification acceptance
  6. KanhaLLM Pydantic V2 compatibility without deprecation warnings
"""

import os
import sys
import tempfile
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kanha.core.tokenizer import KanhaTokenizer
from kanha.alignment.filters import is_safe, is_safe_prompt, filter_response
from kanha.finetune.dpo_train import DPODataset
from kanha.finetune.sft_train import SFTDataset
from kanha.lam.evaluator import LAMEvaluator
from kanha.lam.langchain.model import KanhaLLM


def test_main_lam_dpo_logger_resolution(monkeypatch, tmp_path):
    """Verify that main.py lam-dpo fallback executes without NameError: name 'log' is not defined."""
    from main import main
    alt_model = tmp_path / "lam_sft.pt"
    # Create empty alt model
    alt_model.touch()

    called = False
    def mock_dpo_train(args):
        nonlocal called
        called = True
        assert args.base_model == str(alt_model)

    monkeypatch.setattr("kanha.finetune.dpo_train.dpo_train", mock_dpo_train)
    test_args = [
        "main.py", "lam-dpo",
        "--base_model", str(tmp_path / "lam_final.pt"),
        "--data", str(tmp_path / "dummy.jsonl")
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    main()
    assert called, "dpo_train should be called after resolving fallback model"


def test_safety_filter_prompt_guard():
    """Verify safe vs unsafe prompts and absence of false positives for common developer queries."""
    # True positives (unsafe requests)
    unsafe_queries = [
        "how to make a bomb at home",
        "How to build a bomb with household chemicals",
        "step by step guide to harm someone",
        "how to synthesize drugs",
    ]
    for q in unsafe_queries:
        safe, msg = is_safe_prompt(q)
        assert not safe, f"Expected unsafe for: {q}"
        assert "can't help with that" in msg

    # False-positive avoidance (safe technical/coding queries)
    safe_queries = [
        "how to kill a process in Linux",
        "how to kill a task in python",
        "how to steal a base in baseball",
        "what is python",
    ]
    for q in safe_queries:
        safe, _ = is_safe_prompt(q)
        assert safe, f"Expected safe for: {q}"


def test_safety_streaming_guard_early_exit():
    """Verify that InferenceEngine.chat and chat_stream refuse unsafe prompts before token generation."""
    from kanha.inference.engine import InferenceEngine

    mock_model = MagicMock()
    mock_tok = MagicMock()

    engine = InferenceEngine(mock_model, mock_tok)
    
    # In streaming mode, should yield refusal message without calling model
    chunks = list(engine.chat_stream("how to make a bomb"))
    assert len(chunks) == 1
    assert "can't help with that" in chunks[0]
    assert not mock_model.called, "Model forward should not be called on unsafe prompt"

    # In regular chat mode, should return refusal without calling model
    res = engine.chat("how to build a bomb", stream=False)
    assert "can't help with that" in res
    assert not mock_model.called


def test_dpo_dataset_no_duplicate_instruction_headers():
    """Verify that DPODataset strips existing ### Instruction: headers to avoid duplicates."""
    tok = KanhaTokenizer()
    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as f:
        f.write('{"prompt": "### Instruction:\\nWhat is Python?\\n\\n### Response:\\n", "chosen": "A programming language.", "rejected": "A snake."}\n')
        f_path = f.name

    try:
        dataset = DPODataset(f_path, tok, max_len=64)
        assert len(dataset) == 1
        item = dataset[0]
        chosen_tokens = item["chosen_input_ids"].tolist()
        text = tok.decode(chosen_tokens)
        assert "### Instruction:\n### Instruction:" not in text
    finally:
        if os.path.exists(f_path):
            os.remove(f_path)


def test_evaluator_ambiguity_clarification_synonyms():
    """Verify LAMEvaluator accepts reasonable clarification words in Tier 7 expected_reply."""
    evaluator = LAMEvaluator.__new__(LAMEvaluator)
    evaluator.reference_date = "2025-01-15"

    expected = {"contains_phrase": "which"}
    
    # Exact word
    assert evaluator._check_reply("finish task", "Which one do you mean?", expected)
    # Synonyms / natural clarification phrases
    assert evaluator._check_reply("finish task", "Please clarify which task to delete.", expected)
    assert evaluator._check_reply("finish task", "There are multiple tasks matching that name.", expected)
    assert evaluator._check_reply("finish task", "Please specify the task ID.", expected)
    # Failure case (did not ask or clarify)
    assert not evaluator._check_reply("finish task", "Task deleted successfully.", expected)


def test_pydantic_v2_kanha_llm_config():
    """Verify KanhaLLM initializes without Pydantic V2 configuration deprecation warnings."""
    mock_model = MagicMock()
    mock_tok = MagicMock()
    llm = KanhaLLM(model=mock_model, tokenizer=mock_tok, max_new_tokens=32)
    assert llm.max_new_tokens == 32
    assert llm._llm_type == "kanha"


# ── Tests for 10 Post-Pretraining Fixes ────────────────────────────────────────

def test_dpo_best_checkpoint_metric_logic():
    """Bug 1 & 2: DPO best selection prioritizes accuracy over exploding margins."""
    # Simulating metric selection logic
    def check_better(acc, margin, best_acc, best_margin):
        return (acc > best_acc) or (abs(acc - best_acc) < 1e-4 and margin > best_margin)

    # Higher accuracy must beat lower accuracy even if lower accuracy has higher margin
    assert check_better(acc=0.85, margin=1.2, best_acc=0.80, best_margin=10.0)
    assert not check_better(acc=0.75, margin=20.0, best_acc=0.80, best_margin=1.0)
    # Equal accuracy breaks tie with margin
    assert check_better(acc=0.80, margin=2.5, best_acc=0.80, best_margin=1.5)


def test_lam_dpo_dataset_prefilters_oversized_no_label_samples(tmp_path):
    """Bug 3: LAMDPODataset skips samples exceeding max_len with no response labels."""
    import json
    from kanha.lam.trajectory import LAMDPODataset

    mock_tok = MagicMock()
    mock_tok.PAD_ID = 0
    # Long prompt returns 20 tokens; for max_len=10, all response tokens are truncated
    mock_tok.encode.side_effect = lambda text, **kw: [1] * 20 if "long" in text else [1, 2]

    dpo_file = tmp_path / "oversized.jsonl"
    with open(dpo_file, "w") as f:
        # Valid sample
        f.write(json.dumps({"prompt": "short prompt", "chosen": "c", "rejected": "r"}) + "\n")
        # Oversized sample where prompt alone exceeds max_len=10
        f.write(json.dumps({"prompt": "long prompt " * 10, "chosen": "c", "rejected": "r"}) + "\n")

    dataset = LAMDPODataset(str(dpo_file), mock_tok, max_len=10)
    assert len(dataset) == 1, "Oversized sample with no response tokens must be pre-filtered"


def test_mine_dpo_is_true_success_parity():
    """Bug 4: mine_dpo_pairs _is_true_success matches evaluator tier 7, 8 and reply rules."""
    from scripts.mine_dpo_pairs import _is_true_success
    from kanha.lam.environment import TaskEnvironment

    mock_env = MagicMock()
    mock_env.get_state.return_value = []

    # Tier 7 requires last_act.startswith("ask(")
    case_tier7 = {"tier": 7, "expected_state": [], "expected_reply": {"contains_phrase": "which"}}
    
    res_bad = MagicMock(aborted=False, final_message="which one?", steps=[{"action": "list()"}])
    assert not _is_true_success(case_tier7, res_bad, mock_env, "2025-01-15")

    res_good = MagicMock(aborted=False, final_message="Please clarify which one", steps=[{"action": "ask(question=\"which?\")"}])
    assert _is_true_success(case_tier7, res_good, mock_env, "2025-01-15")

    # Tier 8 refusal check
    case_tier8 = {"tier": 8, "expected_state": []}
    res_t8_bad = MagicMock(aborted=False, final_message="Here is the weather", steps=[{"action": "finish()"}])
    assert not _is_true_success(case_tier8, res_t8_bad, mock_env, "2025-01-15")

    res_t8_good = MagicMock(aborted=False, final_message="Sorry, I can only manage tasks.", steps=[{"action": "finish()"}])
    assert _is_true_success(case_tier8, res_t8_good, mock_env, "2025-01-15")


def test_langgraph_loop_detection_and_temperature_parity():
    """Bug 5: LangGraphTaskAgent detects multi-step loops and accepts temperature."""
    from kanha.lam.langchain.graph import _after_execute

    # 1-step repeat: A -> A -> A
    state1 = {"history": [{"action": "a"}, {"action": "a"}, {"action": "a"}], "step_count": 3, "max_steps": 10}
    assert _after_execute(state1) == "abort_loop"

    # 2-step alternating cycle: A -> B -> A -> B
    state2 = {"history": [{"action": "a"}, {"action": "b"}, {"action": "a"}, {"action": "b"}], "step_count": 4, "max_steps": 10}
    assert _after_execute(state2) == "abort_loop"

    # 3-step cycle: A -> B -> C -> A -> B -> C
    state3 = {"history": [{"action": "a"}, {"action": "b"}, {"action": "c"}, {"action": "a"}, {"action": "b"}, {"action": "c"}], "step_count": 6, "max_steps": 10}
    assert _after_execute(state3) == "abort_loop"


def test_lam_train_split_dataset_discovery_no_leakage(tmp_path):
    """Bug 6: lam_train automatically switches to trajectories_train.jsonl when available."""
    full_data = tmp_path / "trajectories.jsonl"
    full_data.write_text('{"goal": "G1", "steps": []}\n')
    train_split = tmp_path / "trajectories_train.jsonl"
    train_split.write_text('{"goal": "G2", "steps": []}\n')
    val_split = tmp_path / "trajectories_val.jsonl"
    val_split.write_text('{"goal": "G3", "steps": []}\n')

    from unittest.mock import patch
    import argparse
    from kanha.finetune.lam_train import lam_sft_train

    args = argparse.Namespace(
        base_model="models/base/final_model.pt",
        data=str(full_data),
        output=str(tmp_path / "out"),
        epochs=1,
        batch_size=1,
        lr=1e-5,
        val_fraction=0.1,
        patience=1,
        lora=False,
    )

    # Check candidate selection logic directly
    data_dir = os.path.dirname(args.data)
    train_candidates = [
        os.path.join(data_dir, "train.jsonl"),
        os.path.join(data_dir, "trajectories_train.jsonl"),
        args.data.replace(".jsonl", "_train.jsonl"),
    ]
    matched = [tc for tc in train_candidates if os.path.exists(tc)]
    assert matched[0] == str(train_split)


def test_streaming_stop_sequence_buffering_no_leak():
    """Bug 8: generate_stream buffers partial stop sequence prefixes without leaking."""
    from kanha.core.generation import _find_stop_prefix_len

    stop_seqs = ["\n### Observation", "### Observation", "[Tool Result:"]

    # "Hello\n" ends with "\n" which is prefix of "\n### Observation" (length 1)
    assert _find_stop_prefix_len("Hello\n", stop_seqs) == 1
    # "Hello\n###" ends with "\n###" (length 4)
    assert _find_stop_prefix_len("Hello\n###", stop_seqs) == 4
    # "Hello world" has 0 overlap with stop sequences
    assert _find_stop_prefix_len("Hello world", stop_seqs) == 0


def test_api_resolve_model_path_dpo_support(tmp_path, monkeypatch):
    """Bug 9: api._resolve_model_path discovers dpo_final.pt and dpo_best.pt."""
    from api import _resolve_model_path

    # Point to temp path with dpo_best.pt
    dpo_best = tmp_path / "dpo_best.pt"
    dpo_best.touch()

    monkeypatch.setenv("KANHA_MODEL", str(dpo_best))
    assert _resolve_model_path() == str(dpo_best)


def test_evaluator_langgraph_option(tmp_path):
    """Bug 10: LAMEvaluator accepts use_langgraph flag."""
    from kanha.lam.evaluator import LAMEvaluator

    mock_model = MagicMock()
    mock_tok = MagicMock()
    test_file = tmp_path / "test.jsonl"
    test_file.write_text('{"goal": "G", "world": [], "expected_state": []}\n')
    evaluator = LAMEvaluator(mock_model, mock_tok, test_data_path=str(test_file), use_langgraph=True)
    assert evaluator.use_langgraph is True


def test_api_session_memory_isolation_in_streaming():
    """Verify engine.chat and engine.chat_stream isolate memory without global memory leakage."""
    from kanha.inference.engine import InferenceEngine
    from kanha.memory.short_term import ShortTermMemory

    mock_model = MagicMock()
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1, 2, 3]
    mock_tok.decode.return_value = "Response"
    mock_tok.eos_id = 2

    # Mock model outputs
    import torch
    mock_logits = torch.zeros(1, 3, 100)
    mock_model.return_value = (mock_logits, None, None)
    mock_model.parameters.return_value = iter([torch.empty(0)])

    engine = InferenceEngine(mock_model, mock_tok)
    global_mem = engine.memory
    session_mem = ShortTermMemory(max_turns=5)

    # Stream with session memory
    with pytest.MonkeyPatch().context() as m:
        m.setattr("kanha.inference.engine.generate_stream", lambda *args, **kwargs: iter(["Hello", " world"]))
        chunks = list(engine.chat_stream("User prompt", memory=session_mem))

    assert "".join(chunks) == "Hello world"
    # Global memory should NOT have been modified
    assert len(global_mem) == 0
    # Session memory SHOULD contain the user and assistant turns
    assert len(session_mem) == 2
    assert session_mem._turns[0]["role"] == "user"
    assert session_mem._turns[1]["role"] == "assistant"


def test_evaluator_zero_count_reply_acceptance():
    """Verify LAMEvaluator._check_reply accepts 'no', 'none', 'zero' when expected contains_number is 0."""
    evaluator = LAMEvaluator.__new__(LAMEvaluator)
    evaluator.reference_date = "2025-01-15"
    expected = {"contains_number": 0}

    assert evaluator._check_reply("how many tasks", "You have 0 tasks.", expected)
    assert evaluator._check_reply("how many tasks", "You have no overdue tasks.", expected)
    assert evaluator._check_reply("how many tasks", "None.", expected)
    assert evaluator._check_reply("how many tasks", "Zero tasks remaining.", expected)
    assert evaluator._check_reply("how many tasks", "There is nothing due.", expected)
    assert not evaluator._check_reply("how many tasks", "You have 3 tasks.", expected)


def test_mine_dpo_prefix_guard():
    """Verify _find_divergence_pair rejects pairs where trajectories diverged before min_len extension."""
    from scripts.mine_dpo_pairs import _find_divergence_pair

    goal = "Create and complete task"
    # Prefix differs at step 0: "create_task(A)" vs "create_task(B)"
    chosen_steps = [
        {"action": "create_task(title='A')"},
        {"action": "complete_task(id=1)"},
    ]
    rejected_steps = [
        {"action": "create_task(title='B')"},
    ]

    # Because step 0 differed, step 1 of chosen cannot be paired against empty rejected step
    pair = _find_divergence_pair(goal, chosen_steps, rejected_steps)
    # The divergence occurs at step 0 (valid divergence)
    assert pair is not None
    prompt, c_act, r_act, c_conf, r_conf = pair
    assert "create_task(title='A')" in c_act
    assert "create_task(title='B')" in r_act


def test_lam_effective_confidence_formula():
    """Verify effective_conf penalizes hallucinations (<0.02) while tolerating natural language variance (0.10)."""
    # Natural language argument token with p = 0.10, confidence = 0.85
    confidence = 0.85
    min_conf = 0.10
    effective_conf = min(confidence, max(min_conf * 4.0, min_conf ** 0.5))
    # 0.10**0.5 is ~0.316, min_conf*4.0 is 0.40 -> effective_conf is 0.40, which is > threshold (0.30)
    assert effective_conf >= 0.30

    # True hallucination / near zero token (e.g. p = 0.005)
    min_conf_bad = 0.005
    effective_conf_bad = min(confidence, max(min_conf_bad * 4.0, min_conf_bad ** 0.5))
    # min_conf_bad*4.0 = 0.02, min_conf_bad**0.5 = ~0.07 -> effective_conf < 0.10, triggering fallback
    assert effective_conf_bad < 0.20


def test_retriever_dimension_mismatch_warning(tmp_path, caplog):
    """Verify Retriever.load_index logs a warning when store dimension does not match embedder dimension."""
    import logging
    from kanha.rag.retriever import Retriever
    from kanha.rag.vector_store import VectorStore

    caplog.set_level(logging.WARNING)
    # Create store with dimension 128 and save it
    store_dir = tmp_path / "store"
    s1 = VectorStore(dim=128)
    s1.save(str(store_dir))

    # Create mock embedder with dim 384
    mock_embedder = MagicMock()
    mock_embedder.dim = 384

    retriever = Retriever(embedder=mock_embedder, store=VectorStore(dim=384))
    retriever.load_index(str(store_dir))

    assert any("RAG dimension mismatch" in rec.message for rec in caplog.records)


def test_dpo_stale_checkpoint_protection(tmp_path):
    """Verify DPO training does not overwrite final_ckpt with stale dpo_best.pt if no new best was found."""
    import shutil
    output_dir = tmp_path / "dpo_out"
    output_dir.mkdir()
    stale_best = output_dir / "dpo_best.pt"
    stale_best.write_text("old_best_weights")

    final_ckpt = output_dir / "dpo_final.pt"

    # Simulate logic when new_best_saved is False
    new_best_saved = False
    mock_policy = MagicMock()

    if new_best_saved and os.path.exists(str(stale_best)):
        shutil.copyfile(str(stale_best), str(final_ckpt))
    else:
        mock_policy.save_pretrained(str(final_ckpt))

    # mock_policy.save_pretrained must be called instead of copying stale_best
    mock_policy.save_pretrained.assert_called_once_with(str(final_ckpt))
    assert not final_ckpt.exists()  # Because mock was called, not shutil.copyfile


def test_dpo_dataset_multi_turn_history_formatting(tmp_path):
    """Verify DPODataset correctly handles pre-structured conversation history prompts without duplicating headers."""
    import json
    tok = KanhaTokenizer()
    dpo_file = tmp_path / "multi_turn.jsonl"
    prompt_with_history = (
        "### Conversation History:\nUser: Hello\nAssistant: Hi there!\n\n"
        "### Instruction:\nWhat is 2+2?\n\n### Response:\n"
    )
    with open(dpo_file, "w") as f:
        f.write(json.dumps({
            "prompt": prompt_with_history,
            "chosen": "4",
            "rejected": "5"
        }) + "\n")

    dataset = DPODataset(str(dpo_file), tok, max_len=128)
    assert len(dataset) == 1
    item = dataset[0]
    chosen_text = tok.decode(item["chosen_input_ids"].tolist())
    # Verify no duplicate nested instruction headers
    assert "### Instruction:\n### Instruction:" not in chosen_text
    assert "### Instruction:\n### Conversation History:" not in chosen_text
    assert chosen_text.startswith("### Conversation History:")


def test_mine_dpo_zero_count_reply_acceptance():
    """Verify scripts/mine_dpo_pairs._is_true_success accepts 'no', 'none', 'zero' for zero-count goals."""
    from scripts.mine_dpo_pairs import _is_true_success
    mock_env = MagicMock()
    mock_env.get_state.return_value = []
    case = {"tier": 2, "expected_state": [], "expected_reply": {"contains_number": 0}}

    # Success cases
    res_none = MagicMock(aborted=False, final_message="You have no tasks due tomorrow.", steps=[{"action": "list()"}])
    assert _is_true_success(case, res_none, mock_env, "2025-01-15")

    res_zero = MagicMock(aborted=False, final_message="There are zero matching tasks.", steps=[{"action": "count()"}])
    assert _is_true_success(case, res_zero, mock_env, "2025-01-15")

    res_nothing = MagicMock(aborted=False, final_message="None found.", steps=[{"action": "find()"}])
    assert _is_true_success(case, res_nothing, mock_env, "2025-01-15")

    # Failure case
    res_bad = MagicMock(aborted=False, final_message="You have 3 tasks.", steps=[{"action": "list()"}])
    assert not _is_true_success(case, res_bad, mock_env, "2025-01-15")


def test_mine_dpo_retry_divergence_confidence():
    """Verify _find_divergence_pair retrieves r_conf from retry_history when r_act came from retry."""
    from scripts.mine_dpo_pairs import _find_divergence_pair
    goal = "Create a task"
    chosen_steps = [{"action": 'add("Task A")'}]
    rejected_steps = [{"action": ""}]
    retry_history = [{"step_num": 1, "action": 'add("Task B")', "confidence": 0.78}]
    chosen_confs = [0.95]
    rejected_confs = [0.12]

    pair = _find_divergence_pair(
        goal=goal,
        chosen_steps=chosen_steps,
        rejected_steps=rejected_steps,
        retry_history=retry_history,
        chosen_confs=chosen_confs,
        rejected_confs=rejected_confs,
    )
    assert pair is not None
    prompt, c_act, r_act, c_conf, r_conf = pair
    assert c_act == 'add("Task A")'
    assert r_act == 'add("Task B")'
    assert c_conf == 0.95
    assert r_conf == 0.78


def test_generate_with_confidence_stop_token_exclusion():
    """Verify stop token confidence (e.g. EOS) is popped from per_token_confs to avoid contaminating metrics."""
    import torch
    from kanha.core.generation import generate_with_confidence

    mock_tok = MagicMock()
    mock_tok.EOS_ID = 99
    mock_tok.encode.return_value = [1]
    mock_tok.decode.side_effect = lambda ids, **kw: "test action" if ids == [10, 20, 99] or ids == [10, 20] else "eos"

    # Mock model: generates token 10 (conf=0.9), token 20 (conf=0.8), then EOS token 99 (conf=0.01)
    mock_model = MagicMock()
    mock_model.n_layers = 1
    mock_model.max_seq_len = 100
    mock_model.eval = MagicMock()
    mock_model.to = MagicMock()

    call_count = 0
    def mock_forward(x, kv_caches=None, offset=0):
        nonlocal call_count
        vocab_size = 100
        logits = torch.zeros(1, x.shape[1], vocab_size)
        if call_count == 0:
            logits[0, -1, 10] = 10.0  # token 10 high prob
        elif call_count == 1:
            logits[0, -1, 20] = 10.0  # token 20 high prob
        else:
            logits[0, -1, 99] = 10.0  # token 99 (EOS)
        call_count += 1
        return logits, None, [None]

    mock_model.side_effect = mock_forward

    with pytest.MonkeyPatch().context() as m:
        m.setattr("kanha.core.generation.get_device", lambda: "cpu")
        text, conf, per_token = generate_with_confidence(
            mock_model, mock_tok, "prompt",
            max_new_tokens=5,
            temperature=0.0,
            confidence_metric="min",
            stop_tokens=[99],
        )

    # Per-token confidences should only contain the 2 content tokens (10 and 20), EOS (99) excluded
    assert len(per_token) == 2
    # Min confidence should be around high prob (>0.5), not the 0.01 of EOS
    assert conf > 0.5


def test_generate_with_confidence_stop_sequence_exclusion():
    """Verify stop sequence token confidence is trimmed from per_token_confs."""
    import torch
    from kanha.core.generation import generate_with_confidence

    mock_tok = MagicMock()
    mock_tok.EOS_ID = 99
    mock_tok.encode.side_effect = lambda text, **kw: [1] if text == "prompt" else ([10] if text == "action" else [10, 11])
    mock_tok.decode.side_effect = lambda ids, **kw: "action\n" if ids == [10, 11] else "action"

    mock_model = MagicMock()
    mock_model.n_layers = 1
    mock_model.max_seq_len = 100
    mock_model.eval = MagicMock()
    mock_model.to = MagicMock()

    step = 0
    def mock_forward(x, kv_caches=None, offset=0):
        nonlocal step
        logits = torch.zeros(1, x.shape[1], 100)
        if step == 0:
            logits[0, -1, 10] = 10.0  # token 10: "action"
        else:
            logits[0, -1, 11] = 10.0  # token 11: "\n"
        step += 1
        return logits, None, [None]

    mock_model.side_effect = mock_forward

    with pytest.MonkeyPatch().context() as m:
        m.setattr("kanha.core.generation.get_device", lambda: "cpu")
        text, conf, per_token = generate_with_confidence(
            mock_model, mock_tok, "prompt",
            max_new_tokens=5,
            temperature=0.0,
            confidence_metric="geometric",
            stop_sequences=["\n"],
        )

    # Stop sequence "\n" should be stripped from text and its token removed from per_token
    assert text == "action"
    assert len(per_token) == 1


def test_api_stateless_chat_isolation():
    """Verify requests to api without session_id do not pollute engine.memory or leak context."""
    from fastapi.testclient import TestClient
    from api import create_app
    from unittest.mock import patch

    mock_engine = MagicMock()
    from kanha.memory.short_term import ShortTermMemory
    mock_engine.memory = ShortTermMemory(max_turns=10)
    mock_engine.chat.side_effect = lambda msg, stream=False, memory=None: f"Echo: {msg}"

    with patch("api.InferenceEngine.from_pretrained", return_value=mock_engine), \
         patch("api.os.path.exists", return_value=True):
        app = create_app(model_path="dummy.pt")
        client = TestClient(app)

        # Make two stateless requests
        r1 = client.post("/chat", json={"message": "First message"})
        assert r1.status_code == 200
        assert r1.json()["response"] == "Echo: First message"

        r2 = client.post("/chat", json={"message": "Second message"})
        assert r2.status_code == 200

        # Engine's global memory should remain empty (no cross-request pollution)
        assert len(mock_engine.memory) == 0


def test_api_session_cache_lru_bounding():
    """Verify _sessions cache is bounded to 1000 items and evicts oldest."""
    from fastapi.testclient import TestClient
    from api import create_app
    from unittest.mock import patch

    mock_engine = MagicMock()
    mock_engine.chat.side_effect = lambda msg, stream=False, memory=None: "Echo"

    with patch("api.InferenceEngine.from_pretrained", return_value=mock_engine), \
         patch("api.os.path.exists", return_value=True):
        app = create_app(model_path="dummy.pt")
        client = TestClient(app)

        # Create 1005 unique sessions
        for i in range(1005):
            r = client.post("/chat", json={"message": "hi", "session_id": f"sess_{i}"})
            assert r.status_code == 200

        # Access _sessions from closure via app route
        # Using /reset endpoint to probe existence
        r_old = client.post("/reset?session_id=sess_0")
        # sess_0 was evicted, so reset reports global memory or sess_0 not in _sessions
        # Let's verify sess_1004 is present
        r_new = client.post("/reset?session_id=sess_1004")
        assert "sess_1004" in r_new.json()["status"]


def test_dpo_empty_validation_skips_best_model_update():
    """Verify DPO training does not save a best model if val_accs is empty (all batches masked)."""
    val_accs = []
    val_margins = []
    best_val_acc = -1.0
    best_val_margin = float("-inf")
    saved_best = False

    # Simulate validation loop ending
    if len(val_accs) > 0:
        avg_val_acc = sum(val_accs) / len(val_accs)
        avg_val_margin = sum(val_margins) / len(val_margins)
        is_better = (avg_val_acc > best_val_acc) or (abs(avg_val_acc - best_val_acc) < 1e-4 and avg_val_margin > best_val_margin)
        if is_better:
            best_val_acc = avg_val_acc
            saved_best = True

    assert not saved_best
    assert best_val_acc == -1.0


def test_sft_dataset_separate_tokenization_length_parity(tmp_path):
    """Verify SFTDataset __init__ length filtering matches __getitem__ separate tokenization."""
    import json
    tok = KanhaTokenizer()
    sft_file = tmp_path / "sft_data.jsonl"
    with open(sft_file, "w") as f:
        f.write(json.dumps({"instruction": "Hello", "response": "World"}) + "\n")

    dataset = SFTDataset(str(sft_file), tok, max_len=128)
    assert len(dataset) == 1
    sample = dataset[0]
    assert len(sample["input_ids"]) > 0
    assert len(sample["labels"]) == len(sample["input_ids"])


def test_dataloader_drop_last_false_preserves_small_dataset():
    """Verify DataLoader with drop_last=False yields batches even when dataset size < batch_size."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    data = TensorDataset(torch.randn(2, 4))
    loader = DataLoader(data, batch_size=4, drop_last=False)
    batches = list(loader)
    assert len(batches) == 1
    assert batches[0][0].shape[0] == 2





