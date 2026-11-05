"""
tests/test_sft_pipeline_simulation.py
Simulates the entire end-to-end SFT fine-tuning and inference pipeline
using the real pretrained base checkpoint: models/base/final_model.pt.

Verifies:
  1. Base model loading from models/base/final_model.pt
  2. Dataset formatting, tokenization, and causal LM loss masking
  3. Dynamic batch collation (sft_collate_fn)
  4. SFT training step with optimizer, gradient accumulation, and scheduler
  5. Checkpoint saving in standard KanhaModel format
  6. Post-SFT inference loading, prompt prefix matching, and generation
"""

import os
import sys
import json
import torch
import pytest
from types import SimpleNamespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kanha.core.model import KanhaModel
from kanha.core.tokenizer import KanhaTokenizer
from kanha.core.generation import generate
from kanha.prompting.builder import PromptBuilder
from kanha.finetune.sft_train import SFTDataset, sft_collate_fn, sft_train
from kanha.inference.engine import InferenceEngine


@pytest.fixture(scope="module")
def base_model_path():
    path = REPO_ROOT / "models" / "base" / "final_model.pt"
    if not path.exists():
        pytest.skip(f"Base model checkpoint not found at {path}")
    return str(path)


@pytest.fixture(scope="module")
def tokenizer():
    return KanhaTokenizer()


def test_end_to_end_sft_pipeline_with_final_model(base_model_path, tokenizer, tmp_path):
    # ── 1. Create a diverse synthetic mini-dataset ───────────────────────────
    train_file = tmp_path / "sft_sim_train.jsonl"
    val_file = tmp_path / "sft_sim_val.jsonl"
    out_dir = tmp_path / "sft_out"

    train_data = [
        # Single-turn chat
        {
            "instruction": "What is Python?",
            "response": "Python is a high-level, interpreted programming language.",
            "source": "chat"
        },
        # Multi-turn conversation
        {
            "instruction": "User: Hello!\nAssistant: Hi there! How can I help you today?\nUser: Can you write a function to square a number?",
            "response": "Here is a Python function:\n```python\ndef square(n):\n    return n * n\n```",
            "source": "dialogue"
        },
        # Alpaca code instruction with input
        {
            "instruction": "Sort the given list of numbers in ascending order.\n\nInput:\n[5, 2, 8, 1]",
            "response": "```python\nnumbers = [5, 2, 8, 1]\nnumbers.sort()\nprint(numbers)  # [1, 2, 5, 8]\n```",
            "source": "code"
        },
        # Identity sample
        {
            "instruction": "Who created you?",
            "response": "I was created and trained from scratch by Tanishq Atrey.",
            "source": "identity"
        },
    ]

    val_data = [
        {
            "instruction": "What does KANHA stand for?",
            "response": "KANHA stands for Knowledge-Augmented Neural Heuristic Assistant.",
            "source": "identity-val"
        }
    ]

    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")

    # ── 2. Validate SFTDataset and Loss Masking ──────────────────────────────
    dataset = SFTDataset(str(train_file), tokenizer, max_len=1024)
    assert len(dataset) == len(train_data)

    for i in range(len(dataset)):
        sample = dataset[i]
        inp_ids = sample["input_ids"]
        labels = sample["labels"]

        assert len(inp_ids) == len(labels)
        # Verify loss masking: instruction tokens must be -100
        assert (labels == -100).any(), f"Sample {i} has no masked prompt tokens!"
        # Verify supervision: response tokens must NOT be -100
        assert (labels != -100).any(), f"Sample {i} has no supervised response tokens!"
        # Last token of labels should be EOS_ID
        assert labels[-1].item() == tokenizer.EOS_ID

    # ── 3. Validate Collation with Dynamic Padding ───────────────────────────
    batch = [dataset[0], dataset[1]]
    collated = sft_collate_fn(batch, pad_id=tokenizer.PAD_ID)
    assert collated["input_ids"].shape[0] == 2
    assert collated["labels"].shape[0] == 2
    max_len = max(len(dataset[0]["input_ids"]), len(dataset[1]["input_ids"]))
    assert collated["input_ids"].shape[1] == max_len

    # ── 4. Run SFT Training Simulation ───────────────────────────────────────
    args = SimpleNamespace(
        base_model=base_model_path,
        data=str(train_file),
        val_data=str(val_file),
        output=str(out_dir),
        epochs=1,
        batch_size=2,
        grad_accum_steps=1,
        lr=3e-5,
        val_fraction=0.0,
        patience=3,
        lora=False,
    )

    sft_train(args)

    final_model_ckpt = out_dir / "sft_final.pt"
    assert final_model_ckpt.exists(), "sft_final.pt was not saved!"
    assert final_model_ckpt.stat().st_size > 10_000_000, "Checkpoint is unusually small!"

    # ── 5. Verify Inference Compatibility & Template Match ───────────────────
    engine = InferenceEngine.from_pretrained(
        model_path=str(final_model_ckpt),
        tokenizer_path=None,
    )

    # Prompt builder consistency
    builder = PromptBuilder(include_rules=False)
    q1 = "Who created you?"
    p1 = builder.build(instruction=q1)
    assert p1 == "### Instruction:\nWho created you?\n\n### Response:\n"

    # Multi-turn dialogue consistency
    q2 = "User: Hello\nAssistant: Hi\nUser: Who are you?"
    p2 = builder.build(instruction=q2)
    assert p2 == "### Conversation History:\nUser: Hello\nAssistant: Hi\n\n### Instruction:\nWho are you?\n\n### Response:\n"

    # Generation check (runs forward pass with KV cache without crashing)
    resp = generate(
        engine.model,
        engine.tokenizer,
        p1,
        max_new_tokens=25,
        temperature=0.0,
        device=engine.device,
    )
    assert isinstance(resp, str)
    assert len(resp) > 0
