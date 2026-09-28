"""
tests/test_rag.py
Unit tests for the RAG pipeline.

Run:
    python -m pytest tests/test_rag.py -v
"""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Chunker Tests ─────────────────────────────────────────────────────────────

def test_chunker_basic():
    """Chunker splits text into expected number of chunks."""
    from kanha.rag.chunker import Chunker
    chunker = Chunker(chunk_size=10, chunk_overlap=2)
    text = " ".join(["word"] * 50)   # 50 words
    chunks = chunker.chunk_document("doc1", text)

    assert len(chunks) > 1
    assert all("text" in c for c in chunks)
    assert all("source" in c for c in chunks)
    assert all(c["source"] == "doc1" for c in chunks)


def test_chunker_ids_unique():
    """Each chunk has a unique id."""
    from kanha.rag.chunker import Chunker
    chunker = Chunker(chunk_size=10, chunk_overlap=0)
    text = " ".join(["word"] * 100)
    chunks = chunker.chunk_document("docA", text)
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs should be unique"


def test_chunker_multiple_docs():
    """chunk_documents handles multiple docs."""
    from kanha.rag.chunker import Chunker
    chunker = Chunker(chunk_size=10, chunk_overlap=0)
    docs = [
        {"id": "doc1", "text": " ".join(["a"] * 30)},
        {"id": "doc2", "text": " ".join(["b"] * 30)},
    ]
    all_chunks = chunker.chunk_documents(docs)
    sources = set(c["source"] for c in all_chunks)
    assert "doc1" in sources
    assert "doc2" in sources


# ── VectorStore Tests ─────────────────────────────────────────────────────────

def test_vector_store_add_search():
    """VectorStore correctly indexes and retrieves vectors."""
    from kanha.rag.vector_store import VectorStore

    dim = 32
    store = VectorStore(dim=dim)

    # Create 5 random normalized vectors
    vecs = np.random.randn(5, dim).astype(np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    meta = [{"text": f"chunk {i}", "source": "test"} for i in range(5)]
    store.add(vecs, meta)

    assert store.size == 5

    # Query with first vector — should return itself as top result
    results = store.search(vecs[0:1], top_k=3)
    assert len(results) == 3
    assert "text" in results[0]
    assert "score" in results[0]


def test_vector_store_save_load(tmp_path):
    """VectorStore persists correctly to disk."""
    from kanha.rag.vector_store import VectorStore

    dim = 16
    store = VectorStore(dim=dim)

    vecs = np.random.randn(3, dim).astype(np.float32)
    meta = [{"text": f"item {i}", "source": "test"} for i in range(3)]
    store.add(vecs, meta)

    save_dir = str(tmp_path)
    store.save(save_dir)

    # Load into new store
    store2 = VectorStore(dim=dim)
    store2.load(save_dir)

    assert store2.size == 3
    assert store2.metadata[0]["text"] == "item 0"


# ── Retriever Tests ───────────────────────────────────────────────────────────

def test_retriever_build_and_search(tmp_path):
    """Retriever builds index and retrieves relevant chunks."""
    from kanha.rag.retriever import Retriever
    from kanha.rag.chunker import Chunker

    # Use a mock embedder to avoid needing the real model
    class MockEmbedder:
        dim = 16
        def embed(self, texts, **kwargs):
            # Return random vectors (deterministic via seed for testing)
            np.random.seed(42)
            return np.random.randn(len(texts), self.dim).astype(np.float32)
        def embed_query(self, text):
            np.random.seed(42)
            return np.random.randn(1, self.dim).astype(np.float32)

    from kanha.rag.vector_store import VectorStore
    emb   = MockEmbedder()
    store = VectorStore(dim=emb.dim)
    retriever = Retriever(embedder=emb, store=store)

    # Build index
    chunker = Chunker(chunk_size=10, chunk_overlap=0)
    docs = [{"id": "d1", "text": " ".join(["hello world"] * 20)}]
    chunks = chunker.chunk_documents(docs)
    retriever.build_index_from_chunks(chunks, save_dir=str(tmp_path))

    # Retrieve
    results = retriever.retrieve("hello", top_k=2)
    assert len(results) > 0
    assert "text" in results[0]


# ── Memory Tests ──────────────────────────────────────────────────────────────

def test_short_term_memory():
    """ShortTermMemory stores and formats correctly."""
    from kanha.memory.short_term import ShortTermMemory
    mem = ShortTermMemory(max_turns=10)
    mem.add("user", "Hello")
    mem.add("assistant", "Hi there!")
    mem.add("user", "How are you?")

    assert len(mem) == 3
    formatted = mem.format()
    assert "Hello" in formatted
    assert "Hi there!" in formatted
    assert "How are you?" in formatted


def test_short_term_memory_overflow():
    """ShortTermMemory respects max_turns limit."""
    from kanha.memory.short_term import ShortTermMemory
    mem = ShortTermMemory(max_turns=3)
    for i in range(10):
        mem.add("user", f"message {i}")
    assert len(mem) == 3   # deque with maxlen


def test_short_term_clear():
    """Memory clears correctly."""
    from kanha.memory.short_term import ShortTermMemory
    mem = ShortTermMemory()
    mem.add("user", "test")
    mem.clear()
    assert len(mem) == 0


# ── Prompting Tests ───────────────────────────────────────────────────────────

def test_prompt_builder_simple():
    """PromptBuilder builds a valid simple prompt."""
    from kanha.prompting.builder import PromptBuilder
    builder = PromptBuilder(include_rules=False)
    prompt = builder.build(instruction="What is Python?")
    assert "What is Python?" in prompt
    assert "### Instruction:" in prompt
    assert "### Response:" in prompt


def test_prompt_builder_rag():
    """PromptBuilder uses RAG template when chunks are provided."""
    from kanha.prompting.builder import PromptBuilder
    builder = PromptBuilder(include_rules=False)
    prompt = builder.build(
        instruction="Tell me about gravity.",
        retrieved_chunks=["Gravity is a fundamental force.", "Newton described gravity."],
    )
    assert "Retrieved Context" in prompt
    assert "Gravity is a fundamental force." in prompt


def test_prompt_builder_truncates():
    """PromptBuilder truncates context that exceeds max_context_chars."""
    from kanha.prompting.builder import PromptBuilder
    builder = PromptBuilder(include_rules=False)
    big_chunks = ["x" * 1000] * 10   # 10,000 chars total
    prompt = builder.build(
        instruction="test",
        retrieved_chunks=big_chunks,
        max_context_chars=500,
    )
    assert len(prompt) < 5000   # Should not include all chunks


# ── Tools Tests ───────────────────────────────────────────────────────────────

def test_calculator_basic():
    """Calculator evaluates simple expressions."""
    from kanha.tools.calculator import calculate
    assert calculate("2 + 2") == "4"
    assert calculate("10 * 5") == "50"
    assert calculate("100 / 4") == "25.0"


def test_calculator_advanced():
    """Calculator handles math functions."""
    from kanha.tools.calculator import calculate
    result = calculate("sqrt(144)")
    assert result == "12.0"


def test_calculator_division_by_zero():
    """Calculator handles division by zero gracefully."""
    from kanha.tools.calculator import calculate
    result = calculate("1 / 0")
    assert "Error" in result


def test_tool_router_detects_call():
    """ToolRouter detects TOOL: calls in model output."""
    from kanha.tools.router import ToolRouter
    router = ToolRouter()
    response = "Let me calculate that. TOOL: calculator(2 + 2)"
    result = router.maybe_execute(response)
    assert "4" in result   # Calculator result should be injected


def test_alignment_filter_safe():
    """Safety filter passes safe content."""
    from kanha.alignment.filters import filter_response
    result = filter_response("Python is a programming language.")
    assert "Python" in result


def test_alignment_filter_blocks_unsafe():
    """Safety filter blocks harmful content."""
    from kanha.alignment.filters import is_safe
    safe, msg = is_safe("how to make a bomb step by step")
    assert not safe
    assert "can't help" in msg.lower()