"""
kanha/inference/engine.py
Inference engine — the central hub that connects model, tokenizer,
prompt builder, memory, RAG, and tools.

This is what cli.py and api.py call to generate responses.

CRITICAL FIX: The engine wraps user input in the SAME prompt template
that was used during SFT training. Without this, the SFT model sees
out-of-distribution input and produces garbage (::::????).
"""

import os
from typing import Optional

from kanha.core.model import KanhaModel
from kanha.core.tokenizer import KanhaTokenizer
from kanha.core.generation import generate, generate_stream
from kanha.prompting.builder import PromptBuilder
from kanha.memory.short_term import ShortTermMemory
from kanha.utils.config import cfg
from kanha.utils.helpers import get_device
from kanha.utils.logging import get_logger

log = get_logger("engine")


class InferenceEngine:
    """
    High-level inference engine for KANHA.

    Handles:
      - Model + tokenizer loading
      - Prompt formatting (using the same template as SFT training)
      - Conversation memory
      - RAG retrieval (optional)
      - Tool routing (optional)
      - Text generation (normal + streaming)
    """

    def __init__(
        self,
        model: KanhaModel,
        tokenizer: KanhaTokenizer,
        use_rag: bool = False,
        use_tools: bool = False,
        retriever=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = get_device()

        # Prompt builder — MUST use include_rules=False to match SFT training.
        # SFT training uses PromptBuilder(include_rules=False), so inference
        # must do the same. Adding system rules that weren't in training data
        # causes garbage output for small models (40-60M).
        self.prompt_builder = PromptBuilder(include_rules=False)

        # Conversation memory
        self.memory = ShortTermMemory(max_turns=cfg.memory.short_term_limit)

        # RAG retriever (optional)
        self.use_rag = use_rag
        self.retriever = retriever

        # Tool router (optional)
        self.use_tools = use_tools
        self.tool_router = None
        if use_tools:
            from kanha.tools.router import ToolRouter
            self.tool_router = ToolRouter()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        tokenizer_path: str = None,
        index_dir: str = None,
        use_rag: bool = False,
        use_tools: bool = False,
    ) -> "InferenceEngine":
        """
        Factory method — loads model, tokenizer, and optional RAG index.
        """
        # Load model
        model = KanhaModel.from_pretrained(model_path)

        # Load tokenizer
        tokenizer = KanhaTokenizer(tokenizer_path)

        # Load RAG retriever if index provided
        retriever = None
        if use_rag and index_dir and os.path.isdir(index_dir):
            try:
                from kanha.rag.retriever import Retriever
                retriever = Retriever()
                retriever.load_index(index_dir)
                log.info(f"RAG index loaded from {index_dir}")
            except Exception as e:
                log.warning(f"Failed to load RAG index: {e}")
                use_rag = False

        engine = cls(
            model=model,
            tokenizer=tokenizer,
            use_rag=use_rag,
            use_tools=use_tools,
            retriever=retriever,
        )

        log.info("InferenceEngine ready")
        return engine

    def chat(self, user_message: str, stream: bool = False) -> str:
        """
        Main chat method — takes a user message and returns the model's response.

        Steps:
          1. (Optional) Retrieve RAG chunks
          2. Build prompt using the SAME template as SFT training
          3. Generate response
          4. (Optional) Execute tool calls
          5. (Optional) Filter response
          6. Update memory

        Args:
            user_message : the user's raw text input
            stream       : if True, prints tokens as they're generated

        Returns:
            The model's response string
        """
        # 1. RAG retrieval
        retrieved_chunks = None
        if self.use_rag and self.retriever:
            try:
                results = self.retriever.retrieve(user_message, top_k=cfg.rag.top_k)
                retrieved_chunks = [r["text"] for r in results]
            except Exception as e:
                log.warning(f"RAG retrieval failed: {e}")

        # 2. Build prompt with the SFT template
        # NOTE: We do NOT include memory_context in the prompt because the SFT
        # training data only had single-turn (instruction, response) pairs.
        # Injecting conversation history into the prompt for a 40-60M model
        # would be out-of-distribution and cause degraded output.
        # Multi-turn requires training with conversation history in the data.
        tool_desc = ""
        if self.use_tools and self.tool_router:
            tool_desc = self.tool_router.format_tool_descriptions()

        prompt = self.prompt_builder.build(
            instruction=user_message,
            memory_context="",
            retrieved_chunks=retrieved_chunks,
            tool_descriptions=tool_desc,
        )

        # 3. Generate
        if stream:
            response_parts = []
            for token_text in generate_stream(
                self.model,
                self.tokenizer,
                prompt,
                max_new_tokens=cfg.inference.max_new_tokens,
                temperature=cfg.inference.temperature,
                top_k=cfg.inference.top_k,
                top_p=cfg.inference.top_p,
                device=self.device,
            ):
                print(token_text, end="", flush=True)
                response_parts.append(token_text)
            print()  # newline after streaming
            response = "".join(response_parts)
        else:
            response = generate(
                self.model,
                self.tokenizer,
                prompt,
                max_new_tokens=cfg.inference.max_new_tokens,
                temperature=cfg.inference.temperature,
                top_k=cfg.inference.top_k,
                top_p=cfg.inference.top_p,
                repetition_penalty=cfg.inference.repetition_penalty,
                device=self.device,
            )

        # 4. Tool execution
        if self.use_tools and self.tool_router:
            response = self.tool_router.maybe_execute(response)

        # 5. Safety filter
        try:
            from kanha.alignment.filters import filter_response
            response = filter_response(response)
        except ImportError:
            pass

        # 6. Update memory
        self.memory.add("user", user_message)
        self.memory.add("assistant", response)

        return response

    def reset_memory(self):
        """Clears conversation memory."""
        self.memory.clear()
        log.info("Memory cleared")
