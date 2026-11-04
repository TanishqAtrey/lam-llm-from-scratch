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

        # Load RAG retriever if requested
        retriever = None
        if use_rag:
            target_index = index_dir
            if not target_index and hasattr(cfg, "rag") and hasattr(cfg.rag, "index_path"):
                target_index = cfg.rag.index_path
            if target_index:
                resolved_index = os.path.dirname(target_index) if os.path.isfile(target_index) else target_index
                if os.path.isdir(resolved_index):
                    try:
                        from kanha.rag.retriever import Retriever
                        retriever = Retriever()
                        retriever.load_index(resolved_index)
                        log.info(f"RAG index loaded from {resolved_index}")
                    except Exception as e:
                        log.warning(f"Failed to load RAG index: {e}")
                        use_rag = False
                else:
                    log.warning(f"RAG index directory not found at {resolved_index}")
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

    def chat_stream(self, user_message: str):
        """
        Streaming generator that yields response tokens and tool outputs as they arrive.
        Updates conversation memory upon completion.
        """
        # 1. RAG retrieval
        retrieved_chunks = None
        if self.use_rag and self.retriever:
            try:
                results = self.retriever.retrieve(user_message, top_k=cfg.rag.top_k)
                retrieved_chunks = [r["text"] for r in results]
            except Exception as e:
                log.warning(f"RAG retrieval failed: {e}")

        # 2. Build prompt
        tool_desc = ""
        if self.use_tools and self.tool_router:
            tool_desc = self.tool_router.format_tool_descriptions()

        memory_ctx = self.memory.format() if len(self.memory) > 0 else ""
        prompt = self.prompt_builder.build(
            instruction=user_message,
            memory_context=memory_ctx,
            retrieved_chunks=retrieved_chunks,
            tool_descriptions=tool_desc,
        )

        # 3. Stream tokens from model (with ReAct loop if tools enabled)
        if self.use_tools and self.tool_router:
            from kanha.tools.router import find_tool_calls
            curr_prompt = prompt
            full_response_parts = []
            max_tool_iters = 3

            for _ in range(max_tool_iters):
                step_parts = []
                for token_text in generate_stream(
                    self.model,
                    self.tokenizer,
                    curr_prompt,
                    max_new_tokens=cfg.inference.max_new_tokens,
                    temperature=cfg.inference.temperature,
                    top_k=cfg.inference.top_k,
                    top_p=cfg.inference.top_p,
                    repetition_penalty=cfg.inference.repetition_penalty,
                    stop_sequences=["[Tool Result:", "\nObservation:"],
                    device=self.device,
                ):
                    step_parts.append(token_text)
                    yield token_text

                gen_text = "".join(step_parts)
                tool_calls = find_tool_calls(gen_text)
                if not tool_calls:
                    full_response_parts.append(gen_text)
                    break

                executed = self.tool_router.maybe_execute(gen_text)
                tool_output = executed[len(gen_text):]
                if tool_output:
                    yield tool_output
                full_response_parts.append(gen_text + tool_output)
                curr_prompt = curr_prompt + gen_text + tool_output + "\n"
            final_response = "".join(full_response_parts)
        else:
            response_parts = []
            for token_text in generate_stream(
                self.model,
                self.tokenizer,
                prompt,
                max_new_tokens=cfg.inference.max_new_tokens,
                temperature=cfg.inference.temperature,
                top_k=cfg.inference.top_k,
                top_p=cfg.inference.top_p,
                repetition_penalty=cfg.inference.repetition_penalty,
                device=self.device,
            ):
                response_parts.append(token_text)
                yield token_text

            final_response = "".join(response_parts)

        # 4. Safety filter
        try:
            from kanha.alignment.filters import filter_response
            final_response = filter_response(final_response)
        except ImportError:
            pass

        # 5. Update memory
        self.memory.add("user", user_message)
        self.memory.add("assistant", final_response)

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
        if stream:
            response_parts = []
            for chunk in self.chat_stream(user_message):
                print(chunk, end="", flush=True)
                response_parts.append(chunk)
            print()  # newline after streaming
            return "".join(response_parts)

        # 1. RAG retrieval
        retrieved_chunks = None
        if self.use_rag and self.retriever:
            try:
                results = self.retriever.retrieve(user_message, top_k=cfg.rag.top_k)
                retrieved_chunks = [r["text"] for r in results]
            except Exception as e:
                log.warning(f"RAG retrieval failed: {e}")

        # 2. Build prompt with the SFT template
        tool_desc = ""
        if self.use_tools and self.tool_router:
            tool_desc = self.tool_router.format_tool_descriptions()

        memory_ctx = self.memory.format() if len(self.memory) > 0 else ""
        prompt = self.prompt_builder.build(
            instruction=user_message,
            memory_context=memory_ctx,
            retrieved_chunks=retrieved_chunks,
            tool_descriptions=tool_desc,
        )

        # 3. Generate (with ReAct loop if tools enabled)
        if self.use_tools and self.tool_router:
            from kanha.tools.router import find_tool_calls
            curr_prompt = prompt
            full_response = ""
            max_tool_iters = 3

            for _ in range(max_tool_iters):
                gen_text = generate(
                    self.model,
                    self.tokenizer,
                    curr_prompt,
                    max_new_tokens=cfg.inference.max_new_tokens,
                    temperature=cfg.inference.temperature,
                    top_k=cfg.inference.top_k,
                    top_p=cfg.inference.top_p,
                    repetition_penalty=cfg.inference.repetition_penalty,
                    stop_sequences=["[Tool Result:", "\nObservation:"],
                    device=self.device,
                )
                tool_calls = find_tool_calls(gen_text)
                if not tool_calls:
                    full_response = full_response + gen_text if full_response else gen_text
                    break

                executed = self.tool_router.maybe_execute(gen_text)
                tool_output = executed[len(gen_text):]
                full_response = full_response + gen_text + tool_output
                curr_prompt = curr_prompt + gen_text + tool_output + "\n"
            else:
                if not full_response:
                    full_response = gen_text
            response = full_response
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

        # 4. Safety filter
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
