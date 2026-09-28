"""
kanha/lam/langchain/model.py
Custom LangChain LLM wrapper around the Kanha transformer model.

This makes KanhaModel usable within LangChain chains and LangGraph nodes.
The model generates raw text (action strings in our case), not tool-call JSON,
so we treat it as a plain BaseLLM and handle action parsing separately.
"""

from __future__ import annotations

from typing import Any, List, Optional, Mapping

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from pydantic import Field, PrivateAttr

from kanha.core.generation import generate as kanha_generate


class KanhaLLM(BaseLLM):
    """LangChain-compatible LLM that wraps the Kanha transformer.

    Usage::

        from kanha.core.model import KanhaModel
        from kanha.core.tokenizer import KanhaTokenizer

        model = KanhaModel.from_pretrained("models/lam/lam_final.pt")
        tokenizer = KanhaTokenizer()
        llm = KanhaLLM(model=model, tokenizer=tokenizer)

        output = llm.invoke("### Goal\\nadd milk\\n### Action\\n")
    """

    # Pydantic fields (public config)
    max_new_tokens: int = Field(default=64, description="Max tokens to generate")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    top_k: int = Field(default=1, description="Top-k sampling")
    top_p: float = Field(default=1.0, description="Nucleus sampling")
    repetition_penalty: float = Field(default=1.0, description="Repetition penalty")

    # Private attrs holding the actual model objects (not serialisable)
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, model: Any, tokenizer: Any, **kwargs: Any):
        """Create a KanhaLLM wrapper.

        Args:
            model: A loaded KanhaModel instance.
            tokenizer: A loaded KanhaTokenizer instance.
            **kwargs: Any BaseLLM / pydantic field overrides
                      (max_new_tokens, temperature, top_k, …).
        """
        super().__init__(**kwargs)
        self._model = model
        self._tokenizer = tokenizer

    # ── LangChain required interface ──────────────────────────────────────

    @property
    def _llm_type(self) -> str:
        return "kanha"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text for a single prompt.

        Delegates to ``kanha.core.generation.generate()`` which handles
        KV-cache, sampling, and EOS termination internally.
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        top_k = kwargs.get("top_k", self.top_k)

        raw = kanha_generate(
            self._model,
            self._tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
        )

        # Apply stop sequences if provided
        if stop:
            for seq in stop:
                idx = raw.find(seq)
                if idx != -1:
                    raw = raw[:idx]

        return raw

    def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
        """Batch generate — just loops _call since Kanha has no batch API."""
        from langchain_core.outputs import Generation, LLMResult

        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    # ── Convenience ───────────────────────────────────────────────────────

    def generate_action(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate and return a cleaned action string.

        This is a convenience method that strips trailing markers and
        takes only the first line — the same cleaning the base TaskAgent does.
        """
        raw = self._call(prompt, temperature=temperature)
        return self._clean_action(raw)

    @staticmethod
    def _clean_action(raw: str) -> str:
        """Clean model output to extract just the action call."""
        clean = raw.strip()
        if not clean:
            return ""
        first_line = clean.split("\n")[0].strip()

        # Remove trailing markers the model might have generated
        for marker in ["### Observation", "### Action", "### Goal"]:
            if marker in first_line:
                first_line = first_line.split(marker)[0].strip()

        return first_line
