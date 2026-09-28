"""
kanha/prompting/builder.py
Prompt template builder for KANHA.

CRITICAL: The template used here MUST match what was used during SFT training.
If the training formatted data as:
    ### Instruction:
    {instruction}

    ### Response:
    {response}

Then inference MUST wrap the user query in the same format.
A mismatch here is the #1 cause of garbage output after fine-tuning.
"""

from typing import List, Optional


# ── The canonical prompt template ─────────────────────────────────────────────
# This MUST be identical to what sft_train.py uses.
INSTRUCTION_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"
FULL_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{response}"

# RAG-augmented version
RAG_TEMPLATE = (
    "### Retrieved Context:\n{context}\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)

# System rules (optional, prepended when include_rules=True)
SYSTEM_RULES = (
    "You are KANHA, a helpful and harmless AI assistant. "
    "Answer the user's question accurately and concisely. "
    "If you don't know the answer, say so honestly.\n\n"
)

# Tool-aware system prompt
TOOL_RULES = (
    "You have access to the following tools:\n"
    "{tool_descriptions}\n\n"
    "To use a tool, write: TOOL: tool_name(arguments)\n"
    "Only use a tool if it helps answer the question.\n\n"
)


class PromptBuilder:
    """
    Builds prompts for inference that match the SFT training format.

    Usage:
        builder = PromptBuilder()
        prompt = builder.build(instruction="What is Python?")
        # Returns: "### Instruction:\\nWhat is Python?\\n\\n### Response:\\n"
    """

    def __init__(self, include_rules: bool = True):
        self.include_rules = include_rules

    def build(
        self,
        instruction: str,
        memory_context: str = "",
        retrieved_chunks: Optional[List[str]] = None,
        tool_descriptions: str = "",
        max_context_chars: int = 2000,
    ) -> str:
        """
        Builds the full prompt string for the model.

        Args:
            instruction       : the user's query
            memory_context    : formatted conversation history
            retrieved_chunks  : RAG chunks (if available)
            tool_descriptions : formatted tool descriptions
            max_context_chars : max chars for RAG context

        Returns:
            Formatted prompt string ready for tokenization
        """
        parts = []

        # System rules
        if self.include_rules:
            parts.append(SYSTEM_RULES)

        # Tool descriptions
        if tool_descriptions:
            parts.append(TOOL_RULES.format(tool_descriptions=tool_descriptions))

        # Conversation history
        if memory_context:
            parts.append(f"### Conversation History:\n{memory_context}\n\n")

        # RAG context
        if retrieved_chunks:
            context = self._truncate_context(retrieved_chunks, max_context_chars)
            parts.append(RAG_TEMPLATE.format(
                context=context,
                instruction=instruction,
            ))
        else:
            parts.append(INSTRUCTION_TEMPLATE.format(instruction=instruction))

        return "".join(parts)

    def build_training_pair(self, instruction: str, response: str) -> str:
        """
        Builds the full training sequence (instruction + response).
        Used by sft_train.py to ensure template consistency.
        """
        return FULL_TEMPLATE.format(instruction=instruction, response=response)

    @staticmethod
    def _truncate_context(chunks: List[str], max_chars: int) -> str:
        """Concatenates chunks up to max_chars limit."""
        result = []
        total = 0
        for chunk in chunks:
            if total + len(chunk) > max_chars:
                break
            result.append(chunk)
            total += len(chunk)
        return "\n---\n".join(result)


# ── LAM (Language Action Model) prompt helpers ────────────────────────────────
# Re-exported from trajectory.py so that existing imports keep working.
# The canonical implementation lives in kanha.lam.trajectory to avoid
# circular imports and to keep a SINGLE source of truth for the template.

from kanha.lam.trajectory import (
    GOAL_MARKER as LAM_GOAL_MARKER,
    ACTION_MARKER as LAM_ACTION_MARKER,
    OBSERVATION_MARKER as LAM_OBSERVATION_MARKER,
    format_step_prompt as build_lam_step_prompt,
)


def build_lam_training_pair(goal: str, history: list, next_action: str) -> str:
    """Builds a complete training example: prompt + the next action as the response.

    The boundary between prompt (masked) and response (trained) is the
    last '### Action\\n' marker.
    """
    prompt = build_lam_step_prompt(goal, history)
    return prompt + next_action

