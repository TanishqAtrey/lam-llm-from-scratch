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

# RAG-augmented version — canonical instruction prefix with embedded context
RAG_TEMPLATE = (
    "### Instruction:\n"
    "Retrieved Context:\n{context}\n\n"
    "{instruction}\n\n"
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

    def __init__(self, include_rules: bool = False):
        self.include_rules = include_rules

    def build(
        self,
        instruction: str,
        memory_context: str = "",
        retrieved_chunks: Optional[List[str]] = None,
        tool_descriptions: str = "",
        max_context_chars: int = None,
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
        import re
        # Idempotency check: if already formatted with canonical template markers, return directly
        if instruction.startswith("### Instruction:\n") or instruction.startswith("### Conversation History:\n"):
            return instruction

        # Normalise single-turn role prefixes (e.g. User: or Human:) and extract dialogue history if present
        if not memory_context and re.search(r'(?i)(?:^|\n)\s*(?:human|user|assistant)\s*:', instruction):
            history, inst = self.parse_dialogue(instruction)
            instruction = inst
            if history:
                memory_context = history

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
            # Scale max context to ~4 tokens/char × max_seq_len, leave room for response
            if max_context_chars is None:
                try:
                    from kanha.utils.config import cfg
                    # Use ~40% of max_seq_len for context (at ~4 chars/token)
                    max_context_chars = int(cfg.model.max_seq_len * 0.4 * 4)
                except Exception:
                    max_context_chars = 2000
            context = self._truncate_context(retrieved_chunks, max_context_chars)
            parts.append(RAG_TEMPLATE.format(
                context=context,
                instruction=instruction,
            ))
        else:
            parts.append(INSTRUCTION_TEMPLATE.format(instruction=instruction))

        return "".join(parts)

    def build_prompt_prefix(self, instruction: str, memory_context: str = "") -> tuple[str, str]:
        """
        Returns (prompt_prefix, cleaned_instruction) with consistent dialogue/role parsing.
        """
        import re
        if not memory_context and re.search(r'(?i)(?:^|\n)\s*(?:human|user|assistant)\s*:', instruction):
            history, inst = self.parse_dialogue(instruction)
            instruction = inst
            if history:
                memory_context = history
        prefix = self.build(instruction=instruction, memory_context=memory_context)
        return prefix, instruction

    def build_training_pair(self, instruction: str, response: str, memory_context: str = "") -> str:
        """
        Builds the full training sequence (instruction + response).
        If memory_context is provided (or if instruction contains multi-turn dialogue),
        it formats with '### Conversation History:' to ensure 100% inference template match.
        Used by sft_train.py and dpo_train.py to ensure template consistency.
        """
        prefix, _ = self.build_prompt_prefix(instruction, memory_context=memory_context)
        return prefix + response

    @staticmethod
    def parse_dialogue(text: str) -> tuple[str, str]:
        """
        Parses multi-turn dialogue into (memory_context, last_instruction).
        Handles formats like:
          Human: ... \n\nAssistant: ... \n\nHuman: ...
          User: ... \n\nAssistant: ... \n\nUser: ...
        If single turn or no dialogue markers, returns ('', clean_text).
        If text starts with a non-role preamble (e.g. "Summarize this conversation:\nUser: ..."),
        it does not destroy the preamble and treats the text as a task instruction.
        """
        import re
        text = text.strip()

        # Only reparse as multi-turn dialogue if the text actually begins with a role marker.
        # This prevents dropping preambles like "Summarize the following conversation:\nUser: ..."
        is_role_start = bool(re.match(r'^(?:human|user|assistant)\s*:', text, re.IGNORECASE))
        if not is_role_start:
            return "", text

        pattern = re.compile(r'(?i)(?:^|\n)\s*(human|user|assistant)\s*:\s*', re.MULTILINE)
        matches = list(pattern.finditer(text))
        if not matches:
            clean = text
            for prefix in ("human:", "user:", "assistant:"):
                if clean.lower().startswith(prefix):
                    clean = clean[len(prefix):].strip()
            return "", clean

        roles = [m.group(1).lower() for m in matches]
        has_assistant = any(r == "assistant" for r in roles)
        has_user = any(r in ("human", "user") for r in roles)
        if not (has_assistant and has_user):
            clean = text
            for prefix in ("human:", "user:", "assistant:"):
                if clean.lower().startswith(prefix):
                    clean = clean[len(prefix):].strip()
            return "", clean

        pairs = []
        for i, match in enumerate(matches):
            role_raw = match.group(1).lower()
            role = "Assistant" if role_raw == "assistant" else "User"
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start_pos:end_pos].strip()
            pairs.append((role, content))

        if not pairs:
            return "", text

        # Find the last User turn for instruction; all prior and trailing turns become history
        user_indices = [idx for idx, p in enumerate(pairs) if p[0] == "User"]
        if user_indices:
            last_u_idx = user_indices[-1]
            last_inst = pairs[last_u_idx][1]
            history_pairs = pairs[:last_u_idx] + pairs[last_u_idx + 1:]
            history_str = "\n".join(f"{role}: {content}" for role, content in history_pairs)
            return history_str, last_inst

        return "", text

    @staticmethod
    def _truncate_context(chunks: List[str], max_chars: int) -> str:
        """Concatenates chunks up to max_chars limit."""
        if not chunks or max_chars <= 0:
            return ""
        result = []
        total = 0
        sep = "\n---\n"
        for chunk in chunks:
            needed = (len(sep) if result else 0) + len(chunk)
            if total + needed <= max_chars:
                result.append(chunk)
                total += needed
            else:
                remaining = max_chars - total - (len(sep) if result else 0)
                if remaining > 0:
                    result.append(chunk[:remaining])
                break
        return sep.join(result)


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

