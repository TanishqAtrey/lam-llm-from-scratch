"""
kanha/memory/short_term.py
Short-term conversation memory using a bounded deque.
"""

from collections import deque
from typing import Optional


class ShortTermMemory:
    """
    Stores recent conversation turns as (role, content) pairs.
    Automatically drops oldest turns when max_turns is exceeded.
    """

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self._turns = deque(maxlen=max_turns)

    def add(self, role: str, content: str):
        """Add a turn to memory. role is 'user' or 'assistant'."""
        self._turns.append({"role": role, "content": content})

    def format(self) -> str:
        """Formats conversation history as a string for the prompt."""
        lines = []
        for turn in self._turns:
            if turn["role"] == "user":
                lines.append(f"User: {turn['content']}")
            else:
                lines.append(f"Assistant: {turn['content']}")
        return "\n".join(lines)

    def get_context(self, mode: str = "full") -> str:
        """Returns formatted context. mode='full' or 'summary'."""
        if not self._turns:
            return ""
        return self.format()

    def clear(self):
        """Clears all memory."""
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)

    def __repr__(self) -> str:
        return f"ShortTermMemory(turns={len(self)}, max={self.max_turns})"
