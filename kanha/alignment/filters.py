"""
kanha/alignment/filters.py
Basic safety filters for model output.

Simple keyword-based filtering for known harmful patterns.
Not a replacement for RLHF/DPO — just a last-resort guardrail.
"""

from typing import Tuple

# Patterns that trigger safety filtering
UNSAFE_PATTERNS = [
    "how to make a bomb",
    "how to build a bomb",
    "how to kill",
    "how to murder",
    "step by step guide to harm",
    "how to synthesize drugs",
    "how to hack into",
    "how to steal",
    "suicide method",
]

REFUSAL = "I can't help with that request. Let me know if there's something else I can assist with."


def is_safe(text: str) -> Tuple[bool, str]:
    """
    Checks if text contains unsafe patterns.

    Returns:
        (is_safe, message) — if not safe, message contains the refusal
    """
    text_lower = text.lower()
    for pattern in UNSAFE_PATTERNS:
        if pattern in text_lower:
            return False, REFUSAL
    return True, ""


def filter_response(response: str) -> str:
    """
    Filters model response for safety.
    Returns the response as-is if safe, or a refusal message if not.
    """
    safe, msg = is_safe(response)
    if not safe:
        return msg
    return response
