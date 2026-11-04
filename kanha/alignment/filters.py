"""
kanha/alignment/filters.py
Safety filters for model input and output.

Keyword and pattern-based filtering for known harmful patterns.
Not a replacement for RLHF/DPO — just a reliable first/last-resort guardrail.
"""

import re
from typing import Tuple

# Patterns that trigger safety filtering on prompt inputs
UNSAFE_PATTERNS = [
    r"\bhow to (?:make|build|construct|create) a bomb\b",
    r"\bhow to (?:kill|murder|harm) (?:someone|a person|people|myself|somebody|a human)\b",
    r"\bstep by step guide to harm\b",
    r"\bhow to synthesize (?:drugs|meth|fentanyl|explosives)\b",
    r"\bhow to hack into\b",
    r"\bhow to steal (?:credit card|identity|passwords?)\b",
    r"\bsuicide method\b",
]

REFUSAL = "I can't help with that request. Let me know if there's something else I can assist with."


def is_safe(text: str) -> Tuple[bool, str]:
    """
    Checks if text contains unsafe patterns.
    Uses regex word boundaries to prevent false positives on harmless
    phrases like 'kill a process' or 'kill a task'.

    Returns:
        (is_safe, message) — if not safe, message contains the refusal
    """
    text_lower = text.lower()
    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, text_lower):
            return False, REFUSAL
    return True, ""


def is_safe_prompt(prompt: str) -> Tuple[bool, str]:
    """Validates user input before token generation begins."""
    return is_safe(prompt)


def filter_response(response: str) -> str:
    """
    Filters model response for safety.
    Returns the response as-is if safe, or a refusal message if not.
    """
    safe, msg = is_safe(response)
    if not safe:
        return msg
    return response
