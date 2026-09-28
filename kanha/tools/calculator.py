"""
kanha/tools/calculator.py
Safe math expression evaluator tool.
"""

import re
import math
from typing import Optional


def calculate(expression: str) -> str:
    """
    Safely evaluates a math expression string.

    Examples:
        calculate("2 + 2")         → "4"
        calculate("sqrt(144)")     → "12.0"
        calculate("sin(pi / 2)")   → "1.0"

    Args:
        expression : math expression string

    Returns:
        Result as string, or error message
    """
    # Whitelist: only allow safe math characters
    # FIX: the regex was compiled but never actually checked — enforce it now.
    allowed = re.compile(r"^[\d\s\+\-\*\/\(\)\.\,\^a-z_]+$")

    safe_expr = expression.strip()
    safe_expr = safe_expr.replace("^", "**")   # support ^ for exponent

    if not allowed.match(safe_expr):
        return "Error: Expression contains disallowed characters."

    # Safe namespace with math functions only
    safe_globals = {
        "__builtins__": {},
        "sqrt": math.sqrt,
        "sin":  math.sin,
        "cos":  math.cos,
        "tan":  math.tan,
        "log":  math.log,
        "log2": math.log2,
        "log10":math.log10,
        "exp":  math.exp,
        "abs":  abs,
        "pi":   math.pi,
        "e":    math.e,
        "pow":  math.pow,
        "round":round,
    }

    try:
        result = eval(safe_expr, safe_globals)
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as ex:
        return f"Error: {str(ex)}"