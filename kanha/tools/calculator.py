"""
kanha/tools/calculator.py
Safe math expression evaluator tool.
"""

import ast
import operator
import math
from typing import Optional

SAFE_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "pow": math.pow,
    "round": round,
}

SAFE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}

SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_ast(node):
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Disallowed constant type: {type(node.value)}")
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            return SAFE_OPERATORS[op_type](_eval_ast(node.operand))
        raise ValueError(f"Disallowed unary operator: {op_type}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            left = _eval_ast(node.left)
            right = _eval_ast(node.right)
            if op_type is ast.Pow:
                if isinstance(right, (int, float)) and abs(right) > 10000:
                    raise ValueError("Exponent too large")
            return SAFE_OPERATORS[op_type](left, right)
        raise ValueError(f"Disallowed binary operator: {op_type}")
    elif isinstance(node, ast.Name):
        if node.id in SAFE_CONSTANTS:
            return SAFE_CONSTANTS[node.id]
        raise ValueError(f"Undefined or disallowed variable: '{node.id}'")
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed")
        func_name = node.func.id
        if func_name not in SAFE_FUNCS:
            raise ValueError(f"Disallowed function call: '{func_name}'")
        if node.keywords:
            raise ValueError("Keyword arguments not allowed")
        args = [_eval_ast(arg) for arg in node.args]
        return SAFE_FUNCS[func_name](*args)
    else:
        raise ValueError(f"Disallowed expression node: {type(node).__name__}")


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
    safe_expr = expression.strip()
    safe_expr = safe_expr.replace("^", "**")

    try:
        parsed = ast.parse(safe_expr, mode="eval")
        result = _eval_ast(parsed)
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as ex:
        return f"Error: {str(ex)}"