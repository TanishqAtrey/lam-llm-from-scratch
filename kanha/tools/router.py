"""
kanha/tools/router.py
Tool router — detects TOOL: calls in model output and executes them.

The model emits: TOOL: calculator(2 + 2)
The router intercepts, runs the tool, and appends the result.
"""

import re
from typing import Dict, Callable, Optional
from kanha.tools.calculator import calculate
from kanha.tools.search import web_search, format_search_results
from kanha.utils.logging import get_logger

log = get_logger(__name__)

# Pattern to detect tool call start in model output
TOOL_CALL_START_PATTERN = re.compile(r"TOOL:\s*([a-zA-Z0-9_]+)\s*\(", re.IGNORECASE)


def find_tool_calls(text: str) -> list[tuple[str, str]]:
    """
    Finds all TOOL: tool_name(args) invocations in text, correctly
    handling nested parentheses (e.g. calculator(sqrt(144))).
    """
    results = []
    pos = 0
    while True:
        match = TOOL_CALL_START_PATTERN.search(text, pos)
        if not match:
            break
        tool_name = match.group(1)
        start_idx = match.end()
        depth = 1
        i = start_idx
        in_str = None
        while i < len(text) and depth > 0:
            char = text[i]
            if in_str:
                if char == "\\":
                    i += 2
                    continue
                elif char == in_str:
                    in_str = None
            else:
                if char in ('"', "'"):
                    in_str = char
                elif char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        break
            i += 1
        if depth == 0:
            args = text[start_idx:i]
            results.append((tool_name, args))
            pos = i + 1
        else:
            pos = start_idx
    return results


class ToolRouter:
    """
    Manages available tools and routes model-generated tool calls.

    Adding a new tool:
        router.register("my_tool", my_function, "Description of tool")
    """

    def __init__(self):
        self._tools: Dict[str, Callable]     = {}
        self._descriptions: Dict[str, str]   = {}

        # Register built-in tools
        self.register(
            name="calculator",
            fn=calculate,
            description="Evaluates a math expression. Usage: TOOL: calculator(2 + 2 * 3)",
        )
        self.register(
            name="search",
            fn=lambda q: format_search_results(web_search(q)),
            description="Searches the web. Usage: TOOL: search(Python async tutorial)",
        )

    def register(self, name: str, fn: Callable, description: str):
        """Registers a new tool."""
        self._tools[name] = fn
        self._descriptions[name] = description
        log.info(f"Tool registered: {name}")

    def format_tool_descriptions(self) -> str:
        """Returns formatted string of all available tools for the prompt."""
        lines = []
        for name, desc in self._descriptions.items():
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def maybe_execute(self, response: str) -> str:
        """
        Scans model output for TOOL: calls and executes them.
        Appends the tool result to the response.

        Example:
            Input:  "Let me calculate that. TOOL: calculator(100 * 1.08)"
            Output: "Let me calculate that. TOOL: calculator(100 * 1.08)\nResult: 108.0"
        """
        matches = find_tool_calls(response)

        if not matches:
            return response

        result_lines = [response]
        for tool_name, args in matches:
            tool_name = tool_name.lower()
            if tool_name not in self._tools:
                result_lines.append(f"[Tool '{tool_name}' not found]")
                continue

            clean_args = args.strip().strip('"').strip("'")
            try:
                result = self._tools[tool_name](clean_args)
                result_lines.append(f"\n[Tool Result: {tool_name}({clean_args})]\n{result}")
                log.info(f"Tool executed: {tool_name}({clean_args}) → {result[:80]}")
            except Exception as e:
                result_lines.append(f"\n[Tool Error: {str(e)}]")

        return "\n".join(result_lines)