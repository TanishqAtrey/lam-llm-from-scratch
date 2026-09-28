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

# Pattern to detect tool calls in model output
TOOL_CALL_PATTERN = re.compile(r"TOOL:\s*(\w+)\(([^)]*)\)", re.IGNORECASE)


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
        matches = TOOL_CALL_PATTERN.findall(response)

        if not matches:
            return response

        result_lines = [response]
        for tool_name, args in matches:
            tool_name = tool_name.lower()
            if tool_name not in self._tools:
                result_lines.append(f"[Tool '{tool_name}' not found]")
                continue

            try:
                result = self._tools[tool_name](args.strip())
                result_lines.append(f"\n[Tool Result: {tool_name}({args.strip()})]\n{result}")
                log.info(f"Tool executed: {tool_name}({args.strip()}) → {result[:80]}")
            except Exception as e:
                result_lines.append(f"\n[Tool Error: {str(e)}]")

        return "\n".join(result_lines)