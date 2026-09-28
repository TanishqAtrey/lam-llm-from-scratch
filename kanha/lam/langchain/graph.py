"""
kanha/lam/langchain/graph.py
LangGraph StateGraph agent for Kanha Tasks LAM.

Replaces the manual while-loop in ``TaskAgent`` with a proper
LangGraph state machine:

    ┌──────────────┐
    │    START      │
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │   generate   │  ← model produces next action text
    └──────┬───────┘
           ▼
    ┌──────────────┐     parse / validate fail
    │    parse     │ ──────────────────────────────┐
    └──────┬───────┘                               │
           │ success                               ▼
           ▼                              ┌────────────────┐
    ┌──────────────┐                      │  retry / abort  │
    │   execute    │                      └────────┬───────┘
    └──────┬───────┘                               │
           ▼                                       ▼
    ┌──────────────┐                          ┌─────────┐
    │   decide     │ ─── terminal ──────────▸ │   END   │
    └──────┬───────┘                          └─────────┘
           │ continue
           ▼
         (loop back to generate)

Usage::

    from kanha.lam.langchain.graph import LangGraphTaskAgent

    agent = LangGraphTaskAgent(model, tokenizer, env)
    result = agent.run("add buy milk for tomorrow")
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict

from langgraph.graph import StateGraph, END

from kanha.lam.actions import parse_action, validate_action, ActionParseError
from kanha.lam.environment import TaskEnvironment
from kanha.lam.langchain.model import KanhaLLM
from kanha.lam.langchain.tools import build_task_tools, execute_tool_by_name
from kanha.lam.trajectory import format_step_prompt


# ── Graph State ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """State that flows through the LangGraph nodes."""
    goal: str
    history: list  # [{action, observation}]
    raw_action: str
    parsed_name: str
    parsed_args: dict
    observation: str
    is_terminal: bool
    step_count: int
    retry_count: int
    max_retries: int
    max_steps: int
    aborted: bool
    abort_reason: str
    final_message: str
    success: bool


# ── Result dataclass (matches TaskAgent's AgentResult) ────────────────────────

@dataclass
class LangGraphAgentResult:
    """Result from a LangGraph agent run.

    Field-compatible with ``kanha.lam.agent.AgentResult`` so callers
    can switch between the two agent implementations transparently.
    """
    goal: str
    steps: list[dict] = field(default_factory=list)
    final_message: Optional[str] = None
    success: bool = False
    aborted: bool = False
    abort_reason: Optional[str] = None
    num_steps: int = 0


# ── Node functions ────────────────────────────────────────────────────────────

def _make_generate_node(llm: KanhaLLM):
    """Create the 'generate' node that calls the model."""

    def generate_node(state: AgentState) -> dict:
        prompt = format_step_prompt(state["goal"], state["history"])

        temp = 0.0 if state["retry_count"] == 0 else 0.3
        raw = llm.generate_action(prompt, temperature=temp)

        return {"raw_action": raw}

    return generate_node


def _make_parse_node():
    """Create the 'parse' node that parses and validates the action."""

    def parse_node(state: AgentState) -> dict:
        raw = state["raw_action"]

        if not raw:
            return {
                "parsed_name": "",
                "parsed_args": {},
                "retry_count": state["retry_count"] + 1,
            }

        try:
            name, args = parse_action(raw)
        except ActionParseError:
            return {
                "parsed_name": "",
                "parsed_args": {},
                "retry_count": state["retry_count"] + 1,
            }

        valid, error_msg = validate_action(name, args)
        if not valid:
            return {
                "parsed_name": "",
                "parsed_args": {},
                "retry_count": state["retry_count"] + 1,
            }

        return {
            "parsed_name": name,
            "parsed_args": args,
            "retry_count": 0,  # reset on success
        }

    return parse_node


def _make_execute_node(env: TaskEnvironment):
    """Create the 'execute' node that runs the action via the environment.

    Uses ``env.step()`` rather than calling tools directly, so that
    the environment's internal history is kept in sync.
    """

    def execute_node(state: AgentState) -> dict:
        raw_action = state["raw_action"]

        # Execute through the environment (updates env._history)
        observation, is_terminal = env.step(raw_action)

        new_history = state["history"] + [
            {"action": raw_action, "observation": observation}
        ]

        result: dict = {
            "observation": observation,
            "is_terminal": is_terminal,
            "step_count": state["step_count"] + 1,
            "history": new_history,
        }

        if is_terminal:
            result["final_message"] = observation
            result["success"] = True

        return result

    return execute_node



# ── Edge conditions ───────────────────────────────────────────────────────────

def _after_parse(state: AgentState) -> str:
    """Decide what to do after parsing.

    - If parsed_name is empty → parse failed → check retries.
    - Otherwise → execute.
    """
    if state["parsed_name"]:
        return "execute"

    if state["retry_count"] >= state["max_retries"]:
        return "abort"

    return "retry"


def _after_execute(state: AgentState) -> str:
    """Decide what to do after execution.

    - Terminal action → end
    - Step limit → abort
    - Loop detected → abort
    - Otherwise → continue generating
    """
    if state.get("is_terminal"):
        return "end"

    if state["step_count"] >= state["max_steps"]:
        return "abort_limit"

    # Loop detection: last 3 actions identical
    history = state["history"]
    if len(history) >= 3:
        last_3 = [h.get("action") for h in history[-3:]]
        if last_3[0] == last_3[1] == last_3[2]:
            return "abort_loop"

    return "continue"


# ── Abort nodes ───────────────────────────────────────────────────────────────

def _abort_parse_node(state: AgentState) -> dict:
    return {
        "aborted": True,
        "abort_reason": "Action generation failed after retries",
    }


def _abort_limit_node(state: AgentState) -> dict:
    return {
        "aborted": True,
        "abort_reason": "step limit reached",
    }


def _abort_loop_node(state: AgentState) -> dict:
    return {
        "aborted": True,
        "abort_reason": "action loop detected",
    }


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_agent_graph(
    llm: KanhaLLM,
    env: TaskEnvironment,
) -> StateGraph:
    """Build a compiled LangGraph ``StateGraph`` for the agent loop.

    Args:
        llm: The KanhaLLM wrapper.
        env: The TaskEnvironment (actions execute through env.step()).

    Returns:
        A compiled LangGraph graph that accepts ``AgentState`` and
        runs the full generate → parse → execute loop.
    """
    graph = StateGraph(AgentState)

    # ── Nodes ──
    graph.add_node("generate", _make_generate_node(llm))
    graph.add_node("parse", _make_parse_node())
    graph.add_node("execute", _make_execute_node(env))
    graph.add_node("abort_parse", _abort_parse_node)
    graph.add_node("abort_limit", _abort_limit_node)
    graph.add_node("abort_loop", _abort_loop_node)


    # ── Edges ──
    graph.set_entry_point("generate")

    # generate → parse (always)
    graph.add_edge("generate", "parse")

    # parse → execute | retry (back to generate) | abort
    graph.add_conditional_edges(
        "parse",
        _after_parse,
        {
            "execute": "execute",
            "retry": "generate",     # retry with higher temperature
            "abort": "abort_parse",
        },
    )

    # execute → continue (generate) | end | abort_limit | abort_loop
    graph.add_conditional_edges(
        "execute",
        _after_execute,
        {
            "continue": "generate",
            "end": END,
            "abort_limit": "abort_limit",
            "abort_loop": "abort_loop",
        },
    )

    # abort nodes → END
    graph.add_edge("abort_parse", END)
    graph.add_edge("abort_limit", END)
    graph.add_edge("abort_loop", END)

    return graph.compile()


# ── High-level agent class ────────────────────────────────────────────────────

class LangGraphTaskAgent:
    """Drop-in replacement for ``kanha.lam.agent.TaskAgent``
    that uses LangGraph under the hood.

    Usage::

        agent = LangGraphTaskAgent(model, tokenizer, env)
        result = agent.run("add buy milk for tomorrow")
        print(result.final_message)
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        environment: TaskEnvironment,
        max_steps: int = 8,
        max_retries: int = 3,
    ):
        self.llm = KanhaLLM(model=model, tokenizer=tokenizer)
        self.env = environment
        self.tools = build_task_tools(environment.store)
        self.max_steps = max_steps
        self.max_retries = max_retries

        self.graph = build_agent_graph(self.llm, self.env)


    def run(self, goal: str) -> LangGraphAgentResult:
        """Execute a user goal through the LangGraph agent loop.

        Args:
            goal: Natural language user request.

        Returns:
            A ``LangGraphAgentResult`` with the same interface as
            ``AgentResult`` from the base agent.
        """
        # Reset the environment's trajectory tracking
        self.env._history = []

        initial_state: AgentState = {
            "goal": goal,
            "history": [],
            "raw_action": "",
            "parsed_name": "",
            "parsed_args": {},
            "observation": "",
            "is_terminal": False,
            "step_count": 0,
            "retry_count": 0,
            "max_retries": self.max_retries,
            "max_steps": self.max_steps,
            "aborted": False,
            "abort_reason": "",
            "final_message": "",
            "success": False,
        }

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        # Convert graph state → AgentResult
        result = LangGraphAgentResult(goal=goal)
        result.success = final_state.get("success", False)
        result.aborted = final_state.get("aborted", False)
        result.abort_reason = final_state.get("abort_reason") or None
        result.final_message = final_state.get("final_message") or None
        result.num_steps = final_state.get("step_count", 0)

        # Reconstruct step records from history
        for i, h in enumerate(final_state.get("history", [])):
            result.steps.append({
                "action": h["action"],
                "observation": h["observation"],
                "valid": True,
                "step_num": i + 1,
            })

        return result
