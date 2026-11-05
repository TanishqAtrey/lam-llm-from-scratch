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
    confidence: float
    confidence_threshold: float
    confidence_scores: list[float]
    low_confidence_count: int
    retry_history: list[dict]
    temperature: Optional[float]


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
    confidence_scores: list[float] = field(default_factory=list)
    low_confidence_count: int = 0
    retry_history: list[dict] = field(default_factory=list)


# ── Node functions ────────────────────────────────────────────────────────────

def _make_generate_node(llm: KanhaLLM):
    """Create the 'generate' node that calls the model."""

    def generate_node(state: AgentState) -> dict:
        prompt = format_step_prompt(state["goal"], state["history"])

        temp = state.get("temperature")
        if state["retry_count"] == 0:
            gen_temp = temp if temp is not None else 0.0
            k = llm.top_k
        else:
            gen_temp = max(temp if temp is not None else 0.0, 0.3)
            k = max(llm.top_k, 50)
        raw, confidence = llm.generate_action_with_confidence(prompt, temperature=gen_temp, top_k=k)

        return {"raw_action": raw, "confidence": confidence}

    return generate_node


def _make_parse_node():
    """Create the 'parse' node that parses and validates the action."""

    def parse_node(state: AgentState) -> dict:
        raw = state["raw_action"]
        confidence = state.get("confidence", 0.0)
        conf_threshold = state.get("confidence_threshold", 0.3)
        retry_hist = list(state.get("retry_history", []))
        low_conf = state.get("low_confidence_count", 0)

        if not raw:
            retry_hist.append({
                "action": "",
                "observation": "Empty action generated",
                "valid": False,
                "step_num": state["step_count"] + 1,
                "confidence": 0.0,
            })
            return {
                "parsed_name": "",
                "parsed_args": {},
                "retry_count": state["retry_count"] + 1,
                "retry_history": retry_hist,
                "low_confidence_count": low_conf,
            }

        # Confidence gate: refuse if model is too uncertain
        if confidence < conf_threshold:
            error_msg = (
                f"Low confidence ({confidence:.2f} < {conf_threshold}) "
                f"— action '{raw}' not executed. "
                f"Please clarify the goal or try a simpler request."
            )
            retry_hist.append({
                "action": raw,
                "observation": error_msg,
                "valid": False,
                "step_num": state["step_count"] + 1,
                "confidence": confidence,
            })
            return {
                "parsed_name": "",
                "parsed_args": {},
                "retry_count": state["retry_count"] + 1,
                "retry_history": retry_hist,
                "low_confidence_count": low_conf + 1,
            }

        try:
            name, args = parse_action(raw)
        except ActionParseError as e:
            retry_hist.append({
                "action": raw,
                "observation": f"Parse error: {str(e)}",
                "valid": False,
                "step_num": state["step_count"] + 1,
                "confidence": confidence,
            })
            return {
                "parsed_name": "",
                "parsed_args": {},
                "retry_count": state["retry_count"] + 1,
                "retry_history": retry_hist,
                "low_confidence_count": low_conf,
            }

        valid, error_msg = validate_action(name, args)
        if not valid:
            retry_hist.append({
                "action": raw,
                "observation": f"Validation error: {error_msg}",
                "valid": False,
                "step_num": state["step_count"] + 1,
                "confidence": confidence,
            })
            return {
                "parsed_name": "",
                "parsed_args": {},
                "retry_count": state["retry_count"] + 1,
                "retry_history": retry_hist,
                "low_confidence_count": low_conf,
            }

        return {
            "parsed_name": name,
            "parsed_args": args,
            "retry_count": 0,  # reset on success
            "retry_history": retry_hist,
            "low_confidence_count": low_conf,
        }

    return parse_node


def _make_execute_node(env: TaskEnvironment):
    """Create the 'execute' node that runs the action via the environment.

    Uses ``env.step()`` rather than calling tools directly, so that
    the environment's internal history is kept in sync.
    """

    def execute_node(state: AgentState) -> dict:
        raw_action = state["raw_action"]
        confidence = state.get("confidence", 0.0)

        # Execute through the environment (updates env._history)
        observation, is_terminal = env.step(raw_action)

        new_history = state["history"] + [
            {"action": raw_action, "observation": observation, "confidence": confidence}
        ]
        conf_scores = list(state.get("confidence_scores", [])) + [confidence]

        result: dict = {
            "observation": observation,
            "is_terminal": is_terminal,
            "step_count": state["step_count"] + 1,
            "history": new_history,
            "confidence_scores": conf_scores,
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

    # Loop detection parity with TaskAgent:
    # 1. 3 identical consecutive actions (A -> A -> A)
    # 2. 2-step alternating cycle repeated twice (A -> B -> A -> B)
    # 3. 3-step repeating cycle repeated twice (A -> B -> C -> A -> B -> C)
    history = state.get("history", [])
    actions = [h.get("action", "").strip() for h in history if isinstance(h, dict) and "action" in h]

    if len(actions) >= 3 and actions[-1] and actions[-1] == actions[-2] == actions[-3]:
        return "abort_loop"

    if len(actions) >= 4 and actions[-1] and actions[-2]:
        if actions[-4] == actions[-2] and actions[-3] == actions[-1]:
            return "abort_loop"

    if len(actions) >= 6 and all(actions[-i] for i in range(1, 4)):
        if actions[-6:-3] == actions[-3:]:
            return "abort_loop"

    return "continue"


# ── Abort nodes ───────────────────────────────────────────────────────────────

def _abort_parse_node(state: AgentState) -> dict:
    retry_hist = state.get("retry_history", [])
    if retry_hist:
        last_obs = retry_hist[-1].get("observation", "")
        abort_reason = f"Action generation failed after retries: {last_obs}"
    else:
        abort_reason = "Action generation failed after retries"
    return {
        "aborted": True,
        "abort_reason": abort_reason,
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
        max_steps: Optional[int] = None,
        max_retries: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
    ):
        from kanha.utils.config import cfg
        lam_cfg = getattr(cfg, "lam", None)
        self.max_steps = max_steps if max_steps is not None else (getattr(lam_cfg, "max_steps_per_trajectory", 8) if lam_cfg else 8)
        self.max_retries = max_retries if max_retries is not None else (getattr(lam_cfg, "max_retries_on_invalid", 3) if lam_cfg else 3)
        self.confidence_threshold = (
            confidence_threshold if confidence_threshold is not None
            else (getattr(lam_cfg, "confidence_threshold", 0.3) if lam_cfg else 0.3)
        )
        self.llm = KanhaLLM(model=model, tokenizer=tokenizer)
        self.env = environment
        self.tools = build_task_tools(environment.store)

        self.graph = build_agent_graph(self.llm, self.env)


    def run(self, goal: str, temperature: Optional[float] = None) -> LangGraphAgentResult:
        """Execute a user goal through the LangGraph agent loop.

        Args:
            goal: Natural language user request.
            temperature: Optional generation temperature override.

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
            "confidence": 0.0,
            "confidence_threshold": self.confidence_threshold,
            "confidence_scores": [],
            "low_confidence_count": 0,
            "retry_history": [],
            "temperature": temperature,
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
        result.confidence_scores = final_state.get("confidence_scores", [])
        result.low_confidence_count = final_state.get("low_confidence_count", 0)
        result.retry_history = final_state.get("retry_history", [])

        # Reconstruct step records from history
        for i, h in enumerate(final_state.get("history", [])):
            result.steps.append({
                "action": h["action"],
                "observation": h["observation"],
                "valid": True,
                "step_num": i + 1,
                "confidence": h.get("confidence", 0.0),
            })

        # If aborted on parse, append the failed step just like TaskAgent does
        if result.aborted and final_state.get("parsed_name") == "" and result.retry_history:
            last_retry = result.retry_history[-1]
            result.steps.append({
                "action": last_retry.get("action", ""),
                "observation": last_retry.get("observation", ""),
                "valid": False,
                "step_num": result.num_steps + 1,
                "confidence": last_retry.get("confidence", 0.0),
            })

        return result
